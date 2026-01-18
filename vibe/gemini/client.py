"""
Gemini Client - The Brain/Orchestrator for Vibe

Google's Gemini serves as the intelligent orchestrator that:
1. Understands user requests
2. Decomposes complex tasks into atomic steps
3. Decides when to ask for clarification
4. Coordinates Claude (worker) and GLM (reviewer)

Architecture:
  User → Gemini (brain) → Claude (worker)
                ↓              ↓
              GLM (code review/verification)
"""

import asyncio
import logging
import re
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from openai import AsyncOpenAI, OpenAIError

from vibe.exceptions import (
    GeminiConnectionError,
    GeminiRateLimitError,
    GeminiResponseError,
)
from vibe.gemini.prompts import (
    CLARIFICATION_PROMPT,
    ORCHESTRATOR_SYSTEM_PROMPT,
    TASK_DECOMPOSITION_PROMPT,
)
from vibe.logging import (
    GeminiLogEntry,
    gemini_logger,
    get_project_name,
    get_session_id,
    now_iso,
)

logger = logging.getLogger(__name__)

# Investigation keywords - always delegate to Claude without clarification
INVESTIGATION_KEYWORDS = re.compile(
    r"\b(check|debug|investigate|find|search|look|review|analyze|test|verify|"
    r"examine|inspect|diagnose|troubleshoot|explore|what\'s wrong|why is|"
    r"how does|trace|profile|benchmark|audit|scan|monitor)\b",
    re.IGNORECASE,
)

# API configuration
DEFAULT_TIMEOUT = 120.0  # seconds
MAX_RETRIES = 2
RETRY_DELAYS = [1.0, 3.0]

# Circuit breaker
CIRCUIT_BREAKER_THRESHOLD = 3
CIRCUIT_BREAKER_RESET_TIME = 60.0

# OpenRouter API for Gemini
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "google/gemini-2.0-flash-001"  # Gemini 2.0 Flash - fast and capable


def is_investigation_request(text: str) -> bool:
    """Check if request is an investigation task (skip clarification)."""
    return bool(INVESTIGATION_KEYWORDS.search(text))


@dataclass
class GeminiResponse:
    """Response from Gemini API call."""

    content: str
    model: str
    usage: dict[str, int] = field(default_factory=dict)
    finish_reason: str = ""
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class ConversationMessage:
    """A message in the conversation history."""

    role: str  # 'user', 'assistant', or 'system'
    content: str
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, str]:
        """Convert to API-compatible dict."""
        return {"role": self.role, "content": self.content}


class GeminiClient:
    """
    Client for Google's Gemini via OpenRouter.

    Gemini is the BRAIN of Vibe - it orchestrates everything:
    - Understands what the user wants
    - Breaks complex requests into atomic tasks
    - Decides when Claude should execute
    - Coordinates with GLM for code review

    GLM handles ONLY code review and verification.
    """

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ):
        """
        Initialize Gemini client.

        Args:
            api_key: OpenRouter API key
            model: Model identifier (default: google/gemini-2.0-flash-001)
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens in response
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Store config for lazy client creation (fix for "Event loop is closed" errors)
        self._api_key = api_key
        self._headers = {
            "HTTP-Referer": "https://github.com/SEO-Geek/vibe-orchestrator",
            "X-Title": "Vibe Orchestrator",
        }

        # Lazy-initialized client (created in async context)
        self._client: AsyncOpenAI | None = None
        self._client_loop: asyncio.AbstractEventLoop | None = None

        # Usage tracking
        self.total_tokens_used = 0
        self.request_count = 0

        # Conversation history for context
        self._conversation: list[ConversationMessage] = []

        # Circuit breaker state
        self._consecutive_failures = 0
        self._circuit_open_until: float | None = None

    async def _get_client(self) -> AsyncOpenAI:
        """Get or create AsyncOpenAI client, recreating if event loop changed."""
        current_loop = asyncio.get_running_loop()

        # If loop changed, abandon old client (don't try to close - old loop is dead)
        # This prevents "Event loop is closed" errors when asyncio.run() is called repeatedly
        if self._client is not None and self._client_loop is not current_loop:
            self._client = None
            self._client_loop = None

        # Create client if needed
        if self._client is None:
            self._client = AsyncOpenAI(
                api_key=self._api_key,
                base_url=OPENROUTER_BASE_URL,
                default_headers=self._headers,
            )
            self._client_loop = current_loop

        return self._client

    async def close(self) -> None:
        """Close the client and release resources."""
        if self._client is not None:
            try:
                await self._client.close()
            except Exception:
                pass
            self._client = None
            self._client_loop = None

    def _is_circuit_open(self) -> bool:
        """Check if circuit breaker is open."""
        if self._circuit_open_until is None:
            return False

        if time.time() >= self._circuit_open_until:
            self._circuit_open_until = None
            self._consecutive_failures = 0
            logger.info("Gemini circuit breaker reset")
            return False

        return True

    def _record_success(self) -> None:
        """Record successful call."""
        self._consecutive_failures = 0

    def _record_failure(self) -> None:
        """Record failed call, potentially opening circuit."""
        self._consecutive_failures += 1
        if self._consecutive_failures >= CIRCUIT_BREAKER_THRESHOLD:
            self._circuit_open_until = time.time() + CIRCUIT_BREAKER_RESET_TIME
            logger.warning(
                f"Gemini circuit breaker OPEN - {CIRCUIT_BREAKER_THRESHOLD} failures, "
                f"skipping for {CIRCUIT_BREAKER_RESET_TIME}s"
            )

    async def ping(self) -> str:
        """
        Ping Gemini to verify connectivity.

        Returns:
            Model identifier if successful

        Raises:
            GeminiConnectionError: If connection fails
        """
        try:
            client = await self._get_client()
            response = await client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=10,
            )
            self._record_success()
            return self.model
        except Exception as e:
            self._record_failure()
            raise GeminiConnectionError(f"Ping failed: {e}")

    async def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        method: str = "chat",  # For logging which method called this
        user_request: str = "",  # Original user request for debugging
    ) -> GeminiResponse:
        """
        Send a chat request to Gemini.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Override default temperature
            max_tokens: Override default max_tokens
            method: Method name for logging (internal use)
            user_request: Original user request for debugging prompt quality

        Returns:
            GeminiResponse with content and metadata

        Raises:
            GeminiConnectionError: If connection fails
            GeminiRateLimitError: If rate limited
            GeminiResponseError: If response is invalid
        """
        if self._is_circuit_open():
            raise GeminiConnectionError("Gemini circuit breaker is open")

        # Prepare log entry for comprehensive debugging
        request_id = str(uuid.uuid4())
        start_time = time.monotonic()

        # Extract system and user prompts for logging
        system_prompt = ""
        user_prompt = ""
        for msg in messages:
            if msg.get("role") == "system":
                system_prompt = msg.get("content", "")[:2000]
            elif msg.get("role") == "user":
                user_prompt = msg.get("content", "")[:5000]

        log_entry = GeminiLogEntry(
            timestamp=now_iso(),
            request_id=request_id,
            session_id=get_session_id(),
            method=method,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature or self.temperature,
            max_tokens=max_tokens or self.max_tokens,
            project_name=get_project_name(),
            user_request=user_request[:500] if user_request else "",
        )

        try:
            client = await self._get_client()
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature or self.temperature,
                    max_tokens=max_tokens or self.max_tokens,
                ),
                timeout=DEFAULT_TIMEOUT,
            )

            if not response.choices:
                raise GeminiResponseError("Empty response from Gemini")

            choice = response.choices[0]
            if not choice.message:
                raise GeminiResponseError("No message in Gemini response")

            content = choice.message.content or ""

            # Track usage
            if response.usage:
                self.total_tokens_used += response.usage.total_tokens
            self.request_count += 1

            self._record_success()

            # Update log entry with response data
            log_entry.response_content = content[:10000]  # Truncate for log size
            log_entry.model = response.model or self.model
            log_entry.finish_reason = choice.finish_reason or ""
            log_entry.prompt_tokens = response.usage.prompt_tokens if response.usage else 0
            log_entry.completion_tokens = response.usage.completion_tokens if response.usage else 0
            log_entry.total_tokens = response.usage.total_tokens if response.usage else 0
            log_entry.latency_ms = int((time.monotonic() - start_time) * 1000)
            log_entry.estimate_cost()

            # Log successful call
            gemini_logger.info(log_entry.to_json())

            return GeminiResponse(
                content=content,
                model=response.model,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0,
                },
                finish_reason=choice.finish_reason or "",
            )

        except asyncio.TimeoutError:
            self._record_failure()
            log_entry.error = f"Timeout after {DEFAULT_TIMEOUT}s"
            log_entry.error_type = "TimeoutError"
            log_entry.latency_ms = int((time.monotonic() - start_time) * 1000)
            gemini_logger.error(log_entry.to_json())
            raise GeminiConnectionError(f"Gemini timed out after {DEFAULT_TIMEOUT}s")
        except OpenAIError as e:
            self._record_failure()
            error_msg = str(e)
            log_entry.error = error_msg[:500]
            log_entry.error_type = type(e).__name__
            log_entry.latency_ms = int((time.monotonic() - start_time) * 1000)
            gemini_logger.error(log_entry.to_json())
            if "rate" in error_msg.lower():
                raise GeminiRateLimitError(f"Rate limited: {error_msg}")
            elif "auth" in error_msg.lower():
                raise GeminiConnectionError(f"Authentication failed: {error_msg}")
            else:
                raise GeminiConnectionError(f"API error: {error_msg}")
        except GeminiResponseError:
            self._record_failure()
            log_entry.error = "Invalid response structure"
            log_entry.error_type = "GeminiResponseError"
            log_entry.latency_ms = int((time.monotonic() - start_time) * 1000)
            gemini_logger.error(log_entry.to_json())
            raise
        except Exception as e:
            self._record_failure()
            log_entry.error = str(e)[:500]
            log_entry.error_type = type(e).__name__
            log_entry.latency_ms = int((time.monotonic() - start_time) * 1000)
            gemini_logger.error(log_entry.to_json())
            raise GeminiConnectionError(f"Unexpected error: {e}")

    async def decompose_task(
        self,
        user_request: str,
        project_context: str = "",
        recent_tasks: list[str] | None = None,
        pattern_context: str = "",
    ) -> list[dict[str, Any]]:
        """
        Decompose a user request into atomic tasks for Claude.

        This is Gemini's PRIMARY role as orchestrator - understanding
        what the user wants and breaking it into executable steps.

        Args:
            user_request: What the user wants to accomplish
            project_context: STARMAP, CLAUDE.md, and memory context
            recent_tasks: Recently completed tasks for context
            pattern_context: Historical patterns from previous tasks for adaptive learning

        Returns:
            List of task dicts with description, files, constraints, type
        """
        # Build prompt
        recent_context = ""
        if recent_tasks:
            recent_context = "\n\nRecent tasks:\n" + "\n".join(f"- {t}" for t in recent_tasks[-5:])

        prompt = TASK_DECOMPOSITION_PROMPT.format(
            user_request=user_request,
            project_context=project_context[:50000],  # Cap context size
            recent_context=recent_context,
            pattern_context=pattern_context,
        )

        messages = [
            {"role": "system", "content": ORCHESTRATOR_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        try:
            response = await self.chat(
                messages,
                method="decompose_task",
                user_request=user_request,
            )
            content = response.content

            # Extract JSON from response
            import json

            # Try to find JSON array in response
            json_match = re.search(r'\[[\s\S]*\]', content)
            if json_match:
                tasks = json.loads(json_match.group())
                if tasks:
                    # Log task descriptions for debugging prompt quality
                    task_descriptions = [t.get("description", "")[:100] for t in tasks]
                    logger.info(f"Gemini decomposed request into {len(tasks)} tasks")
                    logger.debug(f"Task descriptions: {task_descriptions}")
                    return tasks

            # Fallback: create single task
            logger.warning("Gemini returned no tasks, creating fallback")
            return [{
                "description": user_request,
                "files": [],
                "constraints": [],
                "type": "code",
            }]

        except Exception as e:
            logger.error(f"Gemini decomposition failed: {e}")
            # Never block on Gemini failure - fallback to single task
            return [{
                "description": f"Handle request (Gemini unavailable): {user_request[:200]}",
                "files": [],
                "constraints": ["Gemini unavailable - proceed with caution"],
                "type": "code",
            }]

    async def check_clarification(
        self,
        user_request: str,
        project_context: str = "",
    ) -> dict[str, Any]:
        """
        Check if Gemini needs clarification before proceeding.

        Args:
            user_request: The user's request
            project_context: Project context

        Returns:
            Dict with 'needs_clarification' bool and 'question' if needed
        """
        # Skip clarification for investigation tasks
        if is_investigation_request(user_request):
            return {"needs_clarification": False}

        prompt = CLARIFICATION_PROMPT.format(
            user_request=user_request,
            project_context=project_context[:20000],
        )

        messages = [
            {"role": "system", "content": ORCHESTRATOR_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        try:
            response = await self.chat(
                messages,
                max_tokens=500,
                method="check_clarification",
                user_request=user_request,
            )
            content = response.content.strip().lower()

            # Check if Gemini wants to ask a question
            if "?" in response.content and len(response.content) < 500:
                # Gemini is asking a clarifying question
                logger.info(f"Gemini requesting clarification: {response.content[:100]}")
                return {
                    "needs_clarification": True,
                    "question": response.content,
                }

            # No clarification needed
            logger.debug("Gemini: No clarification needed")
            return {"needs_clarification": False}

        except Exception as e:
            logger.warning(f"Clarification check failed: {e}, proceeding without")
            return {"needs_clarification": False}

    async def stream_chat(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
    ) -> AsyncIterator[str]:
        """
        Stream a chat response from Gemini.

        Args:
            messages: List of message dicts
            temperature: Override temperature

        Yields:
            Content chunks as they arrive
        """
        if self._is_circuit_open():
            raise GeminiConnectionError("Gemini circuit breaker is open")

        try:
            client = await self._get_client()
            stream = await client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature or self.temperature,
                stream=True,
            )

            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

            self._record_success()

        except Exception as e:
            self._record_failure()
            raise GeminiConnectionError(f"Stream error: {e}")

    def get_usage_stats(self) -> dict[str, Any]:
        """Get usage statistics."""
        return {
            "total_tokens": self.total_tokens_used,
            "request_count": self.request_count,
            "model": self.model,
        }

    def clear_conversation(self) -> None:
        """Clear conversation history."""
        self._conversation.clear()


def ping_gemini_sync(api_key: str, timeout: float = 30.0) -> str:
    """
    Synchronous ping for startup checks.

    Handles both cases:
    - Called from sync context: uses asyncio.run()
    - Called from async context: spawns thread with own event loop

    Args:
        api_key: OpenRouter API key
        timeout: Request timeout in seconds

    Returns:
        Model identifier

    Raises:
        GeminiConnectionError: If ping fails
    """
    import asyncio
    import concurrent.futures

    async def _do_ping() -> str:
        """Create client, ping, and close within the same async context."""
        client = GeminiClient(api_key)
        try:
            return await client.ping()
        finally:
            await client.close()  # Critical: close before event loop ends

    try:
        # Check if there's already a running event loop
        asyncio.get_running_loop()  # Raises RuntimeError if no loop
    except RuntimeError:
        # No running loop - simple case, just use asyncio.run()
        return asyncio.run(_do_ping())

    # EDGE CASE: If called from within an async context (e.g., pytest-asyncio),
    # we can't use asyncio.run(). Spawn a thread with its own event loop.
    # Create client inside the thread to avoid httpx cross-thread issues.
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(asyncio.run, _do_ping())
        return future.result(timeout=timeout)
