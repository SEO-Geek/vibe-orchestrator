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

        # OpenAI-compatible client for OpenRouter
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=OPENROUTER_BASE_URL,
            default_headers={
                "HTTP-Referer": "https://github.com/SEO-Geek/vibe-orchestrator",
                "X-Title": "Vibe Orchestrator",
            },
        )

        # Usage tracking
        self.total_tokens_used = 0
        self.request_count = 0

        # Conversation history for context
        self._conversation: list[ConversationMessage] = []

        # Circuit breaker state
        self._consecutive_failures = 0
        self._circuit_open_until: float | None = None

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
            response = await self._client.chat.completions.create(
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
    ) -> GeminiResponse:
        """
        Send a chat request to Gemini.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Override default temperature
            max_tokens: Override default max_tokens

        Returns:
            GeminiResponse with content and metadata

        Raises:
            GeminiConnectionError: If connection fails
            GeminiRateLimitError: If rate limited
            GeminiResponseError: If response is invalid
        """
        if self._is_circuit_open():
            raise GeminiConnectionError("Gemini circuit breaker is open")

        try:
            response = await asyncio.wait_for(
                self._client.chat.completions.create(
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

            # Track usage
            if response.usage:
                self.total_tokens_used += response.usage.total_tokens
            self.request_count += 1

            self._record_success()

            return GeminiResponse(
                content=choice.message.content or "",
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
            raise GeminiConnectionError(f"Gemini timed out after {DEFAULT_TIMEOUT}s")
        except OpenAIError as e:
            self._record_failure()
            error_msg = str(e)
            if "rate" in error_msg.lower():
                raise GeminiRateLimitError(f"Rate limited: {error_msg}")
            elif "auth" in error_msg.lower():
                raise GeminiConnectionError(f"Authentication failed: {error_msg}")
            else:
                raise GeminiConnectionError(f"API error: {error_msg}")
        except GeminiResponseError:
            self._record_failure()
            raise
        except Exception as e:
            self._record_failure()
            raise GeminiConnectionError(f"Unexpected error: {e}")

    async def decompose_task(
        self,
        user_request: str,
        project_context: str = "",
        recent_tasks: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Decompose a user request into atomic tasks for Claude.

        This is Gemini's PRIMARY role as orchestrator - understanding
        what the user wants and breaking it into executable steps.

        Args:
            user_request: What the user wants to accomplish
            project_context: STARMAP, CLAUDE.md, and memory context
            recent_tasks: Recently completed tasks for context

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
        )

        messages = [
            {"role": "system", "content": ORCHESTRATOR_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        try:
            response = await self.chat(messages)
            content = response.content

            # Extract JSON from response
            import json

            # Try to find JSON array in response
            json_match = re.search(r'\[[\s\S]*\]', content)
            if json_match:
                tasks = json.loads(json_match.group())
                if tasks:
                    logger.info(f"Gemini decomposed request into {len(tasks)} tasks")
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
            response = await self.chat(messages, max_tokens=500)
            content = response.content.strip().lower()

            # Check if Gemini wants to ask a question
            if "?" in response.content and len(response.content) < 500:
                # Gemini is asking a clarifying question
                return {
                    "needs_clarification": True,
                    "question": response.content,
                }

            # No clarification needed
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
            stream = await self._client.chat.completions.create(
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

    client = GeminiClient(api_key)

    try:
        # Check if there's already a running event loop
        asyncio.get_running_loop()  # Raises RuntimeError if no loop
    except RuntimeError:
        # No running loop - simple case, just use asyncio.run()
        return asyncio.run(client.ping())

    # EDGE CASE: If called from within an async context (e.g., pytest-asyncio),
    # we can't use asyncio.run(). Spawn a thread with its own event loop instead.
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(asyncio.run, client.ping())
        return future.result(timeout=timeout + 5)
