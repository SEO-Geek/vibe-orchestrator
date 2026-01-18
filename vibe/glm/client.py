"""
GLM Client - OpenRouter API wrapper for GLM-4.7

Provides async interface to GLM-4.7 via OpenRouter API.
Uses OpenAI-compatible API with custom base URL.
"""

import asyncio
import json
import logging
import re
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from openai import AsyncOpenAI, OpenAIError

from vibe.exceptions import GLMConnectionError, GLMRateLimitError, GLMResponseError
from vibe.glm.parser import parse_review_result, parse_task_list
from vibe.glm.prompts import (
    ANALYZE_REVIEW_PROMPT,
    CODE_REVIEW_PROMPT,
    CODE_WRITE_REVIEW_PROMPT,
    DEBUG_REVIEW_PROMPT,
    DEBUG_TASK_PROMPT,
    REVIEWER_SYSTEM_PROMPT,
    SUPERVISOR_SYSTEM_PROMPT,
    TASK_DECOMPOSITION_PROMPT,
    TEST_REVIEW_PROMPT,
)
from vibe.logging import (
    GLMLogEntry,
    get_project_name,
    get_session_id,
    glm_logger,
    now_iso,
)

logger = logging.getLogger(__name__)

# Keywords that indicate investigation/exploration - ALWAYS delegate, never clarify
# Rationale: Investigation tasks benefit from Claude exploring the codebase to find answers.
# Asking clarifying questions for these wastes time - Claude can gather context itself.
INVESTIGATION_KEYWORDS = re.compile(
    r"\b(check|debug|investigate|find|search|look|review|analyze|test|verify|"
    r"examine|inspect|diagnose|troubleshoot|explore|what\'s wrong|why is|"
    r"how does|trace|profile|benchmark|audit|scan|monitor)\b",
    re.IGNORECASE,
)

# API configuration
DEFAULT_TIMEOUT = 240.0  # seconds - allow time for long prompts with context
MAX_RETRIES = 2
RETRY_DELAYS = [1.0, 3.0]  # exponential backoff

# Circuit breaker configuration
# Pattern: After N consecutive failures, stop calling GLM to prevent cascading failures
# and allow the service time to recover. This keeps the orchestrator responsive even
# when GLM is down - we fall back to delegation rather than blocking.
CIRCUIT_BREAKER_THRESHOLD = 3  # failures before circuit opens
CIRCUIT_BREAKER_RESET_TIME = 60.0  # seconds before trying again

# OpenRouter API configuration
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "z-ai/glm-4.7"  # GLM-4.7 (latest) via OpenRouter


def is_investigation_request(text: str) -> bool:
    """
    Check if a request is an investigation/exploration task.
    These tasks should ALWAYS be delegated to Claude without clarification.

    Args:
        text: User's request text

    Returns:
        True if this is an investigation task that should skip clarification
    """
    return bool(INVESTIGATION_KEYWORDS.search(text))


@dataclass
class GLMResponse:
    """Response from GLM API call."""

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


class GLMClient:
    """
    Client for interacting with GLM-4 via OpenRouter.

    Provides async methods for:
    - Simple chat completions
    - Task decomposition (breaking user requests into Claude tasks)
    - Code review (evaluating Claude's output)
    - Streaming responses for real-time display
    """

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ):
        """
        Initialize GLM client.

        Args:
            api_key: OpenRouter API key
            model: Model identifier (default: z-ai/glm-4.7)
            temperature: Default sampling temperature (0.0-1.0)
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

        # Track usage for cost monitoring
        self.total_tokens_used = 0
        self.request_count = 0

        # Circuit breaker state
        self._consecutive_failures = 0
        self._circuit_open_until: datetime | None = None

        # Cancellation state (for ESC key handling)
        self._cancelled = False

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
                pass  # Ignore errors during cleanup
            self._client = None
            self._client_loop = None

    def cancel(self) -> None:
        """Cancel any running API calls. Called when user presses ESC."""
        self._cancelled = True
        logger.debug("GLMClient: cancellation requested")

    def reset_cancellation(self) -> None:
        """Reset cancellation flag for new operations."""
        self._cancelled = False

    def _is_circuit_open(self) -> bool:
        """Check if circuit breaker is open (GLM calls should be skipped)."""
        if self._circuit_open_until is None:
            return False
        if datetime.now() >= self._circuit_open_until:
            # Reset circuit breaker
            logger.info("Circuit breaker reset, allowing GLM calls again")
            self._circuit_open_until = None
            self._consecutive_failures = 0
            return False
        return True

    def _record_success(self) -> None:
        """Record successful API call, reset failure counter."""
        self._consecutive_failures = 0

    def _record_failure(self) -> None:
        """Record failed API call, potentially open circuit breaker."""
        self._consecutive_failures += 1
        if self._consecutive_failures >= CIRCUIT_BREAKER_THRESHOLD:
            from datetime import timedelta

            self._circuit_open_until = datetime.now() + timedelta(seconds=CIRCUIT_BREAKER_RESET_TIME)
            logger.warning(
                f"Circuit breaker OPEN after {self._consecutive_failures} failures. "
                f"Skipping GLM for {CIRCUIT_BREAKER_RESET_TIME}s"
            )

    async def ping(self, timeout: float = 10.0) -> tuple[bool, str]:
        """
        Ping GLM to verify API connectivity.

        Args:
            timeout: Request timeout in seconds

        Returns:
            Tuple of (success, message)
        """
        try:
            client = await self._get_client()
            # Simple completion request to verify connectivity
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": "Hello"}],
                    max_tokens=10,
                    temperature=0,
                ),
                timeout=timeout,
            )

            model_used = response.model or self.model
            return True, f"{model_used.split('/')[-1]}"

        except TimeoutError:
            return False, f"timeout ({timeout}s)"
        except OpenAIError as e:
            error_msg = str(e)
            if "rate_limit" in error_msg.lower():
                return False, "rate limited"
            elif "authentication" in error_msg.lower() or "api_key" in error_msg.lower():
                return False, "invalid API key"
            return False, f"API error: {error_msg[:50]}"
        except Exception as e:
            return False, f"connection failed: {str(e)[:50]}"

    async def chat(
        self,
        system_prompt: str,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        method: str = "chat",  # For logging which method called this
    ) -> GLMResponse:
        """
        Send a chat request to GLM.

        Args:
            system_prompt: System prompt for the conversation
            messages: List of message dicts with 'role' and 'content'
            temperature: Override default temperature
            max_tokens: Override default max tokens
            method: Method name for logging (internal use)

        Returns:
            GLMResponse with content and metadata

        Raises:
            GLMConnectionError: If connection fails
            GLMRateLimitError: If rate limited
            GLMResponseError: If response is invalid
        """
        all_messages = [{"role": "system", "content": system_prompt}] + messages

        # Check for cancellation (ESC key)
        if self._cancelled:
            raise asyncio.CancelledError("Operation cancelled by user")

        # Prepare log entry
        request_id = str(uuid.uuid4())
        start_time = time.monotonic()
        user_prompt = messages[-1].get("content", "") if messages else ""

        log_entry = GLMLogEntry(
            timestamp=now_iso(),
            request_id=request_id,
            session_id=get_session_id(),
            method=method,
            system_prompt=system_prompt[:2000],  # Truncate for log size
            user_prompt=user_prompt[:5000],  # Truncate for log size
            temperature=temperature or self.temperature,
            max_tokens=max_tokens or self.max_tokens,
            project_name=get_project_name(),
        )

        try:
            client = await self._get_client()
            response = await client.chat.completions.create(
                model=self.model,
                messages=all_messages,
                temperature=temperature or self.temperature,
                max_tokens=max_tokens or self.max_tokens,
            )

            self.request_count += 1
            if response.usage:
                self.total_tokens_used += response.usage.total_tokens

            # Check for cancellation after API call
            if self._cancelled:
                raise asyncio.CancelledError("Operation cancelled by user")

            # Validate response structure
            if not response.choices:
                raise GLMResponseError("Empty response from GLM", {"model": self.model})

            choice = response.choices[0]
            if not choice.message:
                raise GLMResponseError("No message in GLM response", {"finish_reason": choice.finish_reason})

            content = choice.message.content or ""
            finish_reason = choice.finish_reason or ""
            current_max = max_tokens or self.max_tokens

            # INTELLIGENT RETRY: If response was truncated mid-sentence, the task decomposition
            # or review will be malformed JSON. Rather than fail, double the token limit and retry.
            # Cap at 16K to prevent runaway costs - if GLM needs more than 16K, something is wrong.
            if finish_reason == "length" and current_max < 16384:
                new_max = min(current_max * 2, 16384)
                logger.warning(f"Response truncated at {current_max} tokens, retrying with {new_max}")
                return await self.chat(
                    system_prompt=system_prompt,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=new_max,
                    method=method,
                )

            # Update log entry with response data
            log_entry.response_content = content[:10000]  # Truncate for log size
            log_entry.model = response.model or self.model
            log_entry.finish_reason = finish_reason
            log_entry.prompt_tokens = response.usage.prompt_tokens if response.usage else 0
            log_entry.completion_tokens = response.usage.completion_tokens if response.usage else 0
            log_entry.total_tokens = response.usage.total_tokens if response.usage else 0
            log_entry.latency_ms = int((time.monotonic() - start_time) * 1000)

            # Log the successful call
            glm_logger.info(log_entry.to_json())

            return GLMResponse(
                content=content,
                model=response.model or self.model,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0,
                },
                finish_reason=finish_reason,
            )

        except OpenAIError as e:
            error_msg = str(e)
            log_entry.error = error_msg[:500]
            log_entry.error_type = type(e).__name__
            log_entry.latency_ms = int((time.monotonic() - start_time) * 1000)
            glm_logger.error(log_entry.to_json())

            if "rate_limit" in error_msg.lower():
                raise GLMRateLimitError(f"Rate limited: {error_msg}")
            elif "authentication" in error_msg.lower():
                raise GLMConnectionError(f"Authentication failed: {error_msg}")
            else:
                raise GLMConnectionError(f"API error: {error_msg}")
        except GLMResponseError:
            log_entry.latency_ms = int((time.monotonic() - start_time) * 1000)
            glm_logger.error(log_entry.to_json())
            raise
        except Exception as e:
            log_entry.error = str(e)[:500]
            log_entry.error_type = type(e).__name__
            log_entry.latency_ms = int((time.monotonic() - start_time) * 1000)
            glm_logger.error(log_entry.to_json())
            raise GLMConnectionError(f"Unexpected error: {str(e)}")

    async def chat_stream(
        self,
        system_prompt: str,
        messages: list[dict[str, str]],
        temperature: float | None = None,
    ) -> AsyncIterator[str]:
        """
        Stream a chat response from GLM.

        Args:
            system_prompt: System prompt for the conversation
            messages: List of message dicts
            temperature: Override default temperature

        Yields:
            String chunks as they arrive

        Raises:
            GLMConnectionError: If connection fails
        """
        all_messages = [{"role": "system", "content": system_prompt}] + messages

        try:
            client = await self._get_client()
            stream = await client.chat.completions.create(
                model=self.model,
                messages=all_messages,
                temperature=temperature or self.temperature,
                max_tokens=self.max_tokens,
                stream=True,
            )

            self.request_count += 1

            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except OpenAIError as e:
            raise GLMConnectionError(f"Stream error: {str(e)}")

    async def decompose_task(
        self,
        user_request: str,
        project_context: str,
        use_workflow_engine: bool = False,
        enable_injection: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Have GLM decompose a user request into atomic tasks.

        NEVER FAILS - if GLM returns garbage, create fallback task.
        Uses circuit breaker to prevent cascading failures.

        Args:
            user_request: The user's request
            project_context: Context about the project (starmap, recent changes, etc.)
            use_workflow_engine: If True, post-process with WorkflowEngine for
                                phase expansion and sub-task injection
            enable_injection: If True and use_workflow_engine is True, inject
                            sub-tasks based on task content

        Returns:
            List of task dictionaries with id, description, files, constraints
            If use_workflow_engine=True, returns ExpandedTask.to_dict() format
        """
        # Circuit breaker check - if open, create fallback task immediately
        if self._is_circuit_open():
            logger.warning("Circuit breaker open, creating fallback task for decomposition")
            return [
                {
                    "id": "task-1",
                    "description": f"Handle request (GLM unavailable): {user_request[:200]}",
                    "files": [],
                    "constraints": ["GLM circuit breaker open - proceed with caution"],
                    "success_criteria": "Task addressed to best ability",
                }
            ]

        # Build the decomposition prompt
        prompt = TASK_DECOMPOSITION_PROMPT.format(
            user_request=user_request,
            project_context=project_context,
        )

        try:
            response = await asyncio.wait_for(
                self.chat(
                    system_prompt=SUPERVISOR_SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    method="decompose_task",
                ),
                timeout=DEFAULT_TIMEOUT,
            )

            # Record success
            self._record_success()

            # Parse JSON task list from GLM's response (handles markdown code blocks, etc.)
            tasks = parse_task_list(response.content)
            if not tasks:
                # GLM sometimes returns prose instead of JSON when confused by the request.
                # Fallback to a generic investigation task so work can still proceed.
                logger.warning("GLM returned empty task list, creating fallback task")
                tasks = [
                    {
                        "id": "task-1",
                        "description": f"Investigate and address: {user_request[:200]}",
                        "files": ["investigate relevant files"],
                        "constraints": ["Report findings before making changes"],
                        "success_criteria": "Issue understood and addressed",
                    }
                ]

            # Optional post-processing: expand tasks into phases (plan/implement/verify)
            # and inject sub-tasks for common patterns (e.g., add tests for new functions).
            # This runs AFTER GLM decomposition to avoid overcomplicating the prompt.
            if use_workflow_engine:
                try:
                    from vibe.orchestrator.workflows import WorkflowEngine

                    engine = WorkflowEngine(
                        enable_workflows=True,
                        enable_injection=enable_injection,
                    )
                    expanded = engine.process_tasks(tasks, expand_to_phases=True)
                    # Convert ExpandedTask objects to dicts
                    tasks = [t.to_dict() for t in expanded]
                    logger.info(f"WorkflowEngine expanded to {len(tasks)} tasks/phases")
                except ImportError as e:
                    logger.warning(f"WorkflowEngine not available: {e}")
                except Exception as e:
                    logger.warning(f"WorkflowEngine failed, using raw tasks: {e}")

            logger.info(f"Decomposed request into {len(tasks)} tasks")
            return tasks

        except TimeoutError:
            self._record_failure()
            logger.error(f"GLM timeout after {DEFAULT_TIMEOUT}s in decompose_task")
            raise GLMConnectionError(
                f"GLM timed out after {DEFAULT_TIMEOUT}s - try a shorter request or check API status"
            )
        except Exception as e:
            self._record_failure()
            logger.error(f"GLM decomposition failed: {e}")
            raise GLMConnectionError(f"GLM failed to decompose task: {e}")

    async def review_changes(
        self,
        task_description: str,
        changes_diff: str,
        claude_summary: str,
        task_type: str = "code_write",
        files_changed: str = "",
    ) -> dict[str, Any]:
        """
        Have GLM review Claude's code changes.

        Uses circuit breaker to prevent cascading failures.
        IMPORTANT: If circuit breaker is open, raises GLMConnectionError
        rather than auto-approving (security requirement).

        Args:
            task_description: The original task description
            changes_diff: Git diff or file changes
            claude_summary: Claude's summary of what was done
            task_type: Type of task (code_write, debug, test, research, refactor)
            files_changed: List of files that were changed

        Returns:
            Review result with approved (bool), issues (list), feedback (str)

        Raises:
            GLMConnectionError: If circuit breaker is open
            GLMResponseError: If response cannot be parsed
        """
        # SECURITY: If GLM is down, we must NOT auto-approve changes.
        # Auto-approval would allow untested code through without any oversight.
        # Better to fail the task and require human intervention.
        if self._is_circuit_open():
            logger.error("Circuit breaker open, cannot review changes")
            raise GLMConnectionError(
                "GLM circuit breaker is open - cannot review changes safely. "
                "Task will be marked as failed (never auto-approve without review)."
            )

        # Select task-type-specific prompt for better review accuracy
        # Different task types have different expectations and rejection criteria
        task_type_lower = task_type.lower()
        if task_type_lower in ("test", "ui_test"):
            review_prompt = TEST_REVIEW_PROMPT.format(
                task_description=task_description,
                files_changed=files_changed or "(not specified)",
                diff_content=changes_diff,
                claude_summary=claude_summary,
            )
        elif task_type_lower in ("research", "analyze"):
            review_prompt = ANALYZE_REVIEW_PROMPT.format(
                task_description=task_description,
                files_changed=files_changed or "(not specified)",
                diff_content=changes_diff,
                claude_summary=claude_summary,
            )
        elif task_type_lower == "code_write":
            review_prompt = CODE_WRITE_REVIEW_PROMPT.format(
                task_description=task_description,
                files_changed=files_changed or "(not specified)",
                diff_content=changes_diff,
                claude_summary=claude_summary,
            )
        else:
            # Default: use general CODE_REVIEW_PROMPT for debug, refactor, etc.
            review_prompt = CODE_REVIEW_PROMPT.format(
                task_description=task_description,
                task_type=task_type,
                files_changed=files_changed or "(not specified)",
                diff_content=changes_diff,
                claude_summary=claude_summary,
            )

        try:
            response = await self.chat(
                system_prompt=REVIEWER_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": review_prompt}],
                temperature=0.1,
                max_tokens=8192,  # Higher limit for review responses
                method="review_changes",
            )

            # Record success
            self._record_success()

            result = parse_review_result(response.content)
            logger.info(f"Review result: {'APPROVED' if result['approved'] else 'REJECTED'}")
            return result

        except GLMResponseError:
            self._record_failure()
            raise
        except Exception as e:
            self._record_failure()
            raise GLMResponseError(
                f"Failed to review changes: {e}",
                {"task": task_description[:200]},
            )

    async def ask_clarification(
        self,
        user_request: str,
        project_context: str,
        clarification_count: int = 0,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> str | None:
        """
        Check if GLM needs clarification before proceeding.

        ROBUST: Uses keyword detection, timeout, and retry logic.
        On ANY failure or timeout, returns None (delegate to Claude).

        Args:
            user_request: The user's request
            project_context: Context about the project
            clarification_count: How many times we've already asked (0 = first time)
            timeout: API timeout in seconds (default 30s)

        Returns:
            Clarification question if needed, None to delegate
        """
        # HARD RULE 1: After 1 clarification, force delegation
        # Prevents infinite clarification loops - if one question wasn't enough, let Claude investigate.
        if clarification_count >= 1:
            logger.info("Clarification limit reached, forcing delegation")
            return None

        # HARD RULE 2: Investigation keywords = instant delegation (no API call needed)
        # These tasks ALWAYS benefit from Claude exploring rather than asking questions.
        if INVESTIGATION_KEYWORDS.search(user_request):
            logger.info("Investigation keyword detected, skipping clarification")
            return None

        # HARD RULE 3: Circuit breaker open = skip GLM, delegate
        # Don't block the user waiting for a failing service.
        if self._is_circuit_open():
            logger.info("Circuit breaker open, skipping clarification")
            return None

        prompt = f"""The user wants: {user_request}

Project context:
{project_context}

## RULES FOR DECIDING:

1. **DELEGATE IMMEDIATELY (respond CLEAR) for:**
   - Investigation/debugging tasks ("check", "find", "why is", "what's wrong")
   - Research tasks ("how does X work", "review", "analyze")
   - Testing tasks ("test", "verify", "validate")
   - Any task where Claude can explore the codebase to find answers

2. **Only ask clarification when:**
   - User must choose between mutually exclusive approaches
   - Information is EXTERNAL (API keys, business decisions, preferences)
   - NEVER ask about file locations, error messages, or current state - Claude can find these

3. **If you already asked once, respond CLEAR** - delegate to Claude to investigate

If the request is clear OR is an investigation task, respond with just: CLEAR

If you MUST ask (see rules above), ask ONE brief question about a DECISION the user must make."""

        # Retry logic with timeout - on ANY failure, delegate
        last_error = None
        for attempt in range(MAX_RETRIES + 1):
            try:
                response = await asyncio.wait_for(
                    self.chat(
                        system_prompt=SUPERVISOR_SYSTEM_PROMPT,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.2,
                        method="ask_clarification",
                    ),
                    timeout=timeout,
                )

                # Success - reset circuit breaker
                self._record_success()

                content = response.content.strip()

                # AGGRESSIVE DETECTION: Bias toward delegation (returning None).
                # GLM sometimes returns long explanations, task JSON, or multi-paragraph
                # responses when it should just say "CLEAR". Detect these patterns and
                # proceed to task decomposition rather than showing confusing output to user.
                should_proceed = (
                    content.upper() == "CLEAR"
                    or content.upper().startswith("CLEAR")
                    or '"tasks"' in content  # GLM jumped ahead to task decomposition
                    or '"id":' in content  # Task structure leaked through
                    or "```json" in content  # JSON block = not a question
                    or "delegate" in content.lower()  # Explicit delegation intent
                    or "let me" in content.lower()  # GLM starting to work
                    or len(content) > 500  # Long response != simple clarifying question
                    or content.count("\n") > 5  # Multi-line != simple question
                )

                if should_proceed:
                    logger.info("GLM indicates ready to proceed")
                    return None

                # Only return as clarification if it's genuinely a short question.
                # This prevents verbose responses from becoming confusing clarification prompts.
                if "?" in content and len(content) < 300:
                    return content

                # FAIL-SAFE: When in doubt, delegate to Claude. Better to let Claude
                # investigate and potentially ask its own questions than to block on
                # ambiguous GLM output.
                logger.info("Ambiguous GLM response, defaulting to proceed")
                return None

            except TimeoutError:
                logger.warning(f"GLM timeout (attempt {attempt + 1}/{MAX_RETRIES + 1}), will delegate")
                last_error = "timeout"
            except (GLMConnectionError, GLMRateLimitError, OpenAIError) as e:
                logger.warning(f"GLM error (attempt {attempt + 1}): {e}")
                last_error = str(e)
            except Exception as e:
                logger.warning(f"Unexpected error in ask_clarification: {e}")
                last_error = str(e)

            # Exponential backoff between retries - use the appropriate delay from RETRY_DELAYS
            if attempt < MAX_RETRIES:
                await asyncio.sleep(RETRY_DELAYS[min(attempt, len(RETRY_DELAYS) - 1)])

        # All retries exhausted - record for circuit breaker tracking
        self._record_failure()

        # CRITICAL DESIGN DECISION: Clarification failures ALWAYS delegate to Claude.
        # We never want GLM issues to block the user's workflow. Claude can always
        # ask its own questions or investigate the codebase.
        logger.warning(f"All GLM attempts failed ({last_error}), forcing delegation to Claude")
        return None

    # =========================================================================
    # DEBUG WORKFLOW METHODS
    # =========================================================================

    async def generate_debug_task(
        self,
        problem: str,
        iterations_summary: str = "No previous attempts.",
        hypothesis: str | None = None,
    ) -> dict[str, Any]:
        """
        Generate a specific debugging task for Claude.

        Args:
            problem: The problem being debugged
            iterations_summary: Summary of previous attempts
            hypothesis: Current hypothesis (if any)

        Returns:
            Dict with task details: task, starting_points, what_to_look_for, success_criteria
        """
        prompt = DEBUG_TASK_PROMPT.format(
            problem=problem,
            iterations_summary=iterations_summary,
            hypothesis=hypothesis or "None yet - this is the initial investigation",
        )

        try:
            debug_system_prompt = "You are GLM generating debugging tasks. Output ONLY the JSON object."
            response = await self.chat(
                system_prompt=debug_system_prompt,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=8192,  # Higher limit to prevent truncation
                method="generate_debug_task",
            )
            content = response.content

            # Extract JSON from markdown code blocks if present - GLM often wraps JSON in ```
            json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", content)
            if json_match:
                content = json_match.group(1).strip()

            result = json.loads(content)
            logger.debug(f"Generated debug task: {result.get('task', '')[:100]}")
            return result

        except Exception as e:
            # Debugging must never be blocked by GLM failures - provide a sensible fallback
            # that allows Claude to investigate the problem independently.
            logger.warning(f"Failed to generate debug task: {e}")
            fallback_task = (
                f"Investigate: {problem}. Explore the codebase, find relevant files, and identify the root cause."
            )
            return {
                "task": fallback_task,
                "starting_points": ["Search codebase for relevant keywords"],
                "what_to_look_for": "Error messages, stack traces, related code",
                "success_criteria": "Root cause identified with evidence",
            }

    async def review_debug_iteration(
        self,
        problem: str,
        task: str,
        output: str,
        files_changed: list[str],
        must_preserve: list[str],
        previous_iterations: str = "",
    ) -> dict[str, Any]:
        """
        Review Claude's debugging work and decide next steps.

        Args:
            problem: The original problem
            task: What Claude was asked to do
            output: Claude's output
            files_changed: Files Claude modified
            must_preserve: Features that must still work
            previous_iterations: Summary of previous iterations

        Returns:
            Dict with: approved, is_problem_solved, feedback, next_task
        """
        prompt = DEBUG_REVIEW_PROMPT.format(
            problem=problem,
            task=task,
            output=output[:5000],  # Truncate very long outputs
            files_changed=", ".join(files_changed) if files_changed else "None",
            must_preserve="\n".join(f"- {f}" for f in must_preserve) if must_preserve else "None specified",
            previous_iterations=previous_iterations or "This is the first iteration.",
        )

        try:
            review_system_prompt = (
                "You are GLM reviewing Claude's debugging work. "
                "Output ONLY the JSON object, no explanation or reasoning before it."
            )
            response = await self.chat(
                system_prompt=review_system_prompt,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=8192,  # Higher limit for review responses
                method="review_debug_iteration",
            )
            content = response.content

            # Parse JSON from response
            json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", content)
            if json_match:
                content = json_match.group(1).strip()

            result = json.loads(content)

            # Ensure required fields exist - GLM may omit fields in edge cases.
            # Default to conservative values: not approved, not solved.
            result.setdefault("approved", False)
            result.setdefault("is_problem_solved", False)
            result.setdefault("feedback", "No feedback provided")
            result.setdefault("next_task", None)

            logger.debug(f"Debug review: approved={result['approved']}, solved={result['is_problem_solved']}")
            return result

        except Exception as e:
            logger.warning(f"Failed to review debug iteration: {e}")
            # Fallback: request more information
            feedback_msg = f"GLM review failed ({e}). Please provide more details about what you found."
            return {
                "approved": False,
                "is_problem_solved": False,
                "feedback": feedback_msg,
                "next_task": "Continue investigation and provide detailed findings.",
            }

    def get_usage_stats(self) -> dict[str, Any]:
        """Get usage statistics."""
        return {
            "total_tokens": self.total_tokens_used,
            "request_count": self.request_count,
            "model": self.model,
        }


# Synchronous wrapper for startup validation - needed because CLI startup is synchronous
# but we need to test async API connectivity before entering the main event loop.
def ping_glm_sync(api_key: str, timeout: float = 10.0) -> tuple[bool, str]:
    """
    Synchronous ping for startup validation.

    Handles the case where an event loop may or may not already be running.

    Args:
        api_key: OpenRouter API key
        timeout: Request timeout

    Returns:
        Tuple of (success, message)
    """

    async def _do_ping() -> tuple[bool, str]:
        """Create client, ping, and close within the same async context."""
        client = GLMClient(api_key)
        try:
            return await client.ping(timeout)
        finally:
            await client.close()  # Critical: close before event loop ends

    try:
        # Check if there's already a running event loop
        asyncio.get_running_loop()  # Raises RuntimeError if no loop
    except RuntimeError:
        # No running loop - simple case, just use asyncio.run()
        return asyncio.run(_do_ping())

    # EDGE CASE: If called from within an async context (e.g., pytest-asyncio),
    # we can't use asyncio.run(). Spawn a thread with its own event loop instead.
    # This is slower but avoids "cannot run event loop while another is running".
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(asyncio.run, _do_ping())
        return future.result(timeout=timeout + 5)
