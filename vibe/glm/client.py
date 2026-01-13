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
    DEBUG_REVIEW_PROMPT,
    DEBUG_TASK_PROMPT,
    REVIEWER_SYSTEM_PROMPT,
    SUPERVISOR_SYSTEM_PROMPT,
    TASK_DECOMPOSITION_PROMPT,
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
        # Don't store api_key as instance variable for security
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Initialize async OpenAI client with OpenRouter base URL
        self._client = AsyncOpenAI(
            api_key=api_key,  # Only passed to client, not stored
            base_url=OPENROUTER_BASE_URL,
            default_headers={
                "HTTP-Referer": "https://github.com/SEO-Geek/vibe-orchestrator",
                "X-Title": "Vibe Orchestrator",
            },
        )

        # Track usage for cost monitoring
        self.total_tokens_used = 0
        self.request_count = 0

        # Circuit breaker state
        self._consecutive_failures = 0
        self._circuit_open_until: datetime | None = None

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

            self._circuit_open_until = datetime.now() + timedelta(
                seconds=CIRCUIT_BREAKER_RESET_TIME
            )
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
            # Simple completion request to verify connectivity
            response = await asyncio.wait_for(
                self._client.chat.completions.create(
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
            response = await self._client.chat.completions.create(
                model=self.model,
                messages=all_messages,
                temperature=temperature or self.temperature,
                max_tokens=max_tokens or self.max_tokens,
            )

            self.request_count += 1
            if response.usage:
                self.total_tokens_used += response.usage.total_tokens

            # Validate response structure
            if not response.choices:
                raise GLMResponseError("Empty response from GLM", {"model": self.model})

            choice = response.choices[0]
            if not choice.message:
                raise GLMResponseError(
                    "No message in GLM response", {"finish_reason": choice.finish_reason}
                )

            content = choice.message.content or ""
            finish_reason = choice.finish_reason or ""
            current_max = max_tokens or self.max_tokens

            # INTELLIGENT RETRY: If response was truncated, automatically retry with higher limit
            if finish_reason == "length" and current_max < 16384:
                new_max = min(current_max * 2, 16384)  # Double up to 16K max
                logger.warning(
                    f"Response truncated at {current_max} tokens, retrying with {new_max}"
                )
                # Recursive retry with higher limit
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
            stream = await self._client.chat.completions.create(
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

            tasks = parse_task_list(response.content)
            if not tasks:
                # Empty task list - create fallback investigation task
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

            # Post-process with WorkflowEngine if enabled
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

        Returns:
            Review result with approved (bool), issues (list), feedback (str)

        Raises:
            GLMConnectionError: If circuit breaker is open
            GLMResponseError: If response cannot be parsed
        """
        # Circuit breaker check - if open, raise error (never auto-approve)
        if self._is_circuit_open():
            logger.error("Circuit breaker open, cannot review changes")
            raise GLMConnectionError(
                "GLM circuit breaker is open - cannot review changes safely. "
                "Task will be marked as failed (never auto-approve without review)."
            )

        review_prompt = f"""Review this code change:

ORIGINAL TASK:
{task_description}

CHANGES MADE:
```diff
{changes_diff}
```

CLAUDE'S SUMMARY:
{claude_summary}

Evaluate:
1. Does it meet the task requirements?
2. Is it a sustainable solution (not a bush fix)?
3. Are there inline comments for complex logic?
4. Any security/quality issues?

Output JSON: {{"approved": true/false, "issues": [...], "feedback": "..."}}"""

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
        if clarification_count >= 1:
            logger.info("Clarification limit reached, forcing delegation")
            return None

        # HARD RULE 2: Investigation keywords = instant delegation (no API call needed)
        if INVESTIGATION_KEYWORDS.search(user_request):
            logger.info("Investigation keyword detected, skipping clarification")
            return None

        # HARD RULE 3: Circuit breaker open = skip GLM, delegate
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

                # AGGRESSIVE DETECTION: Only return as clarification if it's clearly
                # a SHORT question. Everything else = proceed to task decomposition.

                # Signs that GLM wants to proceed (return None = proceed):
                should_proceed = (
                    content.upper() == "CLEAR"
                    or content.upper().startswith("CLEAR")
                    or '"tasks"' in content  # GLM returned task JSON
                    or '"id":' in content  # Task structure
                    or "```json" in content  # JSON block
                    or "delegate" in content.lower()  # Delegation intent
                    or "let me" in content.lower()  # Starting to work
                    or len(content) > 500  # Long response = not a simple question
                    or content.count("\n") > 5  # Multi-line = not a simple question
                )

                if should_proceed:
                    logger.info("GLM indicates ready to proceed")
                    return None

                # Only return as clarification if it ends with "?" and is short
                if "?" in content and len(content) < 300:
                    return content

                # Default: proceed (fail-safe)
                logger.info("Ambiguous GLM response, defaulting to proceed")
                return None

            except TimeoutError:
                logger.warning(
                    f"GLM timeout (attempt {attempt + 1}/{MAX_RETRIES + 1}), will delegate"
                )
                last_error = "timeout"
            except (GLMConnectionError, GLMRateLimitError, OpenAIError) as e:
                logger.warning(f"GLM error (attempt {attempt + 1}): {e}")
                last_error = str(e)
            except Exception as e:
                logger.warning(f"Unexpected error in ask_clarification: {e}")
                last_error = str(e)

            # Wait before retry (if not last attempt)
            if attempt < MAX_RETRIES:
                await asyncio.sleep(RETRY_DELAYS[min(attempt, len(RETRY_DELAYS) - 1)])

        # All retries failed - record failure for circuit breaker
        self._record_failure()

        # Delegate to Claude (fail-safe)
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
            response = await self.chat(
                system_prompt="You are GLM generating debugging tasks. Output ONLY the JSON object.",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=8192,  # Higher limit to prevent truncation
                method="generate_debug_task",
            )
            content = response.content

            # Parse JSON from response
            json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", content)
            if json_match:
                content = json_match.group(1).strip()

            result = json.loads(content)
            logger.debug(f"Generated debug task: {result.get('task', '')[:100]}")
            return result

        except Exception as e:
            logger.warning(f"Failed to generate debug task: {e}")
            # Fallback: generic investigation task
            return {
                "task": f"Investigate: {problem}. Explore the codebase, find relevant files, and identify the root cause.",
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
            must_preserve="\n".join(f"- {f}" for f in must_preserve)
            if must_preserve
            else "None specified",
            previous_iterations=previous_iterations or "This is the first iteration.",
        )

        try:
            response = await self.chat(
                system_prompt="You are GLM reviewing Claude's debugging work. Output ONLY the JSON object, no explanation or reasoning before it.",
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

            # Ensure required fields
            result.setdefault("approved", False)
            result.setdefault("is_problem_solved", False)
            result.setdefault("feedback", "No feedback provided")
            result.setdefault("next_task", None)

            logger.debug(
                f"Debug review: approved={result['approved']}, solved={result['is_problem_solved']}"
            )
            return result

        except Exception as e:
            logger.warning(f"Failed to review debug iteration: {e}")
            # Fallback: request more information
            return {
                "approved": False,
                "is_problem_solved": False,
                "feedback": f"GLM review failed ({e}). Please provide more details about what you found.",
                "next_task": "Continue investigation and provide detailed findings.",
            }

    def get_usage_stats(self) -> dict[str, Any]:
        """Get usage statistics."""
        return {
            "total_tokens": self.total_tokens_used,
            "request_count": self.request_count,
            "model": self.model,
        }


# Synchronous wrapper for startup validation
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
    client = GLMClient(api_key)

    try:
        # Check if there's already a running event loop
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running loop, safe to use asyncio.run()
        return asyncio.run(client.ping(timeout))

    # If we're here, there's a running loop - use run_until_complete
    # This shouldn't happen in normal CLI usage, but handle it gracefully
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(asyncio.run, client.ping(timeout))
        return future.result(timeout=timeout + 5)
