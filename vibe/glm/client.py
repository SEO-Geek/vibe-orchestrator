"""
GLM Client - OpenRouter API wrapper for GLM-4.7

Provides async interface to GLM-4.7 via OpenRouter API.
Uses OpenAI-compatible API with custom base URL.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, AsyncIterator

from openai import AsyncOpenAI, OpenAIError

from vibe.exceptions import GLMConnectionError, GLMRateLimitError, GLMResponseError
from vibe.glm.parser import parse_task_list, parse_review_result
from vibe.glm.prompts import (
    SUPERVISOR_SYSTEM_PROMPT,
    REVIEWER_SYSTEM_PROMPT,
    TASK_DECOMPOSITION_PROMPT,
)

logger = logging.getLogger(__name__)

# OpenRouter API configuration
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "z-ai/glm-4.7"  # GLM-4.7 (latest) via OpenRouter


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

        except asyncio.TimeoutError:
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
    ) -> GLMResponse:
        """
        Send a chat request to GLM.

        Args:
            system_prompt: System prompt for the conversation
            messages: List of message dicts with 'role' and 'content'
            temperature: Override default temperature
            max_tokens: Override default max tokens

        Returns:
            GLMResponse with content and metadata

        Raises:
            GLMConnectionError: If connection fails
            GLMRateLimitError: If rate limited
            GLMResponseError: If response is invalid
        """
        all_messages = [{"role": "system", "content": system_prompt}] + messages

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
                raise GLMResponseError("No message in GLM response", {"finish_reason": choice.finish_reason})

            content = choice.message.content or ""

            return GLMResponse(
                content=content,
                model=response.model or self.model,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0,
                },
                finish_reason=choice.finish_reason or "",
            )

        except OpenAIError as e:
            error_msg = str(e)
            if "rate_limit" in error_msg.lower():
                raise GLMRateLimitError(f"Rate limited: {error_msg}")
            elif "authentication" in error_msg.lower():
                raise GLMConnectionError(f"Authentication failed: {error_msg}")
            else:
                raise GLMConnectionError(f"API error: {error_msg}")
        except Exception as e:
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
    ) -> list[dict[str, Any]]:
        """
        Have GLM decompose a user request into atomic tasks.

        Args:
            user_request: The user's request
            project_context: Context about the project (starmap, recent changes, etc.)

        Returns:
            List of task dictionaries with id, description, files, constraints

        Raises:
            GLMResponseError: If response cannot be parsed
        """
        # Build the decomposition prompt
        prompt = TASK_DECOMPOSITION_PROMPT.format(
            user_request=user_request,
            project_context=project_context,
        )

        response = await self.chat(
            system_prompt=SUPERVISOR_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,  # Lower temperature for structured output
        )

        try:
            tasks = parse_task_list(response.content)
            logger.info(f"Decomposed request into {len(tasks)} tasks")
            return tasks
        except Exception as e:
            raise GLMResponseError(
                f"Failed to parse task decomposition: {e}",
                {"response": response.content[:500]},
            )

    async def review_changes(
        self,
        task_description: str,
        changes_diff: str,
        claude_summary: str,
    ) -> dict[str, Any]:
        """
        Have GLM review Claude's code changes.

        Args:
            task_description: The original task description
            changes_diff: Git diff or file changes
            claude_summary: Claude's summary of what was done

        Returns:
            Review result with approved (bool), issues (list), feedback (str)

        Raises:
            GLMResponseError: If response cannot be parsed
        """
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

        response = await self.chat(
            system_prompt=REVIEWER_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": review_prompt}],
            temperature=0.1,
        )

        try:
            result = parse_review_result(response.content)
            logger.info(f"Review result: {'APPROVED' if result['approved'] else 'REJECTED'}")
            return result
        except Exception as e:
            raise GLMResponseError(
                f"Failed to parse review result: {e}",
                {"response": response.content[:500]},
            )

    async def ask_clarification(
        self,
        user_request: str,
        project_context: str,
    ) -> str | None:
        """
        Check if GLM needs clarification before proceeding.

        Args:
            user_request: The user's request
            project_context: Context about the project

        Returns:
            Clarification question if needed, None if request is clear
        """
        prompt = f"""The user wants: {user_request}

Project context:
{project_context}

If the request is clear and you can decompose it into tasks, respond with just: CLEAR

If you need clarification, ask ONE specific question to help decompose this into tasks.
Keep your question brief and focused."""

        response = await self.chat(
            system_prompt=SUPERVISOR_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )

        content = response.content.strip()
        if content.upper() == "CLEAR" or content.upper().startswith("CLEAR"):
            return None

        return content

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
