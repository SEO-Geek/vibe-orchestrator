"""
Perplexity Research Integration

Provides research capabilities via Perplexity API for:
- Technical documentation lookup
- Best practices research
- Problem solving assistance
"""

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from openai import AsyncOpenAI, OpenAIError

from vibe.exceptions import ResearchError

logger = logging.getLogger(__name__)

# Perplexity API configuration
PERPLEXITY_BASE_URL = "https://api.perplexity.ai"
DEFAULT_MODEL = "sonar"  # Updated 2025 - old llama-3.1-sonar-* models deprecated


@dataclass
class ResearchResult:
    """Result from a Perplexity research query."""

    query: str
    answer: str
    citations: list[str] = field(default_factory=list)
    model: str = ""
    created_at: datetime = field(default_factory=datetime.now)


class PerplexityClient:
    """
    Client for Perplexity API research queries.

    Uses the Perplexity API (OpenAI-compatible) to perform
    online research for technical questions.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
    ):
        """
        Initialize Perplexity client.

        Args:
            api_key: Perplexity API key (or from PERPLEXITY_API_KEY env)
            model: Model to use for research
        """
        # Don't store api_key as instance variable for security
        resolved_key = api_key or os.environ.get("PERPLEXITY_API_KEY", "")
        self.model = model

        if not resolved_key:
            logger.warning("PERPLEXITY_API_KEY not set - research will be unavailable")
            self._client = None
        else:
            self._client = AsyncOpenAI(
                api_key=resolved_key,  # Only passed to client, not stored
                base_url=PERPLEXITY_BASE_URL,
            )

        self.request_count = 0

    @property
    def is_available(self) -> bool:
        """Check if Perplexity is available."""
        return self._client is not None

    async def research(
        self,
        query: str,
        context: str = "",
        max_tokens: int = 1024,
    ) -> ResearchResult:
        """
        Perform a research query.

        Args:
            query: The research question
            context: Optional context about the project
            max_tokens: Maximum response length

        Returns:
            ResearchResult with answer and citations

        Raises:
            ResearchError: If research fails
        """
        if not self._client:
            raise ResearchError("Perplexity API key not configured")

        # Build the research prompt
        messages = []

        if context:
            messages.append({
                "role": "system",
                "content": f"You are a technical research assistant. Context: {context}",
            })

        messages.append({
            "role": "user",
            "content": query,
        })

        try:
            response = await self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
            )

            self.request_count += 1

            if not response.choices:
                raise ResearchError("Empty response from Perplexity")

            answer = response.choices[0].message.content or ""

            # Extract citations if present (Perplexity format)
            citations = []
            if hasattr(response, "citations"):
                citations = response.citations or []

            return ResearchResult(
                query=query,
                answer=answer,
                citations=citations,
                model=response.model or self.model,
            )

        except OpenAIError as e:
            raise ResearchError(f"Perplexity API error: {e}")
        except Exception as e:
            raise ResearchError(f"Research failed: {e}")

    async def research_for_task(
        self,
        task_description: str,
        project_context: str,
    ) -> ResearchResult:
        """
        Research best practices for a specific task.

        Args:
            task_description: The task to research
            project_context: Context about the project

        Returns:
            ResearchResult with recommendations
        """
        query = f"""Research best practices and implementation approaches for:

Task: {task_description}

Project context: {project_context}

Provide:
1. Recommended approach
2. Key considerations
3. Common pitfalls to avoid
4. Relevant documentation links"""

        return await self.research(query, context=project_context)

    async def lookup_error(
        self,
        error_message: str,
        stack_trace: str = "",
        context: str = "",
    ) -> ResearchResult:
        """
        Research a specific error.

        Args:
            error_message: The error message
            stack_trace: Optional stack trace
            context: Project context

        Returns:
            ResearchResult with solutions
        """
        query = f"""Help debug this error:

Error: {error_message}
"""
        if stack_trace:
            query += f"""
Stack trace:
{stack_trace[:500]}
"""
        query += """
Provide:
1. What causes this error
2. How to fix it
3. How to prevent it"""

        return await self.research(query, context=context)

    async def lookup_library(
        self,
        library_name: str,
        question: str,
    ) -> ResearchResult:
        """
        Research a specific library or framework.

        Args:
            library_name: Name of the library
            question: Specific question about it

        Returns:
            ResearchResult with documentation
        """
        query = f"""For the {library_name} library/framework:

{question}

Provide current best practices and code examples."""

        return await self.research(query)

    def get_stats(self) -> dict[str, Any]:
        """Get usage statistics."""
        return {
            "available": self.is_available,
            "model": self.model,
            "request_count": self.request_count,
        }
