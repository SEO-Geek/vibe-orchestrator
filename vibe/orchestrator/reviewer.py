"""
GLM Reviewer - Code review gate

Has GLM review Claude's output before accepting changes.
Placeholder for Phase 5 implementation.
"""

from typing import Any

from vibe.state import Task


class Reviewer:
    """
    Reviewer that uses GLM to evaluate Claude's work.

    Review criteria:
    - Task completion: Did Claude complete the requested task?
    - Code quality: Sustainable solution vs bush fix?
    - Documentation: Inline comments present?
    - Scope: Did Claude stay within task boundaries?
    """

    def __init__(self, glm_client: Any):  # Will be GLMClient in Phase 3
        """
        Initialize reviewer.

        Args:
            glm_client: GLM client for API calls
        """
        self.glm_client = glm_client

    async def review(
        self,
        task: Task,
        claude_result: dict[str, Any],
        diff: str | None = None,
    ) -> dict[str, Any]:
        """
        Review Claude's task output.

        Args:
            task: The task that was executed
            claude_result: Claude's execution result
            diff: Git diff of changes (if available)

        Returns:
            Review result:
            {
                "approved": bool,
                "issues": ["List of issues"],
                "feedback": "Feedback for Claude if rejected"
            }
        """
        # TODO: Implement in Phase 5
        raise NotImplementedError("Reviewer not yet implemented - Phase 5")

    def build_review_prompt(
        self,
        task: Task,
        claude_result: dict[str, Any],
        diff: str | None = None,
    ) -> str:
        """
        Build the review prompt for GLM.

        Args:
            task: The task
            claude_result: Claude's result
            diff: Git diff if available

        Returns:
            Formatted review prompt
        """
        parts = [
            "Review this code change:",
            "",
            "ORIGINAL TASK:",
            task.description,
            "",
            "CHANGES MADE:",
            diff or "No diff available",
            "",
            "CLAUDE'S SUMMARY:",
            claude_result.get("result", "No summary provided"),
            "",
            "Evaluate:",
            "1. Does it meet the task requirements?",
            "2. Is it a sustainable solution (not a bush fix)?",
            "3. Are there inline comments for complex logic?",
            "4. Any security/quality issues?",
            "",
            'Output JSON: {"approved": true/false, "issues": [...], "feedback": "..."}',
        ]

        return "\n".join(parts)
