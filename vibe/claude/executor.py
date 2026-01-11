"""
Claude Code Executor

Executes tasks via Claude Code CLI with proper subprocess management,
timeout handling, and output parsing.

Placeholder for Phase 4 implementation - will use claude-agent-sdk.
"""

import asyncio
from dataclasses import dataclass
from typing import Any, Callable

from vibe.exceptions import ClaudeError, ClaudeNotFoundError, ClaudeTimeoutError


# Timeout tiers based on task complexity
TIMEOUT_TIERS = {
    "quick": 30,  # Simple reads, small edits
    "code": 120,  # Normal coding tasks
    "debug": 180,  # Debugging sessions
    "research": 300,  # Research and exploration
}


@dataclass
class TaskResult:
    """Result from a Claude task execution."""

    success: bool
    result: str | None = None
    error: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    file_changes: list[str] | None = None
    cost_usd: float | None = None
    exit_code: int = 0


class ClaudeExecutor:
    """
    Executor for Claude Code CLI tasks.

    Uses claude-agent-sdk for proper async streaming and tool tracking.
    """

    def __init__(
        self,
        project_path: str,
        timeout_tier: str = "code",
        allowed_tools: list[str] | None = None,
    ):
        """
        Initialize Claude executor.

        Args:
            project_path: Working directory for Claude
            timeout_tier: One of 'quick', 'code', 'debug', 'research'
            allowed_tools: List of allowed tool names (default: standard set)
        """
        self.project_path = project_path
        self.timeout = TIMEOUT_TIERS.get(timeout_tier, TIMEOUT_TIERS["code"])
        self.allowed_tools = allowed_tools or [
            "Read",
            "Write",
            "Edit",
            "Bash",
            "Grep",
            "Glob",
        ]

    async def execute(
        self,
        task_description: str,
        files: list[str] | None = None,
        constraints: list[str] | None = None,
        on_progress: Callable[[str], None] | None = None,
    ) -> TaskResult:
        """
        Execute a task via Claude Code.

        Args:
            task_description: What Claude should do
            files: Specific files to work with
            constraints: Constraints for Claude to follow
            on_progress: Callback for progress updates

        Returns:
            TaskResult with success/failure and details

        Raises:
            ClaudeNotFoundError: If Claude CLI not installed
            ClaudeTimeoutError: If task times out
            ClaudeError: For other execution errors
        """
        # TODO: Implement in Phase 4 using claude-agent-sdk
        #
        # The implementation will:
        # 1. Build the prompt from task_description, files, and constraints
        # 2. Call claude-agent-sdk query() with options
        # 3. Stream responses and track tool calls
        # 4. Return structured TaskResult

        raise NotImplementedError("Claude executor not yet implemented - Phase 4")

    def build_prompt(
        self,
        task_description: str,
        files: list[str] | None = None,
        constraints: list[str] | None = None,
    ) -> str:
        """
        Build the prompt for Claude.

        Args:
            task_description: Main task description
            files: Files to work with
            constraints: Constraints to follow

        Returns:
            Formatted prompt string
        """
        parts = [
            "You are working on a specific task. Do ONLY this task.",
            "",
            f"TASK: {task_description}",
        ]

        if files:
            parts.append(f"FILES: {', '.join(files)}")
        else:
            parts.append("FILES: as needed")

        if constraints:
            parts.append("")
            parts.append("CONSTRAINTS:")
            for constraint in constraints:
                parts.append(f"- {constraint}")
        else:
            parts.append("")
            parts.append("CONSTRAINTS:")
            parts.append("- Add inline comments explaining complex logic")
            parts.append("- Follow existing code patterns")
            parts.append("- Do NOT make changes outside the task scope")

        return "\n".join(parts)
