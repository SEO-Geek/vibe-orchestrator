"""
Claude Code Executor

Executes tasks via Claude Code CLI with proper subprocess management,
timeout handling, streaming output parsing, and tool call tracking.
"""

import asyncio
import json
import logging
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

from vibe.exceptions import (
    ClaudeError,
    ClaudeExecutionError,
    ClaudeNotFoundError,
    ClaudeTimeoutError,
)

logger = logging.getLogger(__name__)


# Timeout tiers based on task complexity
TIMEOUT_TIERS = {
    "quick": 30,  # Simple reads, small edits
    "code": 120,  # Normal coding tasks
    "debug": 180,  # Debugging sessions
    "research": 300,  # Research and exploration
}


@dataclass
class ToolCall:
    """A tool call made by Claude during execution."""

    name: str
    input: dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class TaskResult:
    """Result from a Claude task execution."""

    success: bool
    result: str | None = None
    error: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    file_changes: list[str] = field(default_factory=list)
    cost_usd: float = 0.0
    duration_ms: int = 0
    session_id: str = ""
    num_turns: int = 0

    def get_diff_files(self) -> list[str]:
        """Get list of files that were modified (Edit/Write tools)."""
        modified = []
        for call in self.tool_calls:
            if call.name in ("Edit", "Write"):
                file_path = call.input.get("file_path") or call.input.get("path")
                if file_path and file_path not in modified:
                    modified.append(file_path)
        return modified


class ClaudeExecutor:
    """
    Executor for Claude Code CLI tasks.

    Uses subprocess to run Claude with streaming JSON output,
    parses tool calls and results in real-time.
    """

    def __init__(
        self,
        project_path: str,
        timeout_tier: str = "code",
        allowed_tools: list[str] | None = None,
        permission_mode: str = "acceptEdits",
    ):
        """
        Initialize Claude executor.

        Args:
            project_path: Working directory for Claude
            timeout_tier: One of 'quick', 'code', 'debug', 'research'
            allowed_tools: List of allowed tool names (default: standard set)
            permission_mode: Permission mode for Claude (default: acceptEdits)
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
        self.permission_mode = permission_mode

        # Verify Claude CLI is installed
        if not shutil.which("claude"):
            raise ClaudeNotFoundError("Claude Code CLI not found in PATH")

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

        parts.append("")
        parts.append("When complete, summarize what you did in 1-2 sentences.")

        return "\n".join(parts)

    def _build_command(self, prompt: str) -> list[str]:
        """Build the Claude CLI command."""
        cmd = [
            "claude",
            "-p",  # Print mode (non-interactive)
            "--output-format", "stream-json",
            "--verbose",
            "--permission-mode", self.permission_mode,
        ]

        # Add allowed tools
        if self.allowed_tools:
            cmd.extend(["--allowedTools", ",".join(self.allowed_tools)])

        return cmd

    def _clean_environment(self) -> dict[str, str]:
        """
        Create clean environment for Claude subprocess.

        Removes sensitive keys that Claude shouldn't have access to.
        Uses pattern matching to catch various credential formats.
        """
        env = os.environ.copy()

        # Patterns that indicate sensitive environment variables
        sensitive_patterns = [
            "_KEY",
            "_TOKEN",
            "_SECRET",
            "_PASSWORD",
            "_CREDENTIAL",
            "API_KEY",
            "AUTH_",
            "AWS_",
            "AZURE_",
            "GCP_",
            "DATABASE_URL",
            "MONGO_URI",
            "REDIS_URL",
        ]

        # Explicit keys to remove (in addition to patterns)
        explicit_keys = [
            "OPENROUTER_API_KEY",
            "OPENAI_API_KEY",
            "PERPLEXITY_API_KEY",
            "ANTHROPIC_API_KEY",
            "GITHUB_TOKEN",
        ]

        # Remove explicit keys
        for key in explicit_keys:
            env.pop(key, None)

        # Remove keys matching patterns (except Claude's own ANTHROPIC key which it needs)
        keys_to_remove = []
        for key in env:
            key_upper = key.upper()
            for pattern in sensitive_patterns:
                if pattern in key_upper:
                    keys_to_remove.append(key)
                    break

        for key in keys_to_remove:
            env.pop(key, None)

        return env

    async def execute(
        self,
        task_description: str,
        files: list[str] | None = None,
        constraints: list[str] | None = None,
        on_progress: Callable[[str], None] | None = None,
        on_tool_call: Callable[[ToolCall], None] | None = None,
    ) -> TaskResult:
        """
        Execute a task via Claude Code.

        Args:
            task_description: What Claude should do
            files: Specific files to work with
            constraints: Constraints for Claude to follow
            on_progress: Callback for progress updates (text chunks)
            on_tool_call: Callback when a tool is called

        Returns:
            TaskResult with success/failure and details

        Raises:
            ClaudeNotFoundError: If Claude CLI not installed
            ClaudeTimeoutError: If task times out
            ClaudeExecutionError: For execution errors
        """
        prompt = self.build_prompt(task_description, files, constraints)
        cmd = self._build_command(prompt)
        env = self._clean_environment()

        tool_calls: list[ToolCall] = []
        file_changes: list[str] = []
        result_text = ""
        session_id = ""
        cost_usd = 0.0
        duration_ms = 0
        num_turns = 0
        is_error = False
        error_message = ""

        process = None
        try:
            # Start subprocess
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.project_path,
                env=env,
            )

            # Send prompt to stdin
            if process.stdin:
                process.stdin.write(prompt.encode())
                await process.stdin.drain()
                process.stdin.close()

            # Read and parse streaming output with timeout
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.timeout,
                )
            except asyncio.TimeoutError:
                if process and process.returncode is None:
                    process.kill()
                    await process.wait()
                raise ClaudeTimeoutError(
                    f"Task timed out after {self.timeout}s",
                    self.timeout,
                )

            # Parse each line of JSON output
            for line in stdout.decode().split("\n"):
                if not line.strip():
                    continue

                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                msg_type = data.get("type")

                # Track tool calls from assistant messages
                if msg_type == "assistant":
                    message = data.get("message", {})
                    content = message.get("content", [])

                    for block in content:
                        block_type = block.get("type")

                        if block_type == "tool_use":
                            # Tool call
                            tool_call = ToolCall(
                                name=block.get("name", ""),
                                input=block.get("input", {}),
                            )
                            tool_calls.append(tool_call)

                            # Track file modifications
                            if tool_call.name in ("Edit", "Write"):
                                file_path = (
                                    tool_call.input.get("file_path")
                                    or tool_call.input.get("path")
                                )
                                if file_path and file_path not in file_changes:
                                    file_changes.append(file_path)

                            if on_tool_call:
                                on_tool_call(tool_call)

                            if on_progress:
                                on_progress(f"Using {tool_call.name}...")

                        elif block_type == "text":
                            # Text response chunk
                            text = block.get("text", "")
                            if on_progress and text:
                                on_progress(text)

                # Final result
                elif msg_type == "result":
                    result_text = data.get("result", "")
                    session_id = data.get("session_id", "")
                    cost_usd = data.get("total_cost_usd", 0.0)
                    duration_ms = data.get("duration_ms", 0)
                    num_turns = data.get("num_turns", 0)
                    is_error = data.get("is_error", False)

                    if is_error:
                        error_message = data.get("subtype", "unknown error")

            # Check exit code
            if process.returncode != 0 and not is_error:
                is_error = True
                error_message = f"Process exited with code {process.returncode}"
                if stderr:
                    error_message += f": {stderr.decode()[:200]}"

            return TaskResult(
                success=not is_error,
                result=result_text if not is_error else None,
                error=error_message if is_error else None,
                tool_calls=tool_calls,
                file_changes=file_changes,
                cost_usd=cost_usd,
                duration_ms=duration_ms,
                session_id=session_id,
                num_turns=num_turns,
            )

        except ClaudeTimeoutError:
            raise
        except Exception as e:
            # Ensure subprocess is cleaned up on any exception
            if process and process.returncode is None:
                try:
                    process.kill()
                    await process.wait()
                except Exception:
                    pass  # Best effort cleanup
            logger.exception("Claude execution failed")
            raise ClaudeExecutionError(
                f"Execution failed: {str(e)}",
                exit_code=-1,
                stderr=str(e),
            )

    async def execute_simple(
        self,
        prompt: str,
        timeout: int | None = None,
    ) -> str:
        """
        Execute a simple prompt and return just the result text.

        Args:
            prompt: Direct prompt to send
            timeout: Optional timeout override

        Returns:
            Result text from Claude

        Raises:
            ClaudeError: On any failure
        """
        original_timeout = self.timeout
        if timeout:
            self.timeout = timeout

        try:
            result = await self.execute(prompt)
            if not result.success:
                raise ClaudeExecutionError(
                    result.error or "Unknown error",
                    exit_code=1,
                )
            return result.result or ""
        finally:
            self.timeout = original_timeout


def get_git_diff(project_path: str, files: list[str] | None = None) -> str:
    """
    Get git diff for changed files.

    Args:
        project_path: Path to git repo
        files: Specific files to diff (or all if None)

    Returns:
        Git diff string
    """
    import subprocess

    cmd = ["git", "diff", "--no-color"]
    if files:
        cmd.extend(["--"] + files)

    try:
        result = subprocess.run(
            cmd,
            cwd=project_path,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout or "(no changes)"
    except Exception:
        return "(could not get diff)"
