"""
Claude Code Executor

Executes tasks via Claude Code CLI with proper subprocess management,
timeout handling, streaming output parsing, and tool call tracking.

Includes tool enforcement to ensure Claude uses appropriate tools
(e.g., Playwright for browser testing, not curl).
"""

import asyncio
import json
import logging
import os
import shutil
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

from vibe.exceptions import (
    ClaudeError,
    ClaudeExecutionError,
    ClaudeNotFoundError,
    ClaudeTimeoutError,
)
from vibe.logging import (
    claude_logger,
    ClaudeLogEntry,
    now_iso,
    get_session_id,
)
from vibe.config import DEFAULT_DIFF_EXCLUDE_PATTERNS
from vibe.orchestrator.task_enforcer import TaskEnforcer, TaskType


# =============================================================================
# TIMEOUT CHECKPOINT
# Saves partial work when timeout is approaching
# =============================================================================

@dataclass
class TimeoutCheckpoint:
    """
    Checkpoint of partial work saved before timeout.

    Allows recovery of work done before Claude timed out.
    """
    task_description: str
    tool_calls: list["ToolCall"]
    file_changes: list[str]
    partial_output: str
    elapsed_seconds: float
    saved_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "task_description": self.task_description,
            "tool_calls": [
                {"name": tc.name, "input": tc.input, "timestamp": tc.timestamp.isoformat()}
                for tc in self.tool_calls
            ],
            "file_changes": self.file_changes,
            "partial_output": self.partial_output,
            "elapsed_seconds": self.elapsed_seconds,
            "saved_at": self.saved_at.isoformat(),
        }

    def summary(self) -> str:
        """Generate human-readable summary of checkpoint."""
        return (
            f"Checkpoint: {len(self.tool_calls)} tool calls, "
            f"{len(self.file_changes)} files modified, "
            f"{self.elapsed_seconds:.1f}s elapsed"
        )

logger = logging.getLogger(__name__)


# Timeout tiers based on task complexity
TIMEOUT_TIERS = {
    "quick": 120,   # Simple reads, small edits (2 min)
    "code": 900,    # Normal coding tasks (15 min)
    "debug": 900,   # Debugging sessions (15 min)
    "research": 900,  # Research and exploration (15 min)
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
        global_conventions: list[str] | None = None,
    ):
        """
        Initialize Claude executor.

        Args:
            project_path: Working directory for Claude
            timeout_tier: One of 'quick', 'code', 'debug', 'research'
            allowed_tools: List of allowed tool names (default: standard set)
            permission_mode: Permission mode for Claude (default: acceptEdits)
            global_conventions: List of global conventions to enforce
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

        # Task enforcer for tool requirements
        self.task_enforcer = TaskEnforcer(global_conventions=global_conventions)

        # Cancellation support for TUI
        self._cancelled = False
        self._current_process: asyncio.subprocess.Process | None = None

        # Verify Claude CLI is installed
        if not shutil.which("claude"):
            raise ClaudeNotFoundError("Claude Code CLI not found in PATH")

        # Timeout checkpoint for partial work recovery
        self._last_checkpoint: TimeoutCheckpoint | None = None

    def save_checkpoint(
        self,
        task_description: str,
        tool_calls: list[ToolCall],
        file_changes: list[str],
        partial_output: str,
        elapsed_seconds: float,
    ) -> TimeoutCheckpoint:
        """
        Save a checkpoint of partial work.

        Called automatically when timeout is approaching to preserve
        any work done before the timeout.

        Args:
            task_description: The task being executed
            tool_calls: Tool calls made so far
            file_changes: Files modified so far
            partial_output: Any output collected so far
            elapsed_seconds: Time elapsed since start

        Returns:
            The saved checkpoint
        """
        checkpoint = TimeoutCheckpoint(
            task_description=task_description,
            tool_calls=tool_calls,
            file_changes=file_changes,
            partial_output=partial_output,
            elapsed_seconds=elapsed_seconds,
        )
        self._last_checkpoint = checkpoint
        logger.info(f"Saved timeout checkpoint: {checkpoint.summary()}")
        return checkpoint

    def get_last_checkpoint(self) -> TimeoutCheckpoint | None:
        """
        Get the last saved checkpoint.

        Returns:
            The last checkpoint, or None if no checkpoint saved
        """
        return self._last_checkpoint

    def clear_checkpoint(self) -> None:
        """Clear the last saved checkpoint."""
        self._last_checkpoint = None

    def cancel(self) -> None:
        """Cancel the current execution."""
        self._cancelled = True
        if self._current_process and self._current_process.returncode is None:
            self._current_process.terminate()

    def reset_cancellation(self) -> None:
        """Reset cancellation flag for new execution."""
        self._cancelled = False

    def build_prompt(
        self,
        task_description: str,
        files: list[str] | None = None,
        constraints: list[str] | None = None,
        enforce_tools: bool = True,
        debug_context: str | None = None,
    ) -> str:
        """
        Build the prompt for Claude.

        Args:
            task_description: Main task description
            files: Files to work with
            constraints: Constraints to follow
            enforce_tools: Whether to include tool enforcement section
            debug_context: Optional debug session context to inject

        Returns:
            Formatted prompt string
        """
        parts = []

        # If debug context provided, add it FIRST so Claude sees it
        if debug_context:
            parts.append(debug_context)
            parts.append("")

        parts.extend([
            "You are working on a specific task. Do ONLY this task.",
            "",
            f"TASK: {task_description}",
        ])

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

        # Add tool enforcement section
        if enforce_tools:
            enforcement = self.task_enforcer.generate_enforcement_prompt(task_description)
            parts.append(enforcement)

        parts.append("")
        parts.append("When complete, summarize what you did in 1-2 sentences.")

        return "\n".join(parts)

    def verify_tool_usage(
        self,
        task_description: str,
        tool_calls: list[ToolCall],
    ) -> dict[str, Any]:
        """
        Verify that Claude used the required tools for the task.

        Args:
            task_description: The task that was executed
            tool_calls: List of tool calls made by Claude

        Returns:
            Verification result with passed, missing_tools, violations
        """
        # Convert ToolCall objects to dicts for the enforcer
        calls = [{"name": tc.name, "input": tc.input} for tc in tool_calls]
        return self.task_enforcer.verify_tool_usage(task_description, calls)

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
        debug_context: str | None = None,
        timeout_tier: str | None = None,
    ) -> TaskResult:
        """
        Execute a task via Claude Code.

        Args:
            task_description: What Claude should do
            files: Specific files to work with
            constraints: Constraints for Claude to follow
            on_progress: Callback for progress updates (text chunks)
            on_tool_call: Callback when a tool is called
            debug_context: Optional debug session context to inject
            timeout_tier: Override timeout tier ('quick', 'code', 'debug', 'research')

        Returns:
            TaskResult with success/failure and details

        Raises:
            ClaudeNotFoundError: If Claude CLI not installed
            ClaudeTimeoutError: If task times out
            ClaudeExecutionError: For execution errors
        """
        # Temporarily override timeout if tier specified
        original_timeout = self.timeout
        if timeout_tier and timeout_tier in TIMEOUT_TIERS:
            self.timeout = TIMEOUT_TIERS[timeout_tier]
            logger.info(f"Using timeout tier '{timeout_tier}': {self.timeout}s")
        prompt = self.build_prompt(
            task_description, files, constraints,
            enforce_tools=True, debug_context=debug_context
        )
        cmd = self._build_command(prompt)
        env = self._clean_environment()

        # Prepare log entry for this execution
        execution_id = str(uuid.uuid4())
        log_entry = ClaudeLogEntry(
            timestamp=now_iso(),
            execution_id=execution_id,
            session_id=get_session_id(),
            prompt=prompt[:20000],  # Truncate for log size
            files=files or [],
            constraints=constraints or [],
            timeout_tier=timeout_tier or "code",
            allowed_tools=self.allowed_tools,
            project_path=self.project_path,
        )

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
            start_time = datetime.now()
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.timeout,
                )
            except asyncio.TimeoutError:
                # Calculate elapsed time for checkpoint
                elapsed = (datetime.now() - start_time).total_seconds()

                # Save checkpoint of any partial work done before timeout
                # This preserves tool calls and file changes so work isn't lost
                if tool_calls or file_changes:
                    self.save_checkpoint(
                        task_description=task_description,
                        tool_calls=tool_calls,
                        file_changes=file_changes,
                        partial_output=result_text,
                        elapsed_seconds=elapsed,
                    )
                    logger.warning(
                        f"Timeout checkpoint saved: {len(tool_calls)} tool calls, "
                        f"{len(file_changes)} files modified"
                    )

                if process and process.returncode is None:
                    process.kill()
                    await process.wait()
                raise ClaudeTimeoutError(
                    f"Task timed out after {self.timeout}s (checkpoint saved)",
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

            # Update log entry with results
            log_entry.result = result_text[:5000] if result_text else None
            log_entry.success = not is_error
            log_entry.error = error_message if is_error else None
            log_entry.tool_calls = [
                {"name": tc.name, "input": tc.input, "timestamp": tc.timestamp.isoformat()}
                for tc in tool_calls
            ]
            log_entry.file_changes = file_changes
            log_entry.duration_ms = duration_ms
            log_entry.cost_usd = cost_usd
            log_entry.num_turns = num_turns

            # Log the execution
            if is_error:
                claude_logger.error(log_entry.to_json())
            else:
                claude_logger.info(log_entry.to_json())

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
            # Log timeout
            log_entry.error = f"Timeout after {self.timeout}s"
            log_entry.success = False
            claude_logger.error(log_entry.to_json())
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

            # Log the exception
            log_entry.error = str(e)[:500]
            log_entry.success = False
            claude_logger.error(log_entry.to_json())

            raise ClaudeExecutionError(
                f"Execution failed: {str(e)}",
                exit_code=-1,
                stderr=str(e),
            )
        finally:
            # Restore original timeout if it was overridden
            if timeout_tier:
                self.timeout = original_timeout

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

    async def execute_streaming(
        self,
        task_description: str,
        files: list[str] | None = None,
        constraints: list[str] | None = None,
        debug_context: str | None = None,
        timeout_tier: str | None = None,
    ):
        """
        Execute a task with real-time streaming output.

        This is an async generator that yields (event_type, data) tuples
        as Claude works, allowing for real-time UI updates and cancellation.

        Args:
            task_description: What Claude should do
            files: Files to work with
            constraints: Constraints to follow
            debug_context: Optional debug session context
            timeout_tier: Optional timeout tier override

        Yields:
            Tuples of (event_type, data) where event_type is one of:
            - 'progress': Text progress update
            - 'tool_call': ToolCall object
            - 'text': Text output from Claude
            - 'result': Final TaskResult
            - 'error': Error message
            - 'cancelled': Cancellation notice
        """
        from typing import AsyncGenerator

        self.reset_cancellation()

        # Override timeout if specified
        original_timeout = self.timeout
        if timeout_tier:
            self.timeout = TIMEOUT_TIERS.get(timeout_tier, self.timeout)

        # Build prompt
        prompt = self.build_prompt(
            task_description=task_description,
            files=files,
            constraints=constraints,
            debug_context=debug_context,
        )

        # Build command (prompt sent via stdin, not as -p argument)
        cmd = [
            "claude",
            "-p",  # Print mode (non-interactive)
            "--output-format",
            "stream-json",
            "--verbose",
            "--permission-mode",
            self.permission_mode,
        ]

        # Add allowed tools if specified
        if self.allowed_tools:
            cmd.extend(["--allowedTools", ",".join(self.allowed_tools)])

        # Clean environment to remove sensitive variables
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

        try:
            # Start subprocess
            self._current_process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.project_path,
                env=env,
            )
            process = self._current_process

            # Send prompt to stdin
            if process.stdin:
                process.stdin.write(prompt.encode())
                await process.stdin.drain()
                process.stdin.close()

            yield ("progress", "Claude started...")

            # Read stdout line by line with timeout
            start_time = asyncio.get_event_loop().time()

            while True:
                # Check cancellation
                if self._cancelled:
                    if process.returncode is None:
                        process.terminate()
                        await process.wait()
                    yield ("cancelled", "Operation cancelled by user")
                    return

                # Check timeout
                elapsed = asyncio.get_event_loop().time() - start_time
                if elapsed > self.timeout:
                    if process.returncode is None:
                        process.kill()
                        await process.wait()
                    yield ("error", f"Timeout after {self.timeout}s")
                    return

                # Try to read a line with short timeout
                try:
                    line = await asyncio.wait_for(
                        process.stdout.readline(),
                        timeout=1.0,
                    )
                except asyncio.TimeoutError:
                    # No data ready, check if process ended
                    if process.returncode is not None:
                        break
                    continue

                if not line:
                    # EOF - process ended
                    break

                line_str = line.decode().strip()
                if not line_str:
                    continue

                # Parse JSON
                try:
                    data = json.loads(line_str)
                except json.JSONDecodeError:
                    continue

                msg_type = data.get("type")

                # Handle different message types
                if msg_type == "assistant":
                    message = data.get("message", {})
                    content = message.get("content", [])

                    for block in content:
                        block_type = block.get("type")

                        if block_type == "tool_use":
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

                            yield ("tool_call", tool_call)

                        elif block_type == "text":
                            text = block.get("text", "")
                            if text:
                                yield ("text", text)

                elif msg_type == "result":
                    result_text = data.get("result", "")
                    session_id = data.get("session_id", "")
                    cost_usd = data.get("total_cost_usd", 0.0)
                    duration_ms = data.get("duration_ms", 0)
                    num_turns = data.get("num_turns", 0)
                    is_error = data.get("is_error", False)
                    if is_error:
                        error_message = data.get("subtype", "unknown error")

            # Wait for process to complete
            await process.wait()

            # Check exit code
            if process.returncode != 0 and not is_error:
                is_error = True
                stderr_data = await process.stderr.read()
                error_message = f"Process exited with code {process.returncode}"
                if stderr_data:
                    error_message += f": {stderr_data.decode()[:200]}"

            # Yield final result
            yield ("result", TaskResult(
                success=not is_error,
                result=result_text if not is_error else None,
                error=error_message if is_error else None,
                tool_calls=tool_calls,
                file_changes=file_changes,
                cost_usd=cost_usd,
                duration_ms=duration_ms,
                session_id=session_id,
                num_turns=num_turns,
            ))

        except asyncio.CancelledError:
            if self._current_process and self._current_process.returncode is None:
                self._current_process.terminate()
            yield ("cancelled", "Operation cancelled")
        except Exception as e:
            if self._current_process and self._current_process.returncode is None:
                try:
                    self._current_process.kill()
                    await self._current_process.wait()
                except Exception:
                    pass
            yield ("error", str(e))
        finally:
            self._current_process = None
            if timeout_tier:
                self.timeout = original_timeout


def get_git_diff(
    project_path: str,
    files: list[str] | None = None,
    max_chars: int = 100_000,
    exclude_patterns: list[str] | None = None,
) -> tuple[str, bool]:
    """
    Get git diff for changed files with truncation for large diffs.
    Also handles untracked files by showing their content.

    Filters out noisy files like lock files, node_modules, etc. by default.

    Args:
        project_path: Path to git repo
        files: Specific files to diff (or all if None)
        max_chars: Maximum characters to return (default 100k, ~25k tokens)
        exclude_patterns: File patterns to exclude (default: lock files, etc.)

    Returns:
        Tuple of (diff_content, was_truncated):
        - diff_content: Git diff string (may be truncated)
        - was_truncated: True if diff was truncated due to size
    """
    import fnmatch
    import subprocess
    from pathlib import Path

    # Use default exclusion patterns if not specified
    if exclude_patterns is None:
        exclude_patterns = DEFAULT_DIFF_EXCLUDE_PATTERNS

    def should_exclude(filepath: str) -> bool:
        """Check if file matches any exclusion pattern."""
        for pattern in exclude_patterns:
            if fnmatch.fnmatch(filepath, pattern):
                return True
            # Also check basename for patterns like "*.lock"
            if fnmatch.fnmatch(Path(filepath).name, pattern):
                return True
        return False

    # If no files specified, return no changes (don't show unrelated diffs)
    if not files:
        return "(no code changes - task was information-only)", False

    # Filter out excluded files
    filtered_files = [f for f in files if not should_exclude(f)]

    # Log if files were filtered
    excluded_count = len(files) - len(filtered_files)
    if excluded_count > 0:
        logger = logging.getLogger(__name__)
        logger.debug(f"Excluded {excluded_count} files from diff (noisy patterns)")

    if not filtered_files:
        return f"(all {len(files)} changed files were filtered out as noise)", False

    output_parts = []

    # First, check which files are untracked using git status
    untracked_files = set()
    try:
        status_result = subprocess.run(
            ["git", "status", "--porcelain", "--"] + filtered_files,
            cwd=project_path,
            capture_output=True,
            text=True,
            timeout=10,
        )
        for line in status_result.stdout.strip().split("\n"):
            if line.startswith("??"):  # Untracked file marker
                # Format: "?? path/to/file"
                untracked_path = line[3:].strip()
                untracked_files.add(untracked_path)
    except Exception:
        pass  # Continue even if status check fails

    # Get regular git diff for tracked files
    tracked_files = [f for f in filtered_files if f not in untracked_files]
    if tracked_files:
        cmd = ["git", "diff", "--no-color", "--"]
        cmd.extend(tracked_files)

        try:
            result = subprocess.run(
                cmd,
                cwd=project_path,
                capture_output=True,
                text=True,
                timeout=10,
            )
            diff = result.stdout.strip()
            if diff:
                output_parts.append(diff)
        except Exception:
            output_parts.append("(could not get diff for tracked files)")

    # For untracked files, show their content as "new file" diffs
    for untracked in untracked_files:
        file_path = Path(project_path) / untracked
        if file_path.exists():
            try:
                content = file_path.read_text()
                # Format like a git diff for new file
                lines = content.split("\n")
                diff_lines = [f"+{line}" for line in lines]
                new_file_diff = (
                    f"diff --git a/{untracked} b/{untracked}\n"
                    f"new file (untracked)\n"
                    f"--- /dev/null\n"
                    f"+++ b/{untracked}\n"
                    f"@@ -0,0 +1,{len(lines)} @@\n"
                    + "\n".join(diff_lines)
                )
                output_parts.append(new_file_diff)
            except Exception:
                output_parts.append(f"(could not read untracked file: {untracked})")

    # Combine all output
    if not output_parts:
        return f"(no changes detected in: {', '.join(filtered_files)})", False

    combined = "\n\n".join(output_parts)
    original_len = len(combined)

    # Truncate large diffs to prevent context window issues
    if len(combined) > max_chars:
        truncated_diff = combined[:max_chars]
        # Try to truncate at a line boundary
        last_newline = truncated_diff.rfind("\n")
        if last_newline > max_chars - 1000:
            truncated_diff = truncated_diff[:last_newline]
        diff_content = (
            f"{truncated_diff}\n\n"
            f"... [TRUNCATED - diff was {original_len:,} chars, showing first {len(truncated_diff):,}]"
        )
        return diff_content, True

    return combined, False
