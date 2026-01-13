"""Vibe TUI - Main Textual application with escape-to-cancel support."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Vertical
from textual.widgets import Footer, Header, Input, RichLog, Static
from textual.worker import Worker, WorkerState, get_current_worker

if TYPE_CHECKING:
    from vibe.claude.executor import ClaudeExecutor
    from vibe.config import Project, VibeConfig
    from vibe.glm.client import GLMClient
    from vibe.memory.keeper import VibeMemory
    from vibe.state import SessionContext


class StatusBar(Static):
    """Status bar showing current operation."""

    def __init__(self) -> None:
        super().__init__("Ready")
        self.add_class("status-bar")

    def set_status(self, text: str) -> None:
        self.update(f"[bold cyan]{text}[/bold cyan]")

    def set_ready(self) -> None:
        self.update("[dim]Ready - Press Escape to cancel operations[/dim]")


class VibeApp(App):
    """Vibe TUI - GLM orchestrator with escape-to-cancel."""

    TITLE = "Vibe Orchestrator"
    SUB_TITLE = "GLM + Claude"

    CSS = """
    #output-container {
        height: 1fr;
        border: solid $primary;
        margin: 1;
    }

    #output {
        height: 100%;
        scrollbar-gutter: stable;
    }

    #input-container {
        height: auto;
        margin: 0 1 1 1;
    }

    #prompt {
        dock: bottom;
    }

    .status-bar {
        height: 1;
        background: $surface;
        color: $text-muted;
        padding: 0 1;
    }

    .glm-message {
        color: $success;
    }

    .claude-message {
        color: $primary;
    }

    .user-message {
        color: $warning;
    }

    .error-message {
        color: $error;
    }

    .system-message {
        color: $text-muted;
        text-style: italic;
    }
    """

    BINDINGS = [
        Binding("escape", "cancel_operation", "Cancel", show=True, priority=True),
        Binding("ctrl+c", "quit", "Quit", show=True),
        Binding("ctrl+l", "clear_output", "Clear", show=True),
    ]

    def __init__(
        self,
        config: VibeConfig | None = None,
        project: Project | None = None,
        glm_client: GLMClient | None = None,
        context: SessionContext | None = None,
        memory: VibeMemory | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.project = project
        self.glm_client = glm_client
        self.context = context
        self.memory = memory

        # Track current operation
        self._current_operation: str | None = None
        self._claude_process: asyncio.subprocess.Process | None = None

    def compose(self) -> ComposeResult:
        """Create the UI layout."""
        yield Header()
        yield Container(
            RichLog(id="output", highlight=True, markup=True, wrap=True),
            id="output-container",
        )
        yield StatusBar()
        yield Container(
            Input(placeholder="Talk to GLM... (Escape to cancel)", id="prompt"),
            id="input-container",
        )
        yield Footer()

    def on_mount(self) -> None:
        """Called when app is mounted."""
        self.query_one("#prompt", Input).focus()
        self._write_system("Vibe TUI started. Type a message to begin.")
        if self.project:
            self._write_system(f"Project: {self.project.name} ({self.project.path})")

    def _write_output(self, text: str, style: str = "") -> None:
        """Write to the output log."""
        output = self.query_one("#output", RichLog)
        if style:
            output.write(f"[{style}]{text}[/{style}]")
        else:
            output.write(text)

    def _write_glm(self, text: str) -> None:
        """Write GLM output."""
        self._write_output(f"[GLM] {text}", "green")

    def _write_claude(self, text: str) -> None:
        """Write Claude output."""
        self._write_output(f"[Claude] {text}", "cyan")

    def _write_user(self, text: str) -> None:
        """Write user message."""
        self._write_output(f"> {text}", "yellow")

    def _write_error(self, text: str) -> None:
        """Write error message."""
        self._write_output(f"ERROR: {text}", "red bold")

    def _write_system(self, text: str) -> None:
        """Write system message."""
        self._write_output(text, "dim italic")

    def _set_status(self, text: str) -> None:
        """Update status bar."""
        status = self.query_one(StatusBar)
        status.set_status(text)
        self._current_operation = text

    def _set_ready(self) -> None:
        """Set status to ready."""
        status = self.query_one(StatusBar)
        status.set_ready()
        self._current_operation = None

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle user input submission."""
        user_input = event.value.strip()
        if not user_input:
            return

        # Clear input
        event.input.value = ""

        # Handle commands
        if user_input.startswith("/"):
            await self._handle_command(user_input)
            return

        # Regular message - process with GLM
        self._write_user(user_input)
        self.process_request(user_input)

    async def _handle_command(self, cmd: str) -> None:
        """Handle slash commands."""
        cmd_lower = cmd.lower()

        if cmd_lower in ("/quit", "/exit", "/q"):
            self.exit()
        elif cmd_lower == "/clear":
            self.action_clear_output()
        elif cmd_lower == "/help":
            self._write_system("Commands: /quit, /clear, /help, /status")
        elif cmd_lower == "/status":
            if self._current_operation:
                self._write_system(f"Current operation: {self._current_operation}")
            else:
                self._write_system("Idle - no operation in progress")
        else:
            self._write_system(f"Unknown command: {cmd}")

    def action_cancel_operation(self) -> None:
        """Cancel the current operation (Escape key handler)."""
        if not self._current_operation:
            self._write_system("No operation to cancel")
            return

        self._write_system(f"Cancelling: {self._current_operation}")

        # Cancel all workers
        self.workers.cancel_all()

        # Kill Claude subprocess if running
        if self._claude_process and self._claude_process.returncode is None:
            self._claude_process.terminate()
            self._write_system("Claude process terminated")

        self._set_ready()
        self.notify("Operation cancelled", severity="warning")

    def action_clear_output(self) -> None:
        """Clear the output log."""
        output = self.query_one("#output", RichLog)
        output.clear()

    @work(exclusive=True)
    async def process_request(self, user_request: str) -> None:
        """Process user request through GLM (async worker)."""
        worker = get_current_worker()

        if not self.glm_client:
            self._write_error("GLM client not initialized")
            return

        if not self.project:
            self._write_error("No project selected")
            return

        try:
            # Phase 1: GLM analyzes and decomposes
            self._set_status("GLM analyzing request...")

            # Load project context
            from vibe.cli import load_project_context
            project_context = load_project_context(self.project, memory=self.memory)

            # Check cancellation
            if worker.is_cancelled:
                self._write_system("Cancelled during context loading")
                return

            # Ask GLM to decompose task
            self._set_status("GLM decomposing into tasks...")
            tasks = await self.glm_client.decompose_task(user_request, project_context)

            if worker.is_cancelled:
                self._write_system("Cancelled during task decomposition")
                return

            if not tasks:
                self._write_glm("No tasks to execute for this request.")
                self._set_ready()
                return

            # Show task plan
            self._write_glm(f"Decomposed into {len(tasks)} task(s):")
            for i, task in enumerate(tasks, 1):
                desc = task.get("description", "No description")
                self._write_output(f"  {i}. {desc}", "dim")

            # Phase 2: Execute each task with Claude
            from vibe.claude.executor import ClaudeExecutor

            executor = ClaudeExecutor(project_path=self.project.path)

            for i, task in enumerate(tasks, 1):
                if worker.is_cancelled:
                    self._write_system(f"Cancelled before task {i}")
                    break

                desc = task.get("description", "Task")
                self._set_status(f"Claude executing task {i}/{len(tasks)}: {desc[:50]}...")

                # Execute with streaming
                await self._execute_claude_task(executor, task, i, len(tasks), worker)

                if worker.is_cancelled:
                    self._write_system(f"Cancelled during task {i}")
                    break

            self._set_ready()
            self._write_system("Request processing complete")

        except asyncio.CancelledError:
            self._write_system("Operation cancelled by user")
            self._set_ready()
        except Exception as e:
            self._write_error(f"Error: {e}")
            self._set_ready()

    async def _execute_claude_task(
        self,
        executor: ClaudeExecutor,
        task: dict,
        task_num: int,
        total_tasks: int,
        worker: Worker,
    ) -> None:
        """Execute a single Claude task with streaming output."""
        description = task.get("description", "")
        files = task.get("files", [])
        constraints = task.get("constraints", [])

        self._write_claude(f"Starting task {task_num}/{total_tasks}: {description}")

        try:
            # Use streaming executor for real-time output
            async for event_type, data in executor.execute_streaming(
                task_description=description,
                files=files,
                constraints=constraints,
            ):
                # Check worker cancellation
                if worker.is_cancelled:
                    executor.cancel()
                    self._write_system("Task cancelled by user")
                    return

                if event_type == "progress":
                    self._write_output(f"  {data}", "dim")
                elif event_type == "tool_call":
                    tool_input = str(data.input)[:60] if data.input else ""
                    self._write_output(f"  [{data.name}] {tool_input}...", "cyan dim")
                elif event_type == "text":
                    self._write_output(f"  {data[:100]}", "dim")
                elif event_type == "result":
                    if data.success:
                        self._write_claude(f"Task {task_num} completed successfully")
                        if data.result:
                            self._write_output(f"  {data.result[:200]}", "dim")
                    else:
                        self._write_error(f"Task {task_num} failed: {data.error}")
                elif event_type == "error":
                    self._write_error(f"Task error: {data}")
                elif event_type == "cancelled":
                    self._write_system(data)
                    return

        except asyncio.CancelledError:
            executor.cancel()
            raise
        except Exception as e:
            self._write_error(f"Task execution error: {e}")

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        """Handle worker state changes."""
        if event.state == WorkerState.CANCELLED:
            self._set_ready()
        elif event.state == WorkerState.ERROR:
            self._write_error(f"Worker error: {event.worker.error}")
            self._set_ready()


def run_tui(
    config: VibeConfig | None = None,
    project: Project | None = None,
    glm_client: GLMClient | None = None,
    context: SessionContext | None = None,
    memory: VibeMemory | None = None,
) -> None:
    """Run the Vibe TUI application."""
    app = VibeApp(
        config=config,
        project=project,
        glm_client=glm_client,
        context=context,
        memory=memory,
    )
    app.run()
