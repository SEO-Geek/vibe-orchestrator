"""Vibe TUI - Main Textual application with escape-to-cancel support."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Any

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Footer, Header, Input, RichLog, Static, Button, ListView, ListItem, Label
from textual.worker import Worker, WorkerState, get_current_worker

from vibe.pricing import CostTracker

if TYPE_CHECKING:
    from vibe.claude.executor import ClaudeExecutor
    from vibe.config import Project, VibeConfig
    from vibe.glm.client import GLMClient
    from vibe.memory.keeper import VibeMemory
    from vibe.orchestrator.supervisor import Supervisor, SupervisorCallbacks
    from vibe.state import SessionContext


class TaskPanel(Static):
    """Visual task progress panel showing todo list."""

    def __init__(self) -> None:
        super().__init__("")
        self.add_class("task-panel")
        self.tasks: list[dict] = []
        self.current_index: int = -1

    def set_tasks(self, tasks: list[dict]) -> None:
        """Set the task list."""
        self.tasks = [dict(t, status="pending") for t in tasks]
        self.current_index = -1
        self._refresh()

    def mark_running(self, index: int) -> None:
        """Mark a task as currently running."""
        if 0 <= index < len(self.tasks):
            self.tasks[index]["status"] = "running"
            self.current_index = index
        self._refresh()

    def mark_complete(self, index: int, success: bool) -> None:
        """Mark a task as complete or failed."""
        if 0 <= index < len(self.tasks):
            self.tasks[index]["status"] = "done" if success else "failed"
        self._refresh()

    def clear_tasks(self) -> None:
        """Clear all tasks."""
        self.tasks = []
        self.current_index = -1
        self._refresh()

    def _refresh(self) -> None:
        """Refresh the display."""
        if not self.tasks:
            self.update("[dim]No tasks[/dim]")
            return
        lines = []
        for i, task in enumerate(self.tasks):
            status = task.get("status", "pending")
            icon = {"done": "[green]✓[/green]", "failed": "[red]✗[/red]", "running": "[yellow]⏳[/yellow]"}.get(status, "○")
            desc = task.get("description", "")[:45]
            marker = " [bold cyan][running][/bold cyan]" if status == "running" else ""
            lines.append(f"{icon} {i+1}. {desc}{marker}")
        self.update("\n".join(lines))


class CostBar(Static):
    """Cost tracker showing GLM + Claude costs."""

    def __init__(self) -> None:
        super().__init__("")
        self.add_class("cost-bar")
        self.glm_cost: float = 0.0
        self.claude_cost: float = 0.0
        self._refresh()

    def add_glm_cost(self, cost: float) -> None:
        """Add GLM cost."""
        self.glm_cost += cost
        self._refresh()

    def add_claude_cost(self, cost: float) -> None:
        """Add Claude cost."""
        self.claude_cost += cost
        self._refresh()

    def reset(self) -> None:
        """Reset costs."""
        self.glm_cost = 0.0
        self.claude_cost = 0.0
        self._refresh()

    def _refresh(self) -> None:
        """Refresh display."""
        total = self.glm_cost + self.claude_cost
        self.update(f"[dim]Costs:[/dim] GLM [cyan]${self.glm_cost:.3f}[/cyan] + Claude [green]${self.claude_cost:.3f}[/green] = [bold]${total:.3f}[/bold]")


class StatusBar(Static):
    """Status bar showing current operation."""

    def __init__(self) -> None:
        super().__init__("Ready")
        self.add_class("status-bar")

    def set_status(self, text: str) -> None:
        self.update(f"[bold cyan]{text}[/bold cyan]")

    def set_ready(self) -> None:
        self.update("[dim]Ready - Press Escape to cancel operations[/dim]")


class PlanReviewScreen(ModalScreen[list[dict[str, Any]] | None]):
    """Modal screen for reviewing task plan before execution."""

    BINDINGS = [
        Binding("enter", "approve", "Approve Plan", show=True),
        Binding("escape", "cancel", "Cancel", show=True),
        Binding("d", "delete_task", "Delete Task", show=True),
    ]

    def __init__(self, tasks: list[dict[str, Any]]) -> None:
        super().__init__()
        self.tasks = list(tasks)  # Copy to allow modifications
        self.selected_index = 0

    def compose(self) -> ComposeResult:
        """Create the modal layout."""
        yield Container(
            Static("[bold]Review Task Plan[/bold]", id="plan-review-title"),
            Static("[dim]Press Enter to approve, Escape to cancel, D to delete selected task[/dim]"),
            RichLog(id="plan-review-tasks"),
            Horizontal(
                Button("Approve", variant="success", id="btn-approve"),
                Button("Cancel", variant="error", id="btn-cancel"),
                id="plan-review-buttons",
            ),
            id="plan-review-container",
        )

    def on_mount(self) -> None:
        """Populate the task list on mount."""
        self._refresh_task_list()

    def _refresh_task_list(self) -> None:
        """Refresh the task list display."""
        log = self.query_one("#plan-review-tasks", RichLog)
        log.clear()
        for i, task in enumerate(self.tasks):
            desc = task.get("description", "No description")
            files = ", ".join(task.get("files", []))[:40] or "any"
            marker = "[bold cyan]>[/bold cyan] " if i == self.selected_index else "  "
            log.write(f"{marker}{i+1}. {desc}")
            if files:
                log.write(f"     [dim]Files: {files}[/dim]")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "btn-approve":
            self.action_approve()
        elif event.button.id == "btn-cancel":
            self.action_cancel()

    def action_approve(self) -> None:
        """Approve the plan and return tasks."""
        self.dismiss(self.tasks)

    def action_cancel(self) -> None:
        """Cancel the plan."""
        self.dismiss(None)

    def action_delete_task(self) -> None:
        """Delete the currently selected task."""
        if self.tasks and 0 <= self.selected_index < len(self.tasks):
            del self.tasks[self.selected_index]
            if self.selected_index >= len(self.tasks):
                self.selected_index = max(0, len(self.tasks) - 1)
            self._refresh_task_list()
            if not self.tasks:
                self.notify("All tasks deleted - plan will be cancelled")

    def key_up(self) -> None:
        """Move selection up."""
        if self.selected_index > 0:
            self.selected_index -= 1
            self._refresh_task_list()

    def key_down(self) -> None:
        """Move selection down."""
        if self.selected_index < len(self.tasks) - 1:
            self.selected_index += 1
            self._refresh_task_list()


class VibeApp(App):
    """Vibe TUI - GLM orchestrator with escape-to-cancel."""

    TITLE = "Vibe Orchestrator"
    SUB_TITLE = "GLM + Claude"

    CSS = """
    #main-container {
        height: 1fr;
        margin: 1;
    }

    #output-container {
        width: 3fr;
        height: 100%;
        border: solid $primary;
    }

    #output {
        height: 100%;
        scrollbar-gutter: stable;
    }

    #sidebar {
        width: 1fr;
        min-width: 30;
        max-width: 50;
        height: 100%;
        margin-left: 1;
    }

    .task-panel {
        height: auto;
        min-height: 5;
        max-height: 15;
        border: solid $secondary;
        padding: 1;
        margin-bottom: 1;
    }

    .cost-bar {
        height: 1;
        background: $surface;
        color: $text;
        padding: 0 1;
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

    /* Plan Review Modal */
    PlanReviewScreen {
        align: center middle;
    }

    #plan-review-container {
        width: 80%;
        height: 80%;
        border: thick $primary;
        background: $surface;
        padding: 1 2;
    }

    #plan-review-title {
        text-align: center;
        text-style: bold;
        margin-bottom: 1;
    }

    #plan-review-tasks {
        height: 1fr;
        border: solid $secondary;
        margin-bottom: 1;
    }

    #plan-review-buttons {
        height: auto;
        align: center middle;
    }

    #plan-review-buttons Button {
        margin: 0 1;
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

        # Supervisor for proper orchestration (with review gate)
        self._supervisor: Supervisor | None = None

        # Track current operation
        self._current_operation: str | None = None
        self._claude_process: asyncio.subprocess.Process | None = None

        # Track costs
        self._glm_cost: float = 0.0
        self._claude_cost: float = 0.0

        # Plan review setting
        self._enable_plan_review: bool = True

    def compose(self) -> ComposeResult:
        """Create the UI layout."""
        yield Header()
        yield Horizontal(
            Container(
                RichLog(id="output", highlight=True, markup=True, wrap=True),
                id="output-container",
            ),
            Vertical(
                TaskPanel(),
                CostBar(),
                id="sidebar",
            ),
            id="main-container",
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
        cmd_lower = cmd.lower().strip()

        if cmd_lower in ("/quit", "/exit", "/q"):
            self.exit()
        elif cmd_lower == "/clear":
            self.action_clear_output()
        elif cmd_lower == "/help":
            self._write_system("Commands:")
            self._write_system("  /quit, /exit, /q - Exit the application")
            self._write_system("  /clear - Clear the output log")
            self._write_system("  /status - Show current operation status")
            self._write_system("  /cost - Show session cost summary")
            self._write_system("  /review - Toggle plan review mode")
            self._write_system("  /compact - Compact old context items")
        elif cmd_lower == "/status":
            if self._current_operation:
                self._write_system(f"Current operation: {self._current_operation}")
            else:
                self._write_system("Idle - no operation in progress")
        elif cmd_lower == "/cost":
            cost_bar = self.query_one(CostBar)
            total = cost_bar.glm_cost + cost_bar.claude_cost
            self._write_system(f"Session Costs:")
            self._write_system(f"  GLM:    ${cost_bar.glm_cost:.4f}")
            self._write_system(f"  Claude: ${cost_bar.claude_cost:.4f}")
            self._write_system(f"  Total:  ${total:.4f}")
        elif cmd_lower == "/review":
            self._enable_plan_review = not self._enable_plan_review
            status = "enabled" if self._enable_plan_review else "disabled"
            self._write_system(f"Plan review mode: {status}")
        elif cmd_lower == "/compact":
            await self._run_compaction()
        else:
            self._write_system(f"Unknown command: {cmd}")

    async def _run_compaction(self) -> None:
        """Run context compaction."""
        if not self.glm_client or not self.memory:
            self._write_error("Compaction requires GLM client and memory")
            return

        self._write_system("Starting context compaction...")
        try:
            from vibe.memory.compaction import compact_context
            result = await compact_context(self.glm_client, self.memory)
            self._write_system(f"Compaction result: {result.get('reason', 'Unknown')}")
            if result.get("compacted", 0) > 0:
                self._write_system(f"  Items compacted: {result['compacted']}")
                self._write_system(f"  Summaries created: {result['summaries']}")
        except Exception as e:
            self._write_error(f"Compaction failed: {e}")

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
        """Process user request through Supervisor with proper review gate."""
        worker = get_current_worker()
        task_panel = self.query_one(TaskPanel)
        cost_bar = self.query_one(CostBar)

        if not self.glm_client:
            self._write_error("GLM client not initialized")
            return

        if not self.project:
            self._write_error("No project selected")
            return

        try:
            # Create Supervisor if not exists or project changed
            if self._supervisor is None or self._supervisor.project != self.project:
                from vibe.orchestrator.supervisor import Supervisor, SupervisorCallbacks
                from vibe.state import SessionState

                # Create callbacks that update TUI
                callbacks = SupervisorCallbacks(
                    on_status=lambda msg: self.call_from_thread(self._set_status, msg),
                    on_progress=lambda msg: self.call_from_thread(self._write_output, f"  {msg}", "dim"),
                    on_task_start=lambda task: self.call_from_thread(self._write_claude, f"Starting: {task.description[:60]}"),
                    on_task_complete=lambda task, success: self.call_from_thread(
                        self._write_output,
                        f"  {'✓' if success else '✗'} {task.description[:50]}",
                        "green" if success else "red"
                    ),
                    on_review_result=lambda approved, feedback: self.call_from_thread(
                        self._write_glm,
                        f"Review: {'APPROVED' if approved else 'REJECTED'} - {feedback[:80]}"
                    ),
                    on_error=lambda msg: self.call_from_thread(self._write_error, msg),
                )

                self._supervisor = Supervisor(
                    glm_client=self.glm_client,
                    project=self.project,
                    memory=self.memory,
                    callbacks=callbacks,
                )
                # Set context to IDLE so it can transition to PLANNING
                self._supervisor.context.transition_to(SessionState.IDLE)

            # Check cancellation before starting
            if worker.is_cancelled:
                self._write_system("Cancelled before processing")
                return

            self._set_status("Processing request through Supervisor...")

            # Optional: Preview tasks before execution
            if self._enable_plan_review:
                # First get task decomposition for preview
                from vibe.cli import load_project_context
                project_context = load_project_context(self.project)

                self._set_status("GLM decomposing into tasks...")
                tasks = await self.glm_client.decompose_task(user_request, project_context)

                if not tasks:
                    self._write_glm("No tasks to execute for this request.")
                    self._set_ready()
                    return

                # Show task plan
                self._write_glm(f"Decomposed into {len(tasks)} task(s):")
                for i, task in enumerate(tasks, 1):
                    desc = task.get("description", "No description")
                    self._write_output(f"  {i}. {desc}", "dim")

                # Show approval modal
                self._set_status("Waiting for plan approval...")
                approved_tasks = await self.push_screen_wait(PlanReviewScreen(tasks))
                if approved_tasks is None or len(approved_tasks) == 0:
                    self._write_system("Plan cancelled by user")
                    task_panel.clear_tasks()
                    self._set_ready()
                    return
                self._write_system(f"Plan approved with {len(approved_tasks)} task(s)")

                # Update TaskPanel
                task_panel.set_tasks(approved_tasks)

            # Run through Supervisor - includes review gate!
            # Skip clarification since we've already decomposed
            result = await self._supervisor.process_user_request(
                request=user_request,
                skip_clarification=True,  # We handled this in preview
            )

            # Update costs
            if result.total_cost_usd > 0:
                cost_bar.add_claude_cost(result.total_cost_usd)

            # Update task panel with final results
            if result.task_results:
                for i, task_result in enumerate(result.task_results):
                    task_panel.mark_complete(i, task_result.success)
                    if task_result.review_approved:
                        self._write_output(f"  ✓ Task {i+1} reviewed and approved", "green")
                    elif task_result.success:
                        self._write_output(f"  ✓ Task {i+1} completed (no changes)", "cyan")
                    else:
                        self._write_output(f"  ✗ Task {i+1} failed: {task_result.error or 'Unknown'}", "red")

            # Final summary
            if result.success:
                self._write_system(f"Request completed: {result.tasks_completed}/{result.total_tasks} tasks succeeded")
            else:
                self._write_error(f"Request failed: {result.tasks_failed}/{result.total_tasks} tasks failed")

            if result.clarification_asked:
                self._write_glm(f"Clarification needed: {result.clarification_asked}")

            self._set_ready()

        except asyncio.CancelledError:
            self._write_system("Operation cancelled by user")
            task_panel.clear_tasks()
            self._set_ready()
        except Exception as e:
            import traceback
            self._write_error(f"Error: {e}")
            self._write_output(traceback.format_exc(), "dim")
            self._set_ready()

    async def _execute_claude_task(
        self,
        executor: ClaudeExecutor,
        task: dict,
        task_num: int,
        total_tasks: int,
        worker: Worker,
    ) -> bool:
        """Execute a single Claude task with streaming output.

        Returns:
            True if task succeeded, False otherwise
        """
        description = task.get("description", "")
        files = task.get("files", [])
        constraints = task.get("constraints", [])
        cost_bar = self.query_one(CostBar)
        success = False

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
                    return False

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
                        # Track Claude cost if available
                        if hasattr(data, "cost_usd") and data.cost_usd > 0:
                            cost_bar.add_claude_cost(data.cost_usd)
                        success = True
                    else:
                        self._write_error(f"Task {task_num} failed: {data.error}")
                        success = False
                elif event_type == "error":
                    self._write_error(f"Task error: {data}")
                    success = False
                elif event_type == "cancelled":
                    self._write_system(data)
                    return False

        except asyncio.CancelledError:
            executor.cancel()
            raise
        except Exception as e:
            self._write_error(f"Task execution error: {e}")
            return False

        return success

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
