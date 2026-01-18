"""
Enhanced prompt with command history and tab completion.

Uses prompt_toolkit to provide:
- Command history (arrow up/down)
- Tab completion for slash commands
- Multiline input support
- Persistent history across sessions
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style

if TYPE_CHECKING:
    from prompt_toolkit.document import Document


# Available slash commands with descriptions
SLASH_COMMANDS = {
    "/help": "Show available commands",
    "/quit": "Exit the application",
    "/exit": "Exit the application",
    "/q": "Exit the application",
    "/status": "Show session status",
    "/usage": "Show GLM usage stats",
    "/memory": "Show memory stats",
    "/history": "Show task history",
    "/redo": "Re-execute failed task",
    "/convention": "Manage global conventions",
    "/debug": "Debug session tracking",
    "/rollback": "Rollback to debug checkpoint",
    "/research": "Research a topic via Perplexity",
    "/github": "Show GitHub repo info",
    "/issues": "List GitHub issues",
    "/prs": "List GitHub pull requests",
}


class VibeCompleter(Completer):
    """Completer for Vibe CLI commands and project files."""

    def __init__(self, project_path: str | None = None):
        """
        Initialize completer.

        Args:
            project_path: Path to project for file completions
        """
        self.project_path = project_path
        self._file_cache: list[str] = []
        self._cache_time: float = 0

    def get_completions(self, document: Document, complete_event) -> list[Completion]:
        """Generate completions for the current input."""
        text = document.text_before_cursor
        word = document.get_word_before_cursor(WORD=True)

        # Slash command completion
        if text.startswith("/"):
            for cmd, desc in SLASH_COMMANDS.items():
                if cmd.startswith(text):
                    # Show completion with description
                    yield Completion(
                        cmd,
                        start_position=-len(text),
                        display_meta=desc,
                    )

        # File path completion (after keywords like "in", "file:", etc.)
        elif self._should_complete_files(text):
            partial = word if word else ""
            for file_path in self._get_project_files(partial):
                yield Completion(
                    file_path,
                    start_position=-len(partial),
                    display_meta="file",
                )

    def _should_complete_files(self, text: str) -> bool:
        """Check if we should offer file completions."""
        triggers = ["in ", "file:", "path:", "edit ", "read ", "modify "]
        text_lower = text.lower()
        return any(trigger in text_lower for trigger in triggers)

    def _get_project_files(self, prefix: str) -> list[str]:
        """Get project files matching prefix (cached)."""
        import time

        # Refresh cache every 30 seconds
        if time.time() - self._cache_time > 30:
            self._refresh_file_cache()

        prefix_lower = prefix.lower()
        return [f for f in self._file_cache if f.lower().startswith(prefix_lower)][:20]

    def _refresh_file_cache(self) -> None:
        """Refresh the file cache from project directory."""
        import time

        self._cache_time = time.time()
        self._file_cache = []

        if not self.project_path:
            return

        project = Path(self.project_path)
        if not project.exists():
            return

        # Walk project directory (limit depth)
        try:
            for root, dirs, files in os.walk(project):
                # Skip hidden and common ignore dirs
                dirs[:] = [
                    d
                    for d in dirs
                    if not d.startswith(".")
                    and d not in ("node_modules", "__pycache__", ".venv", "venv", ".git", "dist", "build")
                ]

                # Calculate relative depth
                rel_path = Path(root).relative_to(project)
                depth = len(rel_path.parts)
                if depth > 4:  # Max depth
                    continue

                for f in files:
                    if not f.startswith("."):
                        rel_file = str(Path(root).relative_to(project) / f)
                        self._file_cache.append(rel_file)

                        if len(self._file_cache) >= 500:  # Max files
                            return
        except Exception:
            pass


def get_history_path() -> Path:
    """Get path to command history file."""
    # Store in ~/.vibe/history
    vibe_dir = Path.home() / ".vibe"
    vibe_dir.mkdir(exist_ok=True)
    return vibe_dir / "history"


def create_prompt_session(project_path: str | None = None) -> PromptSession:
    """
    Create a prompt session with history and completion.

    Args:
        project_path: Path to project for file completions

    Returns:
        Configured PromptSession
    """
    # Style for the prompt
    style = Style.from_dict(
        {
            "prompt": "ansicyan bold",
            "continuation": "ansigray",
        }
    )

    # Key bindings
    bindings = KeyBindings()

    @bindings.add("c-d")  # Ctrl+D to submit multiline
    def _submit(event):
        """Submit on Ctrl+D."""
        event.current_buffer.validate_and_handle()

    # Create session with history and completion
    session: PromptSession = PromptSession(
        history=FileHistory(str(get_history_path())),
        auto_suggest=AutoSuggestFromHistory(),
        completer=VibeCompleter(project_path),
        complete_while_typing=False,  # Only complete on Tab
        multiline=False,  # We handle multiline manually
        key_bindings=bindings,
        style=style,
    )

    return session


def prompt_input(
    session: PromptSession,
    prompt_text: str = "> ",
    multiline: bool = True,
) -> str:
    """
    Get input from user with history and completion.

    Args:
        session: PromptSession to use
        prompt_text: Text to show before input
        multiline: If True, allow multiline input (submit on empty line)

    Returns:
        User input string
    """
    if not multiline:
        # Simple single-line input
        return session.prompt(prompt_text)

    # For multiline, we accumulate lines until double-enter or Ctrl+D
    lines: list[str] = []

    # First line with prompt
    first_line = session.prompt(prompt_text)

    # Commands are single-line
    if first_line.startswith("/"):
        return first_line

    lines.append(first_line)

    # Continue reading until empty line
    while True:
        try:
            line = session.prompt("... ")
            if not line.strip():
                # Empty line - done with input
                break
            lines.append(line)
        except EOFError:
            # Ctrl+D - submit
            break
        except KeyboardInterrupt:
            # Ctrl+C - cancel
            return ""

    return "\n".join(lines)
