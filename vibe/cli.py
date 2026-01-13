#!/usr/bin/env python3
"""
Vibe Orchestrator - CLI Entry Point

Main entry point for the `vibe` command.

NOTE: This module has been split into focused components under vibe/cli/.
This file is a thin shim for backward compatibility.

See:
- vibe/cli/startup.py: Startup validation
- vibe/cli/project.py: Project management UI
- vibe/cli/debug.py: Debug workflow
- vibe/cli/execution.py: Task execution
- vibe/cli/commands.py: Slash command handlers
- vibe/cli/interactive.py: Conversation loop
- vibe/cli/typer_commands.py: CLI commands
- vibe/cli/prompt.py: Enhanced prompt
"""

# Re-export everything from the new modular structure
from vibe.cli import (
    # Prompt utilities
    SLASH_COMMANDS,
    VibeCompleter,
    # Typer app
    app,
    # Interactive
    conversation_loop,
    create_prompt_session,
    # Debug
    execute_debug_workflow,
    # Execution
    execute_task_with_claude,
    execute_tasks,
    get_history_path,
    load_project_context,
    process_user_request,
    prompt_input,
    review_with_glm,
    # Project
    show_project_list,
    show_project_loaded,
    show_startup_panel,
    show_task_result,
    # Startup
    validate_startup,
)

# For backward compatibility with direct imports
from vibe.cli.typer_commands import (
    add,
    list_projects,
    logs,
    main,
    ping,
    remove,
    restore,
)

__all__ = [
    "app",
    "main",
    "add",
    "remove",
    "list_projects",
    "restore",
    "ping",
    "logs",
    "SLASH_COMMANDS",
    "VibeCompleter",
    "create_prompt_session",
    "get_history_path",
    "prompt_input",
    "validate_startup",
    "show_startup_panel",
    "show_project_list",
    "show_project_loaded",
    "load_project_context",
    "conversation_loop",
    "process_user_request",
    "execute_tasks",
    "execute_task_with_claude",
    "review_with_glm",
    "show_task_result",
    "execute_debug_workflow",
]


if __name__ == "__main__":
    app()
