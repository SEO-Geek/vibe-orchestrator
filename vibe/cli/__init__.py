"""
Vibe CLI components.

Split into focused modules:
- startup.py: Startup validation and system checks
- project.py: Project selection and context loading
- debug.py: Debug workflow with GLM
- execution.py: Task execution with Claude and GLM review
- commands.py: Slash command handlers
- interactive.py: Main conversation loop
- typer_commands.py: CLI entry points (add, remove, list, etc.)
- prompt.py: Enhanced prompt with history and completion
"""

# Prompt utilities (for backward compatibility)
from vibe.cli.prompt import (
    SLASH_COMMANDS,
    VibeCompleter,
    create_prompt_session,
    get_history_path,
    prompt_input,
)

# Typer app and main entry point
from vibe.cli.typer_commands import app

# Startup validation
from vibe.cli.startup import (
    validate_startup,
    show_startup_panel,
)

# Project management
from vibe.cli.project import (
    show_project_list,
    show_project_loaded,
    load_project_context,
)

# Conversation loop
from vibe.cli.interactive import (
    conversation_loop,
    process_user_request,
    execute_tasks,
)

# Execution helpers
from vibe.cli.execution import (
    execute_task_with_claude,
    review_with_glm,
    show_task_result,
)

# Debug workflow
from vibe.cli.debug import (
    execute_debug_workflow,
)

# CLI command functions (for backward compatibility)
from vibe.cli.typer_commands import (
    main,
    add,
    remove,
    list_projects,
    restore,
    ping,
    logs,
    run,
)

__all__ = [
    # Typer app
    "app",
    # Prompt
    "SLASH_COMMANDS",
    "VibeCompleter",
    "create_prompt_session",
    "get_history_path",
    "prompt_input",
    # Startup
    "validate_startup",
    "show_startup_panel",
    # Project
    "show_project_list",
    "show_project_loaded",
    "load_project_context",
    # Interactive
    "conversation_loop",
    "process_user_request",
    "execute_tasks",
    # Execution
    "execute_task_with_claude",
    "review_with_glm",
    "show_task_result",
    # Debug
    "execute_debug_workflow",
    # CLI commands
    "main",
    "add",
    "remove",
    "list_projects",
    "restore",
    "ping",
    "logs",
    "run",
]
