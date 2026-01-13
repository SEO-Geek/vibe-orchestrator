"""
Vibe Orchestrator - Configuration Management

Handles loading projects.json, environment variables, and preferences.
Projects are stored in ~/.config/vibe/projects.json
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from vibe.exceptions import ConfigError, ProjectNotFoundError


# Configuration paths
CONFIG_DIR = Path.home() / ".config" / "vibe"
PROJECTS_FILE = CONFIG_DIR / "projects.json"
MEMORY_DB_PATH = Path.home() / "mcp-data" / "memory-keeper" / "context.db"


@dataclass
class Project:
    """A registered project for Vibe to manage."""

    name: str
    path: str
    starmap: str = "STARMAP.md"
    claude_md: str = "CLAUDE.md"
    test_command: str = "pytest -v"
    description: str = ""
    # Hook scripts (paths relative to project directory)
    pre_task_hooks: list[str] = field(default_factory=list)
    post_task_hooks: list[str] = field(default_factory=list)
    # Workflow settings for intelligent task orchestration
    use_workflows: bool = True       # Enable workflow phase expansion
    inject_subtasks: bool = True     # Enable automatic sub-task injection

    def __post_init__(self) -> None:
        # Expand ~ in path
        self.path = str(Path(self.path).expanduser())

    @property
    def full_path(self) -> Path:
        """Get the full path as a Path object."""
        return Path(self.path)

    @property
    def starmap_path(self) -> Path:
        """Get full path to STARMAP.md."""
        return self.full_path / self.starmap

    @property
    def claude_md_path(self) -> Path:
        """Get full path to CLAUDE.md."""
        return self.full_path / self.claude_md

    def exists(self) -> bool:
        """Check if project directory exists."""
        return self.full_path.exists()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "path": self.path,
            "starmap": self.starmap,
            "claude_md": self.claude_md,
            "test_command": self.test_command,
            "description": self.description,
            "pre_task_hooks": self.pre_task_hooks,
            "post_task_hooks": self.post_task_hooks,
            "use_workflows": self.use_workflows,
            "inject_subtasks": self.inject_subtasks,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Project":
        """Create Project from dictionary."""
        return cls(
            name=data["name"],
            path=data["path"],
            starmap=data.get("starmap", "STARMAP.md"),
            claude_md=data.get("claude_md", "CLAUDE.md"),
            test_command=data.get("test_command", "pytest -v"),
            description=data.get("description", ""),
            pre_task_hooks=data.get("pre_task_hooks", []),
            post_task_hooks=data.get("post_task_hooks", []),
            use_workflows=data.get("use_workflows", True),
            inject_subtasks=data.get("inject_subtasks", True),
        )


@dataclass
class VibeConfig:
    """Main configuration container for Vibe."""

    projects: list[Project] = field(default_factory=list)
    openrouter_api_key: str = ""
    default_model: str = "z-ai/glm-4.7"
    claude_timeout_default: int = 120
    max_retries: int = 3

    def get_project(self, name_or_index: str | int) -> Project:
        """
        Get a project by name or index.

        Args:
            name_or_index: Project name (str) or 1-based index (int)

        Returns:
            The matching Project

        Raises:
            ProjectNotFoundError: If project not found
        """
        # Handle numeric index (1-based for user friendliness)
        if isinstance(name_or_index, int):
            idx = name_or_index - 1
            if 0 <= idx < len(self.projects):
                return self.projects[idx]
            raise ProjectNotFoundError(
                f"Project index {name_or_index} out of range",
                {"available": len(self.projects)},
            )

        # Handle string name
        for project in self.projects:
            if project.name.lower() == name_or_index.lower():
                return project

        raise ProjectNotFoundError(
            f"Project '{name_or_index}' not found",
            {"available": [p.name for p in self.projects]},
        )

    def add_project(self, project: Project) -> None:
        """Add a new project to the configuration."""
        # Check for duplicate names
        for existing in self.projects:
            if existing.name.lower() == project.name.lower():
                raise ConfigError(
                    f"Project '{project.name}' already exists",
                    {"existing_path": existing.path},
                )
        self.projects.append(project)

    def remove_project(self, name: str) -> Project:
        """Remove a project by name and return it."""
        for i, project in enumerate(self.projects):
            if project.name.lower() == name.lower():
                return self.projects.pop(i)
        raise ProjectNotFoundError(f"Project '{name}' not found")


def ensure_config_dir() -> None:
    """Ensure configuration directory exists."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)


def load_config() -> VibeConfig:
    """
    Load configuration from files and environment.

    Returns:
        VibeConfig with all settings loaded

    Raises:
        ConfigError: If configuration is invalid
    """
    config = VibeConfig()

    # Load OpenRouter API key from environment
    config.openrouter_api_key = os.environ.get("OPENROUTER_API_KEY", "")

    # Load projects from file if it exists
    if PROJECTS_FILE.exists():
        try:
            with open(PROJECTS_FILE) as f:
                data = json.load(f)

            for project_data in data.get("projects", []):
                project = Project.from_dict(project_data)
                config.projects.append(project)

        except json.JSONDecodeError as e:
            raise ConfigError(
                f"Invalid JSON in {PROJECTS_FILE}",
                {"error": str(e)},
            )
        except KeyError as e:
            raise ConfigError(
                f"Missing required field in project config",
                {"field": str(e)},
            )

    return config


def save_config(config: VibeConfig) -> None:
    """
    Save configuration to files.

    Args:
        config: VibeConfig to save
    """
    ensure_config_dir()

    data = {
        "projects": [p.to_dict() for p in config.projects],
    }

    with open(PROJECTS_FILE, "w") as f:
        json.dump(data, f, indent=2)


def get_openrouter_key() -> str:
    """
    Get OpenRouter API key from environment.

    Returns:
        API key string

    Raises:
        ConfigError: If key is not set
    """
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        raise ConfigError(
            "OPENROUTER_API_KEY environment variable not set",
            {"hint": "Export OPENROUTER_API_KEY=your-key-here"},
        )
    return key
