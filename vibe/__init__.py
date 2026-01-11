"""
Vibe Orchestrator - GLM as brain, Claude as worker.

A CLI tool for AI-assisted pair programming where GLM-4.7 acts as the
project manager/supervisor and Claude Code executes tasks.
"""

__version__ = "0.1.0"
__author__ = "Brian"

from vibe.exceptions import (
    VibeError,
    ConfigError,
    GLMError,
    ClaudeError,
    VibeMemoryError,
    StartupError,
)

__all__ = [
    "__version__",
    "VibeError",
    "ConfigError",
    "GLMError",
    "ClaudeError",
    "VibeMemoryError",
    "StartupError",
]
