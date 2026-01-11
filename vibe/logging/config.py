"""
Logging Configuration for Vibe Orchestrator.

Defines paths, rotation settings, log levels, and privacy patterns.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class LogConfig:
    """Configuration for the Vibe logging system."""

    # Paths
    log_dir: Path = field(default_factory=lambda: Path.home() / ".vibe" / "logs")

    # File settings
    max_file_size_bytes: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5

    # Log levels: DEBUG, INFO, WARNING, ERROR
    glm_level: str = "INFO"
    claude_level: str = "INFO"
    session_level: str = "INFO"

    # Privacy - disabled by default for local use
    redact_enabled: bool = False

    # Console output (in addition to file logging)
    console_enabled: bool = False  # Disabled by default - Rich UI handles console
    console_level: str = "WARNING"

    @classmethod
    def from_env(cls) -> "LogConfig":
        """Load config from environment variables with defaults."""
        config = cls()

        # Override log level from environment
        if level := os.environ.get("VIBE_LOG_LEVEL"):
            config.glm_level = level
            config.claude_level = level
            config.session_level = level

        # Override log directory
        if log_dir := os.environ.get("VIBE_LOG_DIR"):
            config.log_dir = Path(log_dir)

        # Override max file size (in MB)
        if max_size := os.environ.get("VIBE_LOG_MAX_SIZE_MB"):
            try:
                config.max_file_size_bytes = int(max_size) * 1024 * 1024
            except ValueError:
                pass

        return config

    def ensure_log_dir(self) -> None:
        """Create log directory if it doesn't exist."""
        self.log_dir.mkdir(parents=True, exist_ok=True)

    @property
    def glm_log_path(self) -> Path:
        """Path to GLM interaction log."""
        return self.log_dir / "glm.jsonl"

    @property
    def claude_log_path(self) -> Path:
        """Path to Claude execution log."""
        return self.log_dir / "claude.jsonl"

    @property
    def session_log_path(self) -> Path:
        """Path to session lifecycle log."""
        return self.log_dir / "session.jsonl"


# Global config instance - initialized on first import
_config: LogConfig | None = None


def get_config() -> LogConfig:
    """Get the global log config, initializing from env if needed."""
    global _config
    if _config is None:
        _config = LogConfig.from_env()
        _config.ensure_log_dir()
    return _config


def set_config(config: LogConfig) -> None:
    """Set a custom log config (useful for testing)."""
    global _config
    _config = config
    _config.ensure_log_dir()
