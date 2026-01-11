"""
Custom Log Handlers for Vibe Orchestrator.

JSONL rotating file handler for structured log output.
"""

import json
import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any


class JSONLRotatingHandler(RotatingFileHandler):
    """
    Rotating file handler that writes JSONL format.

    Each log record is written as a single JSON line.
    Supports automatic rotation based on file size.
    """

    def __init__(
        self,
        filename: str | Path,
        max_bytes: int = 10_000_000,  # 10MB
        backup_count: int = 5,
    ):
        """
        Initialize JSONL handler.

        Args:
            filename: Path to log file
            max_bytes: Max file size before rotation (default 10MB)
            backup_count: Number of backup files to keep
        """
        # Ensure parent directory exists
        filepath = Path(filename)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        super().__init__(
            str(filepath),
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )

    def emit(self, record: logging.LogRecord) -> None:
        """
        Emit a log record as a JSONL line.

        If the message is already JSON, writes it directly.
        Otherwise wraps the message in a JSON structure.
        """
        try:
            msg = self.format(record)

            # Try to parse as JSON (log entries use .to_json())
            try:
                data = json.loads(msg)
            except json.JSONDecodeError:
                # Plain text message - wrap in JSON structure
                data = {
                    "timestamp": datetime.fromtimestamp(record.created).isoformat(),
                    "level": record.levelname,
                    "message": msg,
                    "logger": record.name,
                }

            # Write as single line with newline
            self.stream.write(json.dumps(data, default=str) + "\n")
            self.flush()

        except Exception:
            self.handleError(record)


class SimpleFormatter(logging.Formatter):
    """
    Formatter that just returns the message as-is.

    Used with JSONLRotatingHandler since entries are already JSON.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Return just the message, no formatting."""
        return record.getMessage()


def create_jsonl_logger(
    name: str,
    filepath: Path,
    level: str = "INFO",
    max_bytes: int = 10_000_000,
    backup_count: int = 5,
) -> logging.Logger:
    """
    Create a logger configured for JSONL output.

    Args:
        name: Logger name
        filepath: Path to log file
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        max_bytes: Max file size before rotation
        backup_count: Number of backup files

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Add JSONL handler
    handler = JSONLRotatingHandler(
        filepath,
        max_bytes=max_bytes,
        backup_count=backup_count,
    )
    handler.setFormatter(SimpleFormatter())
    logger.addHandler(handler)

    # Don't propagate to root logger
    logger.propagate = False

    return logger
