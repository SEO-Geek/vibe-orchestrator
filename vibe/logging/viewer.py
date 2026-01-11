"""
Log Viewer Utilities for Vibe Orchestrator.

Provides functions to query, filter, and display log entries.
Used by the `vibe logs` CLI command.
"""

import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterator

from .config import get_config


def parse_since(since: str) -> datetime:
    """
    Parse a 'since' time string into a datetime.

    Supports:
        - ISO format: "2026-01-11T10:00:00"
        - Relative: "1h", "30m", "2d", "1w"

    Args:
        since: Time string to parse

    Returns:
        datetime object
    """
    # Try ISO format first
    try:
        return datetime.fromisoformat(since)
    except ValueError:
        pass

    # Parse relative time (e.g., "1h", "30m", "2d")
    match = re.match(r"^(\d+)([mhdw])$", since.lower())
    if match:
        value = int(match.group(1))
        unit = match.group(2)

        delta_map = {
            "m": timedelta(minutes=value),
            "h": timedelta(hours=value),
            "d": timedelta(days=value),
            "w": timedelta(weeks=value),
        }
        return datetime.now() - delta_map[unit]

    raise ValueError(f"Invalid time format: {since}. Use ISO format or relative (1h, 30m, 2d)")


def read_jsonl(filepath: Path, limit: int | None = None, since: datetime | None = None) -> Iterator[dict[str, Any]]:
    """
    Read entries from a JSONL file.

    Args:
        filepath: Path to JSONL file
        limit: Maximum entries to return (from end of file)
        since: Only return entries after this time

    Yields:
        Parsed log entries as dicts
    """
    if not filepath.exists():
        return

    entries: list[dict[str, Any]] = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)

                # Filter by time if specified
                if since:
                    ts = entry.get("timestamp", "")
                    try:
                        entry_time = datetime.fromisoformat(ts)
                        if entry_time < since:
                            continue
                    except (ValueError, TypeError):
                        continue

                entries.append(entry)
            except json.JSONDecodeError:
                continue

    # If limit specified, return last N entries
    if limit and len(entries) > limit:
        entries = entries[-limit:]

    yield from entries


def query_logs(
    log_type: str = "all",
    since: str | None = None,
    session_id: str | None = None,
    method: str | None = None,
    success: bool | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """
    Query log entries with filters.

    Args:
        log_type: "glm", "claude", "session", or "all"
        since: Time filter (ISO or relative like "1h")
        session_id: Filter by session ID
        method: Filter GLM logs by method name
        success: Filter Claude logs by success status
        limit: Max entries to return

    Returns:
        List of matching log entries
    """
    config = get_config()
    since_dt = parse_since(since) if since else None

    results: list[dict[str, Any]] = []

    # Determine which files to read
    files: list[tuple[str, Path]] = []
    if log_type in ("glm", "all"):
        files.append(("glm", config.glm_log_path))
    if log_type in ("claude", "all"):
        files.append(("claude", config.claude_log_path))
    if log_type in ("session", "all"):
        files.append(("session", config.session_log_path))

    for source, filepath in files:
        for entry in read_jsonl(filepath, limit=None, since=since_dt):
            # Add source type
            entry["_source"] = source

            # Apply filters
            if session_id and entry.get("session_id") != session_id:
                continue
            if method and entry.get("method") != method:
                continue
            if success is not None and entry.get("success") != success:
                continue

            results.append(entry)

    # Sort by timestamp and limit
    results.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return results[:limit]


def calculate_stats(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Calculate statistics from log entries.

    Args:
        entries: List of log entries

    Returns:
        Dictionary with statistics
    """
    glm_entries = [e for e in entries if e.get("_source") == "glm"]
    claude_entries = [e for e in entries if e.get("_source") == "claude"]

    # GLM stats
    total_glm_tokens = sum(e.get("total_tokens", 0) for e in glm_entries)
    glm_latencies = [e.get("latency_ms", 0) for e in glm_entries if e.get("latency_ms")]
    avg_glm_latency = sum(glm_latencies) / len(glm_latencies) if glm_latencies else 0

    # Claude stats
    claude_successes = sum(1 for e in claude_entries if e.get("success"))
    claude_failures = len(claude_entries) - claude_successes
    success_rate = (claude_successes / len(claude_entries) * 100) if claude_entries else 0
    durations = [e.get("duration_ms", 0) for e in claude_entries if e.get("duration_ms")]
    avg_duration = sum(durations) / len(durations) if durations else 0

    # Method breakdown
    methods: dict[str, int] = {}
    for e in glm_entries:
        m = e.get("method", "unknown")
        methods[m] = methods.get(m, 0) + 1

    return {
        "glm_calls": len(glm_entries),
        "glm_tokens": total_glm_tokens,
        "glm_avg_latency_ms": int(avg_glm_latency),
        "glm_methods": methods,
        "claude_executions": len(claude_entries),
        "claude_successes": claude_successes,
        "claude_failures": claude_failures,
        "claude_success_rate": round(success_rate, 1),
        "claude_avg_duration_ms": int(avg_duration),
    }


def format_entry_line(entry: dict[str, Any]) -> str:
    """
    Format a log entry as a single display line.

    Args:
        entry: Log entry dict

    Returns:
        Formatted string
    """
    source = entry.get("_source", "?")
    ts = entry.get("timestamp", "")[:19]  # Trim to seconds

    if source == "glm":
        method = entry.get("method", "?")
        tokens = entry.get("total_tokens", 0)
        latency = entry.get("latency_ms", 0)
        project = entry.get("project_name", "")
        return f"[{ts}] GLM {method:20s} {tokens:5d} tok  {latency:5d}ms  {project}"

    elif source == "claude":
        task_id = entry.get("task_id", "")[:10] or "direct"
        duration = entry.get("duration_ms", 0) / 1000
        turns = entry.get("num_turns", 0)
        status = "OK" if entry.get("success") else "FAIL"
        return f"[{ts}] CLAUDE {task_id:12s} {duration:5.1f}s  {turns:2d} turns  {status}"

    elif source == "session":
        event = entry.get("event_type", "?")
        project = entry.get("project_name", "")
        request = entry.get("user_request", "")[:40]
        if event == "request" and request:
            return f"[{ts}] SESSION {event:12s} {project}: {request}..."
        return f"[{ts}] SESSION {event:12s} {project}"

    return f"[{ts}] {source.upper()} {json.dumps(entry)[:60]}..."


def format_stats(stats: dict[str, Any]) -> str:
    """
    Format statistics for display.

    Args:
        stats: Stats dictionary from calculate_stats()

    Returns:
        Formatted multi-line string
    """
    lines = [
        "=== GLM Statistics ===",
        f"  Calls:        {stats['glm_calls']}",
        f"  Total Tokens: {stats['glm_tokens']:,}",
        f"  Avg Latency:  {stats['glm_avg_latency_ms']}ms",
    ]

    if stats["glm_methods"]:
        lines.append("  Methods:")
        for method, count in sorted(stats["glm_methods"].items()):
            lines.append(f"    - {method}: {count}")

    lines.extend([
        "",
        "=== Claude Statistics ===",
        f"  Executions:   {stats['claude_executions']}",
        f"  Successes:    {stats['claude_successes']}",
        f"  Failures:     {stats['claude_failures']}",
        f"  Success Rate: {stats['claude_success_rate']}%",
        f"  Avg Duration: {stats['claude_avg_duration_ms']}ms ({stats['claude_avg_duration_ms']/1000:.1f}s)",
    ])

    return "\n".join(lines)


def tail_logs(log_type: str = "all", lines: int = 20) -> list[dict[str, Any]]:
    """
    Get the last N log entries (like tail).

    Args:
        log_type: "glm", "claude", "session", or "all"
        lines: Number of lines to return

    Returns:
        List of log entries
    """
    return query_logs(log_type=log_type, limit=lines)
