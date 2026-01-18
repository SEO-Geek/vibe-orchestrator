"""
Log Viewer Utilities for Vibe Orchestrator.

Provides functions to query, filter, and display log entries.
Used by the `vibe logs` CLI command.
"""

import json
import re
from collections.abc import Iterator
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

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

    with open(filepath, encoding="utf-8") as f:
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


def percentile(values: list[float], p: float) -> float:
    """Calculate percentile of a sorted list."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * p / 100
    f = int(k)
    c = f + 1 if f + 1 < len(sorted_vals) else f
    return sorted_vals[f] + (k - f) * (sorted_vals[c] - sorted_vals[f])


def calculate_stats(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Calculate statistics from log entries.

    Args:
        entries: List of log entries

    Returns:
        Dictionary with statistics including percentiles and costs
    """
    glm_entries = [e for e in entries if e.get("_source") == "glm"]
    claude_entries = [e for e in entries if e.get("_source") == "claude"]

    # GLM stats
    total_glm_tokens = sum(e.get("total_tokens", 0) for e in glm_entries)
    prompt_tokens = sum(e.get("prompt_tokens", 0) for e in glm_entries)
    completion_tokens = sum(e.get("completion_tokens", 0) for e in glm_entries)
    glm_latencies = [e.get("latency_ms", 0) for e in glm_entries if e.get("latency_ms")]
    avg_glm_latency = sum(glm_latencies) / len(glm_latencies) if glm_latencies else 0

    # GLM cost estimation ($0.001/1K input, $0.002/1K output)
    glm_cost = (prompt_tokens / 1000) * 0.001 + (completion_tokens / 1000) * 0.002

    # Claude stats (cost tracked by Claude CLI, not here)
    claude_successes = sum(1 for e in claude_entries if e.get("success"))
    claude_failures = len(claude_entries) - claude_successes
    success_rate = (claude_successes / len(claude_entries) * 100) if claude_entries else 0
    durations = [e.get("duration_ms", 0) for e in claude_entries if e.get("duration_ms")]
    avg_duration = sum(durations) / len(durations) if durations else 0

    # Percentiles for latency and duration
    glm_p50 = percentile(glm_latencies, 50)
    glm_p95 = percentile(glm_latencies, 95)
    claude_p50 = percentile(durations, 50)
    claude_p95 = percentile(durations, 95)

    # Method breakdown with tokens
    methods: dict[str, dict[str, int]] = {}
    for e in glm_entries:
        m = e.get("method", "unknown")
        if m not in methods:
            methods[m] = {"count": 0, "tokens": 0}
        methods[m]["count"] += 1
        methods[m]["tokens"] += e.get("total_tokens", 0)

    # Tool usage breakdown
    tool_counts: dict[str, int] = {}
    for e in claude_entries:
        for tc in e.get("tool_calls", []):
            tool_name = tc.get("name", "unknown")
            tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1

    # Error breakdown
    errors: list[str] = []
    for e in glm_entries + claude_entries:
        if err := e.get("error"):
            errors.append(err[:100])

    return {
        "glm_calls": len(glm_entries),
        "glm_tokens": total_glm_tokens,
        "glm_prompt_tokens": prompt_tokens,
        "glm_completion_tokens": completion_tokens,
        "glm_avg_latency_ms": int(avg_glm_latency),
        "glm_p50_latency_ms": int(glm_p50),
        "glm_p95_latency_ms": int(glm_p95),
        "glm_cost_usd": round(glm_cost, 4),
        "glm_methods": methods,
        "claude_executions": len(claude_entries),
        "claude_successes": claude_successes,
        "claude_failures": claude_failures,
        "claude_success_rate": round(success_rate, 1),
        "claude_avg_duration_ms": int(avg_duration),
        "claude_p50_duration_ms": int(claude_p50),
        "claude_p95_duration_ms": int(claude_p95),
        "claude_tools": tool_counts,
        "errors": errors[:10],  # Last 10 errors
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
    # Extract token counts for formatting
    prompt_tok = stats.get("glm_prompt_tokens", 0)
    comp_tok = stats.get("glm_completion_tokens", 0)
    p50_lat = stats.get("glm_p50_latency_ms", 0)
    p95_lat = stats.get("glm_p95_latency_ms", 0)

    lines = [
        "=== GLM Statistics ===",
        f"  Calls:          {stats['glm_calls']}",
        f"  Total Tokens:   {stats['glm_tokens']:,} (in: {prompt_tok:,}, out: {comp_tok:,})",
        f"  Latency:        avg {stats['glm_avg_latency_ms']}ms, p50 {p50_lat}ms, p95 {p95_lat}ms",
        f"  Est. Cost:      ${stats.get('glm_cost_usd', 0):.4f}",
    ]

    if stats.get("glm_methods"):
        lines.append("  Methods:")
        for method, data in sorted(stats["glm_methods"].items()):
            if isinstance(data, dict):
                lines.append(f"    - {method}: {data['count']} calls, {data['tokens']:,} tokens")
            else:
                lines.append(f"    - {method}: {data}")

    # Extract Claude duration metrics
    avg_dur = stats["claude_avg_duration_ms"] / 1000
    p50_dur = stats.get("claude_p50_duration_ms", 0) / 1000
    p95_dur = stats.get("claude_p95_duration_ms", 0) / 1000
    success_rate = stats["claude_success_rate"]

    # Extract success/fail counts
    succ = stats["claude_successes"]
    fail = stats["claude_failures"]

    lines.extend(
        [
            "",
            "=== Claude Statistics ===",
            f"  Executions:     {stats['claude_executions']}",
            f"  Success/Fail:   {succ}/{fail} ({success_rate}%)",
            f"  Duration:       avg {avg_dur:.1f}s, p50 {p50_dur:.1f}s, p95 {p95_dur:.1f}s",
        ]
    )

    if stats.get("claude_tools"):
        lines.append("  Tools Used:")
        for tool, count in sorted(stats["claude_tools"].items(), key=lambda x: -x[1])[:10]:
            lines.append(f"    - {tool}: {count}")

    if stats.get("errors"):
        lines.extend(
            [
                "",
                "=== Recent Errors ===",
            ]
        )
        for err in stats["errors"][:5]:
            lines.append(f"  - {err[:80]}...")

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


def follow_logs(
    log_type: str = "all",
    callback: Any = None,
    poll_interval: float = 0.5,
) -> Iterator[dict[str, Any]]:
    """
    Follow log files in real-time (like tail -f).

    Args:
        log_type: "glm", "claude", "session", or "all"
        callback: Optional callback for each new entry
        poll_interval: Seconds between polls

    Yields:
        New log entries as they appear
    """
    import time

    config = get_config()

    # Determine which files to watch
    files: list[tuple[str, Path]] = []
    if log_type in ("glm", "all"):
        files.append(("glm", config.glm_log_path))
    if log_type in ("claude", "all"):
        files.append(("claude", config.claude_log_path))
    if log_type in ("session", "all"):
        files.append(("session", config.session_log_path))

    # Track file positions
    positions: dict[Path, int] = {}
    for _, filepath in files:
        if filepath.exists():
            positions[filepath] = filepath.stat().st_size
        else:
            positions[filepath] = 0

    while True:
        for source, filepath in files:
            if not filepath.exists():
                continue

            current_size = filepath.stat().st_size
            last_pos = positions.get(filepath, 0)

            if current_size > last_pos:
                # New content available
                with open(filepath, encoding="utf-8") as f:
                    f.seek(last_pos)
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            entry = json.loads(line)
                            entry["_source"] = source
                            if callback:
                                callback(entry)
                            yield entry
                        except json.JSONDecodeError:
                            continue

                positions[filepath] = current_size

        time.sleep(poll_interval)


def get_session_summary(session_id: str) -> dict[str, Any]:
    """
    Get a summary for a specific session.

    Args:
        session_id: The session ID to summarize

    Returns:
        Summary dictionary with all session activity
    """
    entries = query_logs(log_type="all", session_id=session_id, limit=1000)

    glm_entries = [e for e in entries if e.get("_source") == "glm"]
    claude_entries = [e for e in entries if e.get("_source") == "claude"]
    session_entries = [e for e in entries if e.get("_source") == "session"]

    # Get time range
    timestamps = [e.get("timestamp", "") for e in entries if e.get("timestamp")]
    start_time = min(timestamps) if timestamps else ""
    end_time = max(timestamps) if timestamps else ""

    # Calculate duration
    duration_seconds = 0.0
    if start_time and end_time:
        try:
            start = datetime.fromisoformat(start_time)
            end = datetime.fromisoformat(end_time)
            duration_seconds = (end - start).total_seconds()
        except ValueError:
            pass

    # Get user requests from session events
    requests = [
        e.get("user_request", "") for e in session_entries if e.get("event_type") == "request" and e.get("user_request")
    ]

    return {
        "session_id": session_id,
        "start_time": start_time,
        "end_time": end_time,
        "duration_seconds": duration_seconds,
        "glm_calls": len(glm_entries),
        "claude_executions": len(claude_entries),
        "total_tokens": sum(e.get("total_tokens", 0) for e in glm_entries),
        "user_requests": requests,
        "stats": calculate_stats(entries),
    }
