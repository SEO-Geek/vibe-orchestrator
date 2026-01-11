"""
GLM Response Parser

Parses structured responses from GLM (task decomposition, reviews, etc.)
"""

import json
import re
from typing import Any

from vibe.exceptions import TaskParseError


def extract_json_from_response(response: str) -> dict[str, Any]:
    """
    Extract JSON from GLM's response text.

    GLM may wrap JSON in markdown code blocks or include preamble text.

    Args:
        response: Raw response text from GLM

    Returns:
        Parsed JSON as dictionary

    Raises:
        TaskParseError: If JSON cannot be extracted or parsed
    """
    # Try to find JSON in code blocks first
    code_block_pattern = r"```(?:json)?\s*([\s\S]*?)```"
    matches = re.findall(code_block_pattern, response)

    for match in matches:
        try:
            return json.loads(match.strip())
        except json.JSONDecodeError:
            continue

    # Try to find raw JSON (object or array) - use non-greedy matching
    # Find objects by matching balanced braces
    json_patterns = [
        r"(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})",  # Object with nested objects
        r"(\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\])",  # Array with nested arrays
    ]

    for pattern in json_patterns:
        matches = re.findall(pattern, response)
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue

    # Nothing worked
    raise TaskParseError(
        "Could not extract valid JSON from GLM response",
        {"response_preview": response[:200]},
    )


def parse_task_list(response: str) -> list[dict[str, Any]]:
    """
    Parse task list from GLM's decomposition response.

    Args:
        response: Raw response from GLM

    Returns:
        List of task dictionaries

    Raises:
        TaskParseError: If parsing fails
    """
    data = extract_json_from_response(response)

    # Handle both {"tasks": [...]} and direct [...]
    if isinstance(data, list):
        tasks = data
    elif isinstance(data, dict) and "tasks" in data:
        tasks = data["tasks"]
    else:
        raise TaskParseError(
            "Response does not contain task list",
            {"keys": list(data.keys()) if isinstance(data, dict) else "not a dict"},
        )

    # Validate task list is not empty
    if not tasks:
        raise TaskParseError(
            "GLM returned empty task list - request may be ambiguous or impossible",
            {"response_preview": response[:200]},
        )

    # Validate task structure
    for i, task in enumerate(tasks):
        if not isinstance(task, dict):
            raise TaskParseError(f"Task {i} is not a dictionary")
        if "description" not in task:
            raise TaskParseError(f"Task {i} missing 'description' field")
        # Ensure description is not empty
        if not task.get("description", "").strip():
            raise TaskParseError(f"Task {i} has empty description")

    return tasks


def parse_review_result(response: str) -> dict[str, Any]:
    """
    Parse review result from GLM's review response.

    Args:
        response: Raw response from GLM

    Returns:
        Review result with 'approved', 'issues', and 'feedback' fields

    Raises:
        TaskParseError: If parsing fails
    """
    data = extract_json_from_response(response)

    if "approved" not in data:
        raise TaskParseError(
            "Review response missing 'approved' field",
            {"keys": list(data.keys())},
        )

    return {
        "approved": bool(data.get("approved")),
        "issues": data.get("issues", []),
        "feedback": data.get("feedback", ""),
    }
