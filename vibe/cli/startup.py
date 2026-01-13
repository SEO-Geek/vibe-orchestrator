"""
Vibe CLI - Startup Validation

Validates all required systems are available before starting Vibe.
"""

import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

from rich.console import Console
from rich.panel import Panel

from vibe.config import MEMORY_DB_PATH
from vibe.glm.client import ping_glm_sync

console = Console()


def _check_openrouter(ping_glm: bool) -> tuple[str, tuple[bool, str]]:
    """Check OpenRouter API (optionally ping GLM)."""
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        return ("OpenRouter API", (False, "OPENROUTER_API_KEY not set"))
    elif ping_glm:
        success, message = ping_glm_sync(api_key, timeout=15.0)
        if success:
            return ("OpenRouter API", (True, f"GLM-4 ({message})"))
        else:
            return ("OpenRouter API", (False, message))
    else:
        return ("OpenRouter API", (True, "key configured"))


def _check_claude_cli() -> tuple[str, tuple[bool, str]]:
    """Check Claude CLI availability."""
    claude_path = shutil.which("claude")
    if claude_path:
        try:
            result = subprocess.run(
                ["claude", "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            version = result.stdout.strip().split("\n")[0] if result.stdout else "unknown"
            return ("Claude Code CLI", (True, version))
        except Exception as e:
            return ("Claude Code CLI", (False, str(e)))
    else:
        return ("Claude Code CLI", (False, "not installed"))


def _check_memory_keeper() -> tuple[str, tuple[bool, str]]:
    """Check memory-keeper database."""
    if MEMORY_DB_PATH.exists():
        return ("Memory-keeper", (True, "database found"))
    else:
        return ("Memory-keeper", (False, "not found"))


def _check_github_cli() -> tuple[str, tuple[bool, str]]:
    """Check GitHub CLI authentication."""
    gh_path = shutil.which("gh")
    if gh_path:
        try:
            result = subprocess.run(
                ["gh", "auth", "status"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                for line in result.stderr.split("\n"):
                    if "Logged in" in line:
                        return ("GitHub CLI", (True, "authenticated"))
                return ("GitHub CLI", (True, "authenticated"))
            else:
                return ("GitHub CLI", (False, "not authenticated"))
        except Exception as e:
            return ("GitHub CLI", (False, str(e)))
    else:
        return ("GitHub CLI", (False, "not installed"))


def validate_startup(ping_glm: bool = True) -> dict[str, tuple[bool, str]]:
    """
    Validate all required systems are available.

    Runs all checks in parallel for faster startup.

    Args:
        ping_glm: Whether to actually ping GLM API (slower but more reliable)

    Returns:
        Dict mapping system name to (success, message) tuple
    """
    import logging

    logger = logging.getLogger(__name__)
    results: dict[str, tuple[bool, str]] = {}

    # Run all checks in parallel for faster startup
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(_check_openrouter, ping_glm),
            executor.submit(_check_claude_cli),
            executor.submit(_check_memory_keeper),
            executor.submit(_check_github_cli),
        ]

        for future in as_completed(futures):
            try:
                name, result = future.result(timeout=20.0)
                results[name] = result
            except Exception as e:
                # If a check itself raises, log it but don't crash
                logger.warning(f"Startup check failed: {e}")

    return results


def show_startup_panel(results: dict[str, tuple[bool, str]]) -> bool:
    """
    Display startup validation results.

    Returns:
        True if all checks passed, False otherwise
    """
    console.print()
    console.print(
        Panel.fit(
            "[bold]VIBE ORCHESTRATOR[/bold] - Startup Check",
            border_style="blue",
        )
    )
    console.print()

    all_passed = True
    for system, (success, message) in results.items():
        if success:
            console.print(f"  [green][✓][/green] {system:<20} {message}")
        else:
            console.print(f"  [red][✗][/red] {system:<20} {message}")
            all_passed = False

    console.print()
    return all_passed
