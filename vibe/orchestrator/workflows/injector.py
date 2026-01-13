"""
Sub-Task Injector - Automatically inject relevant sub-tasks based on context.

Analyzes task descriptions and injects additional tasks that should be performed:
- Writing code → add comments, run tests
- Fixing bugs → verify fix, check regressions
- Refactoring → analyze usages first
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class InjectionRule:
    """
    A rule for injecting sub-tasks based on trigger patterns.

    When a task description matches the trigger pattern, the specified
    sub-tasks are injected before or after the main task.
    """

    name: str  # Rule identifier
    trigger_pattern: str  # Regex pattern to match task descriptions
    inject_before: list[str] = field(default_factory=list)  # Tasks to add before
    inject_after: list[str] = field(default_factory=list)  # Tasks to add after
    constraints: list[str] = field(default_factory=list)  # Constraints to add to main task
    priority: int = 0  # Higher priority rules take precedence


# =============================================================================
# INJECTION RULES
# Rules for automatically injecting sub-tasks based on task content
# =============================================================================

INJECTION_RULES: list[InjectionRule] = [
    # Writing new code → add comments and run tests
    InjectionRule(
        name="code_creation",
        trigger_pattern=r"(write|create|implement|add|build)\s+(a\s+)?(new\s+)?(function|class|method|module|component|feature)",
        inject_after=[
            "Add inline comments explaining any complex logic",
            "Run tests to verify no regressions",
        ],
        constraints=["Follow existing code patterns in the project"],
        priority=10,
    ),
    # Bug fixing → verify fix and check for regressions
    InjectionRule(
        name="bug_fixing",
        trigger_pattern=r"(fix|debug|resolve|repair|patch)\s+(the\s+)?(bug|error|issue|problem|crash|exception)",
        inject_after=[
            "Verify the fix resolves the original issue",
            "Check for regressions in related functionality",
        ],
        constraints=["Document root cause in code comments"],
        priority=10,
    ),
    # Refactoring → analyze usages first
    InjectionRule(
        name="refactoring",
        trigger_pattern=r"(refactor|rename|restructure|reorganize|move|extract)\s+",
        inject_before=[
            "Analyze all usages and dependencies of the target code",
        ],
        inject_after=[
            "Update all references and imports",
            "Run tests to verify behavior is preserved",
        ],
        constraints=["Preserve existing behavior"],
        priority=10,
    ),
    # API changes → update documentation
    InjectionRule(
        name="api_changes",
        trigger_pattern=r"(add|change|modify|update)\s+(the\s+)?(api|endpoint|route|handler)",
        inject_after=[
            "Update API documentation if applicable",
        ],
        constraints=["Maintain backward compatibility where possible"],
        priority=5,
    ),
    # Database changes → backup first
    InjectionRule(
        name="database_changes",
        trigger_pattern=r"(modify|update|delete|drop|alter)\s+(the\s+)?(database|table|schema|migration)",
        inject_before=[
            "Check current database state before modifications",
        ],
        inject_after=[
            "Verify data integrity after changes",
        ],
        constraints=["Use transactions for multi-step operations"],
        priority=10,
    ),
    # Test writing → run the tests
    InjectionRule(
        name="test_writing",
        trigger_pattern=r"(write|create|add)\s+(a\s+)?(new\s+)?(test|spec|unit|integration)",
        inject_after=[
            "Run the new test to verify it passes",
        ],
        constraints=["Test should cover edge cases"],
        priority=5,
    ),
    # Configuration changes → validate config
    InjectionRule(
        name="config_changes",
        trigger_pattern=r"(update|modify|change)\s+(the\s+)?(config|configuration|settings|env|environment)",
        inject_after=[
            "Verify configuration is valid and application starts correctly",
        ],
        priority=5,
    ),
    # Security-related changes → audit
    InjectionRule(
        name="security_changes",
        trigger_pattern=r"(auth|authentication|authorization|permission|security|password|credential|token|secret)",
        inject_after=[
            "Verify no security vulnerabilities introduced",
        ],
        constraints=[
            "Never log sensitive data",
            "Use secure coding practices",
        ],
        priority=15,
    ),
    # UI changes → visual verification
    InjectionRule(
        name="ui_changes",
        trigger_pattern=r"(ui|interface|component|button|form|page|modal|dialog|css|style)",
        inject_after=[
            "Take a screenshot to verify visual appearance",
        ],
        constraints=["Test in real browser, not command line"],
        priority=5,
    ),
    # Any code modification → generic verification
    InjectionRule(
        name="generic_code_changes",
        trigger_pattern=r"(edit|modify|update|change)\s+",
        inject_after=[
            "Run tests if available",
        ],
        priority=1,  # Low priority, only applies if no other rule matches
    ),
]


class SubTaskInjector:
    """
    Injects additional sub-tasks based on task content analysis.

    Examines task descriptions and applies injection rules to add
    relevant preparation and verification tasks.
    """

    def __init__(self, rules: list[InjectionRule] | None = None):
        """
        Initialize the injector with rules.

        Args:
            rules: Custom injection rules. Defaults to INJECTION_RULES.
        """
        self.rules = rules or INJECTION_RULES
        # Sort by priority (highest first)
        self.rules = sorted(self.rules, key=lambda r: r.priority, reverse=True)
        # Pre-compile patterns for performance
        self._compiled_patterns: dict[str, re.Pattern] = {}
        for rule in self.rules:
            self._compiled_patterns[rule.name] = re.compile(rule.trigger_pattern, re.IGNORECASE)

    def get_matching_rules(self, task_description: str) -> list[InjectionRule]:
        """
        Find all rules that match the task description.

        Args:
            task_description: The task description to analyze

        Returns:
            List of matching rules, sorted by priority
        """
        matching = []
        for rule in self.rules:
            pattern = self._compiled_patterns[rule.name]
            if pattern.search(task_description):
                matching.append(rule)
                logger.debug(f"Rule '{rule.name}' matched task: {task_description[:50]}")
        return matching

    def inject_subtasks(
        self,
        task: dict[str, Any],
        max_injections: int = 3,
    ) -> list[dict[str, Any]]:
        """
        Inject sub-tasks around a main task based on matching rules.

        Args:
            task: The main task dict (with 'id', 'description', etc.)
            max_injections: Maximum number of sub-tasks to inject

        Returns:
            List of tasks with injections (before, main, after)
        """
        description = task.get("description", "")
        matching_rules = self.get_matching_rules(description)

        if not matching_rules:
            return [task]

        # Collect unique before/after tasks from all matching rules
        before_tasks: list[str] = []
        after_tasks: list[str] = []
        added_constraints: list[str] = []

        for rule in matching_rules:
            for t in rule.inject_before:
                if t not in before_tasks:
                    before_tasks.append(t)
            for t in rule.inject_after:
                if t not in after_tasks:
                    after_tasks.append(t)
            for c in rule.constraints:
                if c not in added_constraints:
                    added_constraints.append(c)

        # Limit injections to prevent bloat
        before_tasks = before_tasks[:max_injections]
        after_tasks = after_tasks[:max_injections]

        # Build result list
        result = []
        task_id = task.get("id", "task-1")

        # Before tasks
        for i, before_desc in enumerate(before_tasks):
            result.append(
                {
                    "id": f"{task_id}-pre-{i + 1}",
                    "description": before_desc,
                    "files": task.get("files", []),
                    "constraints": [],
                    "injected": True,
                    "injection_type": "before",
                    "parent_task": task_id,
                }
            )

        # Main task with added constraints
        main_task = task.copy()
        existing_constraints = main_task.get("constraints", [])
        main_task["constraints"] = existing_constraints + added_constraints
        result.append(main_task)

        # After tasks
        for i, after_desc in enumerate(after_tasks):
            result.append(
                {
                    "id": f"{task_id}-post-{i + 1}",
                    "description": after_desc,
                    "files": task.get("files", []),
                    "constraints": [],
                    "injected": True,
                    "injection_type": "after",
                    "parent_task": task_id,
                }
            )

        logger.info(
            f"Injected {len(before_tasks)} before + {len(after_tasks)} after tasks "
            f"for '{description[:50]}...'"
        )
        return result

    def process_task_list(
        self,
        tasks: list[dict[str, Any]],
        max_injections_per_task: int = 2,
    ) -> list[dict[str, Any]]:
        """
        Process a list of tasks and inject sub-tasks for each.

        Args:
            tasks: List of task dicts
            max_injections_per_task: Max injections per task

        Returns:
            Expanded list with injected tasks
        """
        result = []
        for task in tasks:
            # Skip already-injected tasks to prevent infinite expansion
            if task.get("injected"):
                result.append(task)
                continue
            expanded = self.inject_subtasks(task, max_injections=max_injections_per_task)
            result.extend(expanded)
        return result


# Convenience function for quick injection
def inject_subtasks(task: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Quick function to inject sub-tasks for a single task.

    Args:
        task: Task dict with at least 'description' key

    Returns:
        List of tasks with injections
    """
    injector = SubTaskInjector()
    return injector.inject_subtasks(task)
