"""
Workflow Engine - Orchestrates workflow expansion and phase management.

Expands tasks into multi-phase workflows and manages phase execution context.
This is the main entry point for the intelligent orchestration system.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from vibe.orchestrator.task_enforcer import TaskEnforcer, TaskType
from vibe.orchestrator.workflows.injector import SubTaskInjector
from vibe.orchestrator.workflows.templates import (
    PhaseType,
    WorkflowPhase,
    get_phase_prompt_section,
    get_workflow_template,
)

logger = logging.getLogger(__name__)


# Patterns that indicate simple tasks that don't need workflow expansion
SIMPLE_TASK_PATTERNS = [
    r"^(read|show|display|list|get|view|check)\s+",  # Read-only operations
    r"^what\s+(is|are)\s+",  # Questions
    r"^(where|how|why)\s+",  # Information queries
    r"^(tell|explain|describe)\s+",  # Explanations
]


@dataclass
class ExpandedTask:
    """
    A task that has been expanded with workflow phase information.

    Contains the original task plus phase-specific guidance and constraints.
    """

    id: str
    description: str
    phase: WorkflowPhase | None = None  # The workflow phase this represents
    phase_type: PhaseType | None = None  # Phase type for categorization
    phase_badge: str = ""  # Display badge like "[ANALYZE]"
    original_task_id: str = ""  # Parent task if this was expanded
    files: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    success_criteria: str = ""
    required_tools: list[str] = field(default_factory=list)
    recommended_agents: list[str] = field(default_factory=list)
    timeout_tier: str = "code"
    injected: bool = False  # True if auto-injected sub-task

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for JSON serialization."""
        return {
            "id": self.id,
            "description": self.description,
            "phase_badge": self.phase_badge,
            "phase_type": self.phase_type.value if self.phase_type else None,
            "original_task_id": self.original_task_id,
            "files": self.files,
            "constraints": self.constraints,
            "success_criteria": self.success_criteria,
            "required_tools": self.required_tools,
            "recommended_agents": self.recommended_agents,
            "timeout_tier": self.timeout_tier,
            "injected": self.injected,
        }


class WorkflowEngine:
    """
    Engine for expanding tasks into multi-phase workflows.

    Takes a task and its detected type, then expands it into a sequence
    of workflow phases with appropriate tools, constraints, and guidance.
    """

    def __init__(
        self,
        enable_workflows: bool = True,
        enable_injection: bool = True,
        task_enforcer: TaskEnforcer | None = None,
    ):
        """
        Initialize the workflow engine.

        Args:
            enable_workflows: Whether to expand tasks into workflow phases
            enable_injection: Whether to inject sub-tasks based on content
            task_enforcer: TaskEnforcer instance for type detection
        """
        self.enable_workflows = enable_workflows
        self.enable_injection = enable_injection
        self.task_enforcer = task_enforcer or TaskEnforcer()
        self.injector = SubTaskInjector()

        # Compile simple task patterns
        self._simple_patterns = [re.compile(p, re.IGNORECASE) for p in SIMPLE_TASK_PATTERNS]

    def should_expand_to_workflow(self, task_description: str) -> bool:
        """
        Determine if a task should be expanded into a full workflow.

        Simple read/show tasks skip workflow expansion.

        Args:
            task_description: The task description

        Returns:
            True if task should be expanded to workflow phases
        """
        if not self.enable_workflows:
            return False

        # Check for simple task patterns
        for pattern in self._simple_patterns:
            if pattern.match(task_description):
                logger.debug(f"Task matches simple pattern, skipping workflow: {task_description[:50]}")
                return False

        return True

    def expand_task_to_workflow(
        self,
        task: dict[str, Any],
        task_type: TaskType | None = None,
    ) -> list[ExpandedTask]:
        """
        Expand a single task into workflow phases.

        Args:
            task: Task dict with id, description, files, constraints
            task_type: Override detected task type

        Returns:
            List of ExpandedTask objects representing workflow phases
        """
        description = task.get("description", "")
        task_id = task.get("id", "task-1")

        # Detect task type if not provided
        if task_type is None:
            task_type = self.task_enforcer.detect_task_type(description)

        # Check if this task should be expanded
        if not self.should_expand_to_workflow(description):
            # Return as-is, just converted to ExpandedTask
            return [
                ExpandedTask(
                    id=task_id,
                    description=description,
                    files=task.get("files", []),
                    constraints=task.get("constraints", []),
                    success_criteria=task.get("success_criteria", ""),
                    timeout_tier="code",
                )
            ]

        # Get workflow template
        template = get_workflow_template(task_type)
        logger.info(f"Expanding task to '{template.name}' workflow with {len(template.phases)} phases")

        # Create expanded tasks for each phase
        expanded = []
        for i, phase in enumerate(template.phases):
            phase_task = ExpandedTask(
                id=f"{task_id}-{phase.name}",
                description=f"{phase.description}: {description}",
                phase=phase,
                phase_type=phase.phase_type,
                phase_badge=f"[{phase.phase_type.value.upper()}]",
                original_task_id=task_id,
                files=task.get("files", []),
                constraints=task.get("constraints", []),
                success_criteria=phase.success_criteria,
                required_tools=phase.required_tools,
                recommended_agents=phase.recommended_agents,
                timeout_tier=phase.timeout_tier,
            )
            expanded.append(phase_task)

        return expanded

    def process_tasks(
        self,
        tasks: list[dict[str, Any]],
        expand_to_phases: bool = True,
    ) -> list[ExpandedTask]:
        """
        Process a list of tasks, optionally expanding to workflow phases.

        Args:
            tasks: List of task dicts from GLM decomposition
            expand_to_phases: Whether to expand tasks to workflow phases

        Returns:
            List of ExpandedTask objects ready for execution
        """
        # First, inject sub-tasks if enabled
        if self.enable_injection:
            tasks = self.injector.process_task_list(tasks)
            logger.info(f"After injection: {len(tasks)} tasks")

        # Then, expand to workflow phases if enabled
        if expand_to_phases and self.enable_workflows:
            expanded = []
            for task in tasks:
                # Skip expansion for already-injected tasks
                if task.get("injected"):
                    expanded.append(
                        ExpandedTask(
                            id=task.get("id", "injected-task"),
                            description=task.get("description", ""),
                            files=task.get("files", []),
                            constraints=task.get("constraints", []),
                            injected=True,
                            timeout_tier="quick",
                        )
                    )
                else:
                    expanded.extend(self.expand_task_to_workflow(task))
            return expanded

        # Just convert to ExpandedTask without workflow expansion
        return [
            ExpandedTask(
                id=task.get("id", f"task-{i}"),
                description=task.get("description", ""),
                files=task.get("files", []),
                constraints=task.get("constraints", []),
                success_criteria=task.get("success_criteria", ""),
                injected=task.get("injected", False),
            )
            for i, task in enumerate(tasks)
        ]

    def get_phase_prompt(self, expanded_task: ExpandedTask) -> str:
        """
        Generate prompt guidance for a specific workflow phase.

        Args:
            expanded_task: The expanded task with phase info

        Returns:
            Prompt text to add to Claude's task instruction
        """
        if not expanded_task.phase:
            return ""

        return get_phase_prompt_section(expanded_task.phase)

    def get_recommended_tools_prompt(self, expanded_task: ExpandedTask) -> str:
        """
        Generate prompt section about recommended tools.

        Args:
            expanded_task: The expanded task

        Returns:
            Prompt text about tools to use
        """
        parts = []

        if expanded_task.required_tools:
            tools_str = ", ".join(expanded_task.required_tools)
            parts.append(f"\n**REQUIRED Tools**: You MUST use: {tools_str}")

        if expanded_task.recommended_agents:
            agents_str = ", ".join(expanded_task.recommended_agents)
            parts.append(f"\n**Recommended MCPs/Agents**: Consider: {agents_str}")

        return "\n".join(parts)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def expand_tasks(
    tasks: list[dict[str, Any]],
    enable_workflows: bool = True,
    enable_injection: bool = True,
) -> list[ExpandedTask]:
    """
    Convenience function to expand a list of tasks.

    Args:
        tasks: List of task dicts from GLM
        enable_workflows: Enable workflow phase expansion
        enable_injection: Enable sub-task injection

    Returns:
        List of ExpandedTask objects
    """
    engine = WorkflowEngine(
        enable_workflows=enable_workflows,
        enable_injection=enable_injection,
    )
    return engine.process_tasks(tasks)


def get_workflow_summary(task_type: TaskType) -> str:
    """
    Get a human-readable summary of a workflow.

    Args:
        task_type: The task type

    Returns:
        Summary string describing the workflow
    """
    template = get_workflow_template(task_type)
    phases = [f"[{p.phase_type.value.upper()}] {p.name}" for p in template.phases]
    return f"{template.name}: {' → '.join(phases)}"
