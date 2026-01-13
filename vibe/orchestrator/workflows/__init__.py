"""
Vibe Orchestrator Workflows - Intelligent task orchestration.

This module provides:
- WorkflowTemplate: Multi-phase pipelines for different task types
- SubTaskInjector: Automatic sub-task injection based on task content
- WorkflowEngine: Main orchestration engine for task expansion

Usage:
    from vibe.orchestrator.workflows import WorkflowEngine, expand_tasks

    # Expand GLM tasks into workflow phases
    engine = WorkflowEngine()
    expanded = engine.process_tasks(tasks)

    # Or use the convenience function
    expanded = expand_tasks(tasks)
"""

from vibe.orchestrator.workflows.templates import (
    PhaseType,
    WorkflowPhase,
    WorkflowTemplate,
    WORKFLOW_TEMPLATES,
    get_workflow_template,
    get_phase_prompt_section,
)

from vibe.orchestrator.workflows.injector import (
    InjectionRule,
    SubTaskInjector,
    INJECTION_RULES,
    inject_subtasks,
)

from vibe.orchestrator.workflows.engine import (
    ExpandedTask,
    WorkflowEngine,
    expand_tasks,
    get_workflow_summary,
)

__all__ = [
    # Templates
    "PhaseType",
    "WorkflowPhase",
    "WorkflowTemplate",
    "WORKFLOW_TEMPLATES",
    "get_workflow_template",
    "get_phase_prompt_section",
    # Injector
    "InjectionRule",
    "SubTaskInjector",
    "INJECTION_RULES",
    "inject_subtasks",
    # Engine
    "ExpandedTask",
    "WorkflowEngine",
    "expand_tasks",
    "get_workflow_summary",
]
