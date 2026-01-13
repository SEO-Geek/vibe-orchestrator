"""
Workflow Templates - Multi-phase pipelines for different task types.

Defines workflow phases and templates that guide task decomposition.
Each workflow template specifies the phases to execute for a given task type.
"""

from dataclasses import dataclass, field
from enum import Enum

from vibe.orchestrator.task_enforcer import TaskType


class PhaseType(Enum):
    """Types of workflow phases in the execution pipeline."""

    ANALYZE = "analyze"  # Read code, understand dependencies
    DESIGN = "design"  # Plan the approach
    IMPLEMENT = "implement"  # Write the actual code
    DOCUMENT = "document"  # Add comments, update docs
    VERIFY = "verify"  # Run tests, verify behavior
    CLEANUP = "cleanup"  # Remove dead code, format


@dataclass
class WorkflowPhase:
    """
    A single phase in a workflow pipeline.

    Each phase defines what Claude should focus on, which tools to use,
    and how to determine success.
    """

    name: str  # e.g., "analyze_dependencies"
    phase_type: PhaseType  # Category of phase
    description: str  # Human-readable description
    required_tools: list[str] = field(default_factory=list)  # Tools that MUST be used
    recommended_agents: list[str] = field(default_factory=list)  # e.g., ["perplexity", "context7"]
    success_criteria: str = ""  # How to verify phase completion
    timeout_tier: str = "code"  # quick (30s), code (2m), debug (5m), research (10m)
    can_skip: bool = False  # Whether phase can be skipped if not needed
    prompt_guidance: str = ""  # Additional prompt text for this phase


@dataclass
class WorkflowTemplate:
    """
    A complete workflow template for a specific task type.

    Templates define the sequence of phases to execute for tasks
    like feature development, debugging, refactoring, etc.
    """

    task_type: TaskType  # The task type this template handles
    name: str  # Human-readable name
    description: str  # What this workflow is for
    phases: list[WorkflowPhase] = field(default_factory=list)


# =============================================================================
# WORKFLOW PHASE DEFINITIONS
# Reusable phase definitions used across multiple workflows
# =============================================================================

# Analysis phases - understand before modifying
PHASE_ANALYZE_CONTEXT = WorkflowPhase(
    name="analyze_context",
    phase_type=PhaseType.ANALYZE,
    description="Read relevant files and understand existing patterns",
    required_tools=["Read", "Grep", "Glob"],
    recommended_agents=["context7"],
    success_criteria="Identified relevant files, understood code structure",
    timeout_tier="code",
    can_skip=False,
    prompt_guidance="Before making any changes, read the relevant files and understand the existing patterns. Identify dependencies and potential impact areas.",
)

PHASE_ANALYZE_DEPENDENCIES = WorkflowPhase(
    name="analyze_dependencies",
    phase_type=PhaseType.ANALYZE,
    description="Find all files that import or depend on the target",
    required_tools=["Grep", "Glob"],
    success_criteria="Listed all import references and dependent files",
    timeout_tier="quick",
    can_skip=True,
    prompt_guidance="Search for all imports and usages of the code being modified. List files that may need updates.",
)

PHASE_REPRODUCE_BUG = WorkflowPhase(
    name="reproduce_bug",
    phase_type=PhaseType.ANALYZE,
    description="Reproduce the bug to understand the issue",
    recommended_agents=["perplexity"],
    success_criteria="Bug reproduced and behavior documented",
    timeout_tier="debug",
    can_skip=False,
    prompt_guidance="First, understand and reproduce the bug. Document the exact steps and observed behavior.",
)

PHASE_INVESTIGATE = WorkflowPhase(
    name="investigate",
    phase_type=PhaseType.ANALYZE,
    description="Trace the issue to find root cause",
    required_tools=["Read", "Grep"],
    success_criteria="Root cause identified with evidence",
    timeout_tier="debug",
    can_skip=False,
    prompt_guidance="Trace through the code to find the root cause. Look for stack traces, error messages, and code paths.",
)

# Implementation phases - make the changes
PHASE_IMPLEMENT = WorkflowPhase(
    name="implement",
    phase_type=PhaseType.IMPLEMENT,
    description="Write the code changes",
    required_tools=["Edit", "Write"],
    success_criteria="Code changes complete and syntactically correct",
    timeout_tier="code",
    can_skip=False,
    prompt_guidance="Implement the required changes. Follow existing patterns in the codebase.",
)

PHASE_FIX_BUG = WorkflowPhase(
    name="fix_bug",
    phase_type=PhaseType.IMPLEMENT,
    description="Apply the fix for the identified bug",
    required_tools=["Edit"],
    success_criteria="Fix applied to address root cause",
    timeout_tier="code",
    can_skip=False,
    prompt_guidance="Apply the fix based on the root cause analysis. Ensure the fix is minimal and targeted.",
)

PHASE_REFACTOR = WorkflowPhase(
    name="refactor",
    phase_type=PhaseType.IMPLEMENT,
    description="Perform the refactoring",
    required_tools=["Edit", "Grep"],
    success_criteria="Refactoring complete, all references updated",
    timeout_tier="code",
    can_skip=False,
    prompt_guidance="Perform the refactoring. Update all references and imports. Preserve existing behavior.",
)

# Documentation phases - make code maintainable
PHASE_ADD_COMMENTS = WorkflowPhase(
    name="add_comments",
    phase_type=PhaseType.DOCUMENT,
    description="Add inline comments explaining complex logic",
    required_tools=["Edit"],
    success_criteria="Complex logic documented with comments",
    timeout_tier="quick",
    can_skip=True,
    prompt_guidance="Add inline comments for any complex or non-obvious logic. Focus on 'why' not 'what'.",
)

PHASE_UPDATE_DOCS = WorkflowPhase(
    name="update_docs",
    phase_type=PhaseType.DOCUMENT,
    description="Update documentation files (README, STARMAP, etc.)",
    required_tools=["Edit"],
    success_criteria="Documentation reflects current state",
    timeout_tier="quick",
    can_skip=True,
    prompt_guidance="Update any relevant documentation to reflect the changes made.",
)

# Verification phases - ensure correctness
PHASE_VERIFY = WorkflowPhase(
    name="verify",
    phase_type=PhaseType.VERIFY,
    description="Run tests to verify no regressions",
    required_tools=["Bash"],
    success_criteria="Tests pass without regressions",
    timeout_tier="code",
    can_skip=False,
    prompt_guidance="Run the project's test suite to verify no regressions. Check that existing functionality still works.",
)

PHASE_VERIFY_FIX = WorkflowPhase(
    name="verify_fix",
    phase_type=PhaseType.VERIFY,
    description="Verify the fix actually resolves the bug",
    required_tools=["Bash"],
    success_criteria="Bug no longer reproducible",
    timeout_tier="debug",
    can_skip=False,
    prompt_guidance="Verify that the fix resolves the original bug. Try to reproduce the issue again.",
)

PHASE_VERIFY_BEHAVIOR = WorkflowPhase(
    name="verify_behavior",
    phase_type=PhaseType.VERIFY,
    description="Verify refactoring preserved behavior",
    required_tools=["Bash"],
    success_criteria="Behavior unchanged, tests pass",
    timeout_tier="code",
    can_skip=False,
    prompt_guidance="Run tests to verify the refactoring didn't change behavior. Check all use cases still work.",
)

PHASE_ADD_TEST = WorkflowPhase(
    name="add_test",
    phase_type=PhaseType.VERIFY,
    description="Add a test to prevent regression",
    required_tools=["Write", "Bash"],
    success_criteria="New test added and passing",
    timeout_tier="code",
    can_skip=True,
    prompt_guidance="Add a test case that would have caught this bug, to prevent future regressions.",
)

# Research phases
PHASE_GATHER_INFO = WorkflowPhase(
    name="gather_info",
    phase_type=PhaseType.ANALYZE,
    description="Research and gather information",
    recommended_agents=["perplexity", "context7"],
    success_criteria="Relevant information gathered from documentation",
    timeout_tier="research",
    can_skip=False,
    prompt_guidance="Research the topic using available documentation and web resources. Gather relevant examples.",
)

PHASE_SUMMARIZE = WorkflowPhase(
    name="summarize",
    phase_type=PhaseType.DOCUMENT,
    description="Summarize findings for the user",
    success_criteria="Clear, concise summary provided",
    timeout_tier="quick",
    can_skip=False,
    prompt_guidance="Provide a clear summary of findings. Include key takeaways and actionable recommendations.",
)


# =============================================================================
# WORKFLOW TEMPLATES
# Pre-built workflows for common task types
# =============================================================================

WORKFLOW_TEMPLATES: dict[TaskType, WorkflowTemplate] = {
    # Feature development: analyze → implement → document → verify
    TaskType.CODE_WRITE: WorkflowTemplate(
        task_type=TaskType.CODE_WRITE,
        name="Feature Development",
        description="Build new features with proper analysis, implementation, documentation, and testing",
        phases=[
            PHASE_ANALYZE_CONTEXT,
            PHASE_IMPLEMENT,
            PHASE_ADD_COMMENTS,
            PHASE_VERIFY,
        ],
    ),
    # Bug fixing: reproduce → investigate → fix → verify → add test
    TaskType.DEBUG: WorkflowTemplate(
        task_type=TaskType.DEBUG,
        name="Bug Fixing",
        description="Debug issues systematically with reproduction, investigation, fix, and verification",
        phases=[
            PHASE_REPRODUCE_BUG,
            PHASE_INVESTIGATE,
            PHASE_FIX_BUG,
            PHASE_VERIFY_FIX,
            PHASE_ADD_TEST,
        ],
    ),
    # Refactoring: analyze impact → refactor → verify behavior
    TaskType.CODE_REFACTOR: WorkflowTemplate(
        task_type=TaskType.CODE_REFACTOR,
        name="Code Refactoring",
        description="Refactor code safely by analyzing impact and verifying preserved behavior",
        phases=[
            PHASE_ANALYZE_DEPENDENCIES,
            PHASE_REFACTOR,
            PHASE_VERIFY_BEHAVIOR,
        ],
    ),
    # Research: gather info → summarize
    TaskType.RESEARCH: WorkflowTemplate(
        task_type=TaskType.RESEARCH,
        name="Research",
        description="Research topics and provide summaries",
        phases=[
            PHASE_GATHER_INFO,
            PHASE_SUMMARIZE,
        ],
    ),
    # Browser/UI testing: setup and test
    TaskType.UI_TEST: WorkflowTemplate(
        task_type=TaskType.UI_TEST,
        name="UI Testing",
        description="Test user interfaces with real browser interaction",
        phases=[
            WorkflowPhase(
                name="setup_test",
                phase_type=PhaseType.ANALYZE,
                description="Navigate to the page and prepare test environment",
                required_tools=["mcp__playwright__playwright_navigate"],
                success_criteria="Page loaded and ready for testing",
                timeout_tier="code",
            ),
            WorkflowPhase(
                name="execute_test",
                phase_type=PhaseType.VERIFY,
                description="Execute the UI test with real browser interactions",
                required_tools=[
                    "mcp__playwright__playwright_click",
                    "mcp__playwright__playwright_fill",
                    "mcp__playwright__playwright_screenshot",
                ],
                success_criteria="Test steps executed and results captured",
                timeout_tier="debug",
            ),
            WorkflowPhase(
                name="capture_evidence",
                phase_type=PhaseType.DOCUMENT,
                description="Take screenshots and document results",
                required_tools=["mcp__playwright__playwright_screenshot"],
                success_criteria="Visual evidence captured",
                timeout_tier="quick",
            ),
        ],
    ),
    # API testing
    TaskType.API_TEST: WorkflowTemplate(
        task_type=TaskType.API_TEST,
        name="API Testing",
        description="Test API endpoints thoroughly",
        phases=[
            WorkflowPhase(
                name="test_endpoints",
                phase_type=PhaseType.VERIFY,
                description="Test API endpoints with various inputs",
                required_tools=[
                    "mcp__playwright__playwright_get",
                    "mcp__playwright__playwright_post",
                ],
                success_criteria="All endpoints tested with expected responses",
                timeout_tier="code",
            ),
            WorkflowPhase(
                name="verify_responses",
                phase_type=PhaseType.VERIFY,
                description="Verify response structure and content",
                success_criteria="Response structure matches expectations",
                timeout_tier="quick",
            ),
        ],
    ),
    # Database operations: careful with transactions
    TaskType.DATABASE: WorkflowTemplate(
        task_type=TaskType.DATABASE,
        name="Database Operations",
        description="Database operations with proper backup and verification",
        phases=[
            WorkflowPhase(
                name="backup_check",
                phase_type=PhaseType.ANALYZE,
                description="Check current state before modifications",
                required_tools=["mcp__sqlite__read_query"],
                success_criteria="Current state documented",
                timeout_tier="quick",
            ),
            PHASE_IMPLEMENT,
            WorkflowPhase(
                name="verify_data",
                phase_type=PhaseType.VERIFY,
                description="Verify data integrity after changes",
                required_tools=["mcp__sqlite__read_query"],
                success_criteria="Data integrity verified",
                timeout_tier="quick",
            ),
        ],
    ),
    # Browser work
    TaskType.BROWSER_WORK: WorkflowTemplate(
        task_type=TaskType.BROWSER_WORK,
        name="Browser Work",
        description="Browser-based tasks with visual verification",
        phases=[
            WorkflowPhase(
                name="navigate",
                phase_type=PhaseType.IMPLEMENT,
                description="Navigate and interact with pages",
                required_tools=["mcp__playwright__playwright_navigate"],
                success_criteria="Successfully navigated to target",
                timeout_tier="code",
            ),
            WorkflowPhase(
                name="capture",
                phase_type=PhaseType.DOCUMENT,
                description="Capture visual evidence",
                required_tools=["mcp__playwright__playwright_screenshot"],
                success_criteria="Screenshot captured as evidence",
                timeout_tier="quick",
            ),
        ],
    ),
    # Docker operations
    TaskType.DOCKER: WorkflowTemplate(
        task_type=TaskType.DOCKER,
        name="Docker Operations",
        description="Container operations with proper verification",
        phases=[
            WorkflowPhase(
                name="check_state",
                phase_type=PhaseType.ANALYZE,
                description="Check current container state",
                required_tools=["Bash"],
                success_criteria="Current container state documented",
                timeout_tier="quick",
            ),
            PHASE_IMPLEMENT,
            WorkflowPhase(
                name="verify_health",
                phase_type=PhaseType.VERIFY,
                description="Verify container health",
                required_tools=["Bash"],
                success_criteria="Container healthy and running",
                timeout_tier="code",
            ),
        ],
    ),
    # General fallback - minimal workflow
    TaskType.GENERAL: WorkflowTemplate(
        task_type=TaskType.GENERAL,
        name="General",
        description="Generic task workflow",
        phases=[
            PHASE_IMPLEMENT,
        ],
    ),
}


def get_workflow_template(task_type: TaskType) -> WorkflowTemplate:
    """
    Get the workflow template for a task type.

    Args:
        task_type: The task type to get template for

    Returns:
        WorkflowTemplate for the task type, or GENERAL template if not found
    """
    return WORKFLOW_TEMPLATES.get(task_type, WORKFLOW_TEMPLATES[TaskType.GENERAL])


def get_phase_prompt_section(phase: WorkflowPhase) -> str:
    """
    Generate a prompt section for a workflow phase.

    This text is added to Claude's task prompt to guide execution.

    Args:
        phase: The workflow phase

    Returns:
        Formatted prompt section text
    """
    parts = [f"\n## PHASE: {phase.name.upper().replace('_', ' ')}"]
    parts.append(f"**Objective**: {phase.description}")

    if phase.prompt_guidance:
        parts.append(f"\n{phase.prompt_guidance}")

    if phase.required_tools:
        parts.append(f"\n**Required Tools**: {', '.join(phase.required_tools)}")

    if phase.recommended_agents:
        parts.append(f"**Recommended Agents/MCPs**: {', '.join(phase.recommended_agents)}")

    if phase.success_criteria:
        parts.append(f"\n**Success Criteria**: {phase.success_criteria}")

    return "\n".join(parts)
