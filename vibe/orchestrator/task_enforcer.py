"""
Task Enforcer - Tool Requirements and Verification

Detects task types and enforces tool usage rules to prevent
Claude from taking shortcuts (like using curl instead of real browsers).

Key Features:
- Task type detection from description with confidence scoring
- Required/forbidden tool lists per task type
- Completion checklist generation
- Post-execution verification of tool usage
- SmartTaskDetector with intent pattern matching
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# INTENT PATTERNS FOR SMART DETECTION
# Patterns with confidence scores for more accurate task type detection
# =============================================================================

# Each pattern is a tuple of (regex_pattern, confidence_score)
# Confidence ranges from 0.0 to 1.0, with 1.0 being absolute certainty
INTENT_PATTERNS: dict[str, list[tuple[str, float]]] = {
    "debug": [
        (r"(is|are)\s+(broken|failing|not working)", 0.95),
        (r"(error|exception|crash|stack\s*trace)", 0.85),
        (r"(bug|issue|problem)\s+(in|with|when)", 0.85),
        (r"why\s+(is|does|doesn't|won't)", 0.80),
        (r"(fix|debug|troubleshoot|diagnose)", 0.90),
        (r"(doesn't|does not|won't|cannot)\s+work", 0.85),
        (r"(investigate|find out why)", 0.80),
    ],
    "code_write": [
        (r"(create|implement|add|build)\s+(a|an|the|new)", 0.90),
        (r"(write|develop)\s+(a|an|the|new)", 0.90),
        (r"(add|implement)\s+(function|class|method|feature)", 0.90),
        (r"(new|create)\s+(endpoint|api|route)", 0.85),
        (r"(write|generate)\s+(code|script)", 0.85),
    ],
    "code_refactor": [
        (r"(refactor|restructure|reorganize)", 0.95),
        (r"(rename|move)\s+(the|this)?\s*(function|class|method|variable)", 0.90),
        (r"(extract|split)\s+(into|to)", 0.85),
        (r"(clean\s*up|simplify|improve)\s+(the|this)?\s*code", 0.80),
        (r"(consolidate|merge)\s+(the|these)", 0.80),
    ],
    "research": [
        (r"(research|look\s*up|find\s*out)\s+(about|how|what)", 0.90),
        (r"(what\s+is|how\s+does|how\s+do\s+i)", 0.85),
        (r"(documentation|docs)\s+(for|about)", 0.85),
        (r"(best\s+practice|recommended\s+way)", 0.85),
        (r"(compare|difference\s+between)", 0.80),
    ],
    "ui_test": [
        (r"(test|verify)\s+(the|this)?\s*(ui|interface|page|form)", 0.95),
        (r"(browser|e2e|end-to-end)\s+test", 0.95),
        (r"(check|verify)\s+in\s+(the|a)?\s*browser", 0.90),
        (r"(screenshot|visual)\s+(test|verify)", 0.90),
        (r"(click|fill|submit)\s+(the|a)?\s*(button|form|input)", 0.85),
    ],
    "api_test": [
        (r"(test|verify)\s+(the|this)?\s*(api|endpoint|route)", 0.95),
        (r"(http|rest|graphql)\s+test", 0.90),
        (r"(test|check)\s+(the|this)?\s*(response|status)", 0.85),
    ],
    "database": [
        (r"(database|db|sql|query|table|schema|migration)", 0.85),
        (r"(select|insert|update|delete)\s+from", 0.90),
        (r"(postgres|mysql|sqlite|mongodb)", 0.85),
    ],
    "docker": [
        (r"(docker|container|compose|kubernetes|k8s)", 0.90),
        (r"(build|run|deploy)\s+(the|a)?\s*container", 0.85),
        (r"(dockerfile|docker-compose)", 0.95),
    ],
    "browser_work": [
        (r"(navigate|go)\s+to\s+(url|page|site)", 0.90),
        (r"(open|view)\s+(in|the)?\s*browser", 0.90),
        (r"(browse|visit)\s+(the|this)?\s*(page|site|url)", 0.85),
    ],
}


@dataclass
class TaskTypeDetection:
    """Result of smart task type detection with confidence score."""

    task_type: "TaskType"
    confidence: float  # 0.0 to 1.0
    matched_pattern: str  # The pattern that matched
    detection_method: str  # "intent_pattern", "keyword", or "default"

    @property
    def is_confident(self) -> bool:
        """Check if detection has high confidence (>= 0.7)."""
        return self.confidence >= 0.7

    @property
    def needs_confirmation(self) -> bool:
        """Check if detection should be confirmed by GLM (< 0.6)."""
        return self.confidence < 0.6


class TaskType(Enum):
    """Types of tasks with different tool requirements."""

    UI_TEST = "ui_test"  # Testing UI in browser
    API_TEST = "api_test"  # Testing API endpoints
    BROWSER_WORK = "browser_work"  # Any browser-related work
    CODE_WRITE = "code_write"  # Writing new code
    CODE_REFACTOR = "code_refactor"  # Refactoring existing code
    DEBUG = "debug"  # Debugging issues
    RESEARCH = "research"  # Researching documentation/solutions
    DATABASE = "database"  # Database operations
    DOCKER = "docker"  # Container operations
    GENERAL = "general"  # Default fallback


@dataclass
class ToolRequirement:
    """Tool requirements for a task type."""

    task_type: TaskType
    required_tools: list[str] = field(default_factory=list)
    preferred_tools: list[str] = field(default_factory=list)
    forbidden_tools: list[str] = field(default_factory=list)
    forbidden_patterns: list[str] = field(default_factory=list)  # Regex patterns
    completion_checklist: list[str] = field(default_factory=list)
    description: str = ""


# Tool requirement definitions
TOOL_REQUIREMENTS: dict[TaskType, ToolRequirement] = {
    TaskType.UI_TEST: ToolRequirement(
        task_type=TaskType.UI_TEST,
        description="Testing user interface in real browser",
        required_tools=[
            "mcp__playwright__playwright_navigate",
            "mcp__playwright__playwright_screenshot",
        ],
        preferred_tools=[
            "mcp__playwright__playwright_console_logs",
            "mcp__playwright__playwright_click",
            "mcp__playwright__playwright_fill",
            "mcp__playwright__playwright_get_visible_text",
            "mcp__chrome-devtools__take_screenshot",
            "mcp__chrome-devtools__take_snapshot",
        ],
        forbidden_tools=["curl", "wget", "httpie"],
        forbidden_patterns=[
            r"curl\s+",
            r"wget\s+",
            r"requests\.(get|post|put|delete)",
            r"fetch\(",
            r"axios\.",
        ],
        completion_checklist=[
            "Take screenshot and VISUALLY VERIFY the result",
            "Check browser console for JavaScript errors",
            "Test actual user interaction flow (click, type, submit)",
            "Verify elements are visible and interactive",
            "Do NOT claim complete without screenshot evidence",
        ],
    ),
    TaskType.API_TEST: ToolRequirement(
        task_type=TaskType.API_TEST,
        description="Testing API endpoints",
        required_tools=[],
        preferred_tools=[
            "mcp__playwright__playwright_get",
            "mcp__playwright__playwright_post",
            "mcp__playwright__playwright_put",
            "mcp__playwright__playwright_delete",
        ],
        forbidden_tools=[],
        forbidden_patterns=[],
        completion_checklist=[
            "Test all relevant HTTP methods",
            "Verify response status codes",
            "Check response body structure",
            "Test error cases",
        ],
    ),
    TaskType.BROWSER_WORK: ToolRequirement(
        task_type=TaskType.BROWSER_WORK,
        description="Any work involving browser interaction",
        required_tools=[
            "mcp__playwright__playwright_navigate",
        ],
        preferred_tools=[
            "mcp__playwright__playwright_screenshot",
            "mcp__playwright__playwright_console_logs",
            "mcp__chrome-devtools__take_screenshot",
            "mcp__chrome-devtools__list_console_messages",
        ],
        forbidden_tools=["curl", "wget"],
        forbidden_patterns=[r"curl\s+", r"wget\s+"],
        completion_checklist=[
            "Use real browser, not command-line HTTP tools",
            "Capture screenshot as evidence",
            "Check for console errors",
        ],
    ),
    TaskType.DEBUG: ToolRequirement(
        task_type=TaskType.DEBUG,
        description="Debugging and troubleshooting",
        required_tools=[],
        preferred_tools=[
            "mcp__playwright__playwright_console_logs",
            "mcp__chrome-devtools__list_console_messages",
            "mcp__chrome-devtools__list_network_requests",
        ],
        forbidden_tools=[],
        forbidden_patterns=[],
        completion_checklist=[
            "Identify root cause, not just symptoms",
            "Check logs and console output",
            "Verify the fix actually works",
            "Test edge cases that might trigger same issue",
        ],
    ),
    TaskType.RESEARCH: ToolRequirement(
        task_type=TaskType.RESEARCH,
        description="Researching documentation and solutions",
        required_tools=[],
        preferred_tools=[
            "mcp__perplexity__search",
            "mcp__perplexity__reason",
            "mcp__context7__resolve-library-id",
            "mcp__context7__query-docs",
            "WebSearch",
            "WebFetch",
        ],
        forbidden_tools=[],
        forbidden_patterns=[],
        completion_checklist=[
            "Use up-to-date documentation sources",
            "Verify information is current (check dates)",
            "Cross-reference multiple sources",
        ],
    ),
    TaskType.DATABASE: ToolRequirement(
        task_type=TaskType.DATABASE,
        description="Database operations",
        required_tools=[],
        preferred_tools=[],
        forbidden_tools=[],
        forbidden_patterns=[],
        completion_checklist=[
            "Back up data before destructive operations",
            "Use transactions where appropriate",
            "Verify changes with SELECT after INSERT/UPDATE",
        ],
    ),
    TaskType.DOCKER: ToolRequirement(
        task_type=TaskType.DOCKER,
        description="Docker and container operations",
        required_tools=[],
        preferred_tools=[],
        forbidden_tools=[],
        forbidden_patterns=[],
        completion_checklist=[
            "Check container logs after operations",
            "Verify container health status",
            "Clean up unused resources",
        ],
    ),
    TaskType.CODE_WRITE: ToolRequirement(
        task_type=TaskType.CODE_WRITE,
        description="Writing new code",
        required_tools=[],
        preferred_tools=["Read", "Write", "Edit"],
        forbidden_tools=[],
        forbidden_patterns=[],
        completion_checklist=[
            "Add inline comments for complex logic",
            "Follow existing code patterns in the project",
            "Do not change code outside task scope",
        ],
    ),
    TaskType.CODE_REFACTOR: ToolRequirement(
        task_type=TaskType.CODE_REFACTOR,
        description="Refactoring existing code",
        required_tools=["Read"],
        preferred_tools=["Edit", "Grep", "Glob"],
        forbidden_tools=[],
        forbidden_patterns=[],
        completion_checklist=[
            "Read existing code before modifying",
            "Preserve existing behavior",
            "Update all references when renaming",
            "Run tests after refactoring",
        ],
    ),
    TaskType.GENERAL: ToolRequirement(
        task_type=TaskType.GENERAL,
        description="General tasks",
        required_tools=[],
        preferred_tools=[],
        forbidden_tools=[],
        forbidden_patterns=[],
        completion_checklist=[],
    ),
}


# Keywords for task type detection
TASK_TYPE_KEYWORDS: dict[TaskType, list[str]] = {
    TaskType.UI_TEST: [
        "test ui",
        "test the ui",
        "ui test",
        "frontend test",
        "browser test",
        "e2e test",
        "end-to-end test",
        "integration test",
        "verify in browser",
        "check in browser",
        "test the page",
        "test the form",
        "test the button",
        "visual test",
        "screenshot test",
    ],
    TaskType.API_TEST: [
        "test api",
        "test endpoint",
        "api test",
        "test the route",
        "test the handler",
        "http test",
        "rest test",
    ],
    TaskType.BROWSER_WORK: [
        "open browser",
        "navigate to",
        "go to url",
        "check the website",
        "visit the page",
        "browse to",
        "open the page",
        "look at the page",
        "see the page",
        "view in browser",
    ],
    TaskType.DEBUG: [
        "debug",
        "fix bug",
        "fix error",
        "troubleshoot",
        "investigate",
        "find the issue",
        "what's wrong",
        "why is it",
        "not working",
        "broken",
        "failing",
    ],
    TaskType.RESEARCH: [
        "research",
        "look up",
        "find out",
        "how do i",
        "how to",
        "what is",
        "documentation",
        "docs for",
        "best practice",
    ],
    TaskType.DATABASE: [
        "database",
        "sql",
        "query",
        "migration",
        "schema",
        "table",
        "postgres",
        "mysql",
        "sqlite",
        "mongodb",
    ],
    TaskType.DOCKER: [
        "docker",
        "container",
        "compose",
        "kubernetes",
        "k8s",
        "image",
        "dockerfile",
    ],
    TaskType.CODE_REFACTOR: [
        "refactor",
        "rename",
        "restructure",
        "reorganize",
        "clean up",
        "simplify",
        "extract",
        "move to",
    ],
}


class TaskEnforcer:
    """
    Detects task types and generates tool enforcement rules.

    Used by GLM to ensure Claude uses appropriate tools for each task.
    """

    def __init__(self, global_conventions: list[str] | None = None):
        """
        Initialize task enforcer.

        Args:
            global_conventions: List of global "we always do X" rules
        """
        self.global_conventions = global_conventions or []

    def detect_task_type(self, task_description: str) -> TaskType:
        """
        Detect the task type from its description.

        Args:
            task_description: The task description text

        Returns:
            Detected TaskType enum value
        """
        description_lower = task_description.lower()

        # Check each task type's keywords
        for task_type, keywords in TASK_TYPE_KEYWORDS.items():
            for keyword in keywords:
                if keyword in description_lower:
                    logger.debug(f"Detected task type {task_type.value} from keyword '{keyword}'")
                    return task_type

        return TaskType.GENERAL

    def get_requirements(self, task_type: TaskType) -> ToolRequirement:
        """Get tool requirements for a task type."""
        return TOOL_REQUIREMENTS.get(task_type, TOOL_REQUIREMENTS[TaskType.GENERAL])

    def generate_enforcement_prompt(
        self,
        task_description: str,
        task_type: TaskType | None = None,
    ) -> str:
        """
        Generate the tool enforcement section for Claude's prompt.

        Args:
            task_description: The task description
            task_type: Override detected task type

        Returns:
            Formatted enforcement text to append to prompt
        """
        if task_type is None:
            task_type = self.detect_task_type(task_description)

        req = self.get_requirements(task_type)
        parts = []

        # Task type header
        parts.append(f"\n## TASK TYPE: {req.description.upper()}")

        # Required tools
        if req.required_tools:
            parts.append("\n### REQUIRED TOOLS (you MUST use these):")
            for tool in req.required_tools:
                parts.append(f"  - {tool}")

        # Preferred tools
        if req.preferred_tools:
            parts.append("\n### PREFERRED TOOLS (use when applicable):")
            for tool in req.preferred_tools:
                parts.append(f"  - {tool}")

        # Forbidden tools
        if req.forbidden_tools or req.forbidden_patterns:
            parts.append("\n### FORBIDDEN (do NOT use these):")
            for tool in req.forbidden_tools:
                parts.append(f"  - {tool}")
            if req.forbidden_patterns:
                parts.append("  - Command-line HTTP tools for browser testing")
                parts.append("  - Direct HTTP requests when browser interaction is needed")

        # Completion checklist
        if req.completion_checklist:
            parts.append("\n### BEFORE MARKING COMPLETE:")
            for i, item in enumerate(req.completion_checklist, 1):
                parts.append(f"  {i}. {item}")

        # Global conventions
        if self.global_conventions:
            parts.append("\n### PROJECT CONVENTIONS (always follow):")
            for convention in self.global_conventions:
                parts.append(f"  - {convention}")

        return "\n".join(parts)

    def verify_tool_usage(
        self,
        task_description: str,
        tool_calls: list[dict[str, Any]],
        task_type: TaskType | None = None,
    ) -> dict[str, Any]:
        """
        Verify that required tools were used.

        Args:
            task_description: The task description
            tool_calls: List of tool calls made by Claude
            task_type: Override detected task type

        Returns:
            Verification result with passed, missing_tools, violations
        """
        if task_type is None:
            task_type = self.detect_task_type(task_description)

        req = self.get_requirements(task_type)

        # Get tool names used
        tools_used = {call.get("name", "") for call in tool_calls}

        # Check required tools
        missing_required = []
        for tool in req.required_tools:
            if tool not in tools_used:
                missing_required.append(tool)

        # Check forbidden tools
        forbidden_used = []
        for tool in req.forbidden_tools:
            if tool in tools_used:
                forbidden_used.append(tool)

        # Check forbidden patterns in Bash commands
        pattern_violations = []
        for call in tool_calls:
            if call.get("name") == "Bash":
                command = call.get("input", {}).get("command", "")
                for pattern in req.forbidden_patterns:
                    if re.search(pattern, command):
                        pattern_violations.append(f"Used forbidden pattern in Bash: {pattern}")

        passed = len(missing_required) == 0 and len(forbidden_used) == 0 and len(pattern_violations) == 0

        return {
            "passed": passed,
            "task_type": task_type.value,
            "missing_required": missing_required,
            "forbidden_used": forbidden_used,
            "pattern_violations": pattern_violations,
            "tools_used": list(tools_used),
            "feedback": self._generate_feedback(missing_required, forbidden_used, pattern_violations),
        }

    def _generate_feedback(
        self,
        missing: list[str],
        forbidden: list[str],
        violations: list[str],
    ) -> str:
        """Generate human-readable feedback for violations."""
        parts = []

        if missing:
            parts.append(f"Missing required tools: {', '.join(missing)}")

        if forbidden:
            parts.append(f"Used forbidden tools: {', '.join(forbidden)}")

        if violations:
            parts.append("Pattern violations:")
            for v in violations:
                parts.append(f"  - {v}")

        if not parts:
            return "All tool requirements satisfied."

        return "\n".join(parts)

    def get_available_mcp_tools(self) -> dict[str, list[str]]:
        """
        Get list of available MCP tools by category.

        This is informational, to be included in task context.
        """
        return {
            "browser_automation": [
                "mcp__playwright__playwright_navigate",
                "mcp__playwright__playwright_screenshot",
                "mcp__playwright__playwright_click",
                "mcp__playwright__playwright_fill",
                "mcp__playwright__playwright_console_logs",
                "mcp__playwright__playwright_get_visible_text",
                "mcp__playwright__playwright_get_visible_html",
            ],
            "chrome_devtools": [
                "mcp__chrome-devtools__take_screenshot",
                "mcp__chrome-devtools__take_snapshot",
                "mcp__chrome-devtools__click",
                "mcp__chrome-devtools__fill",
                "mcp__chrome-devtools__list_console_messages",
                "mcp__chrome-devtools__list_network_requests",
                "mcp__chrome-devtools__navigate_page",
            ],
            "research": [
                "mcp__perplexity__search",
                "mcp__perplexity__reason",
                "mcp__perplexity__deep_research",
                "mcp__context7__resolve-library-id",
                "mcp__context7__query-docs",
            ],
            "github": [
                "mcp__github__list_issues",
                "mcp__github__create_issue",
                "mcp__github__create_pull_request",
                "mcp__github__get_pull_request",
            ],
            "memory": [
                "mcp__memory__create_entities",
                "mcp__memory__search_nodes",
                "mcp__memory-keeper__context_save",
                "mcp__memory-keeper__context_get",
            ],
            "filesystem": [
                "mcp__filesystem__read_file",
                "mcp__filesystem__write_file",
                "mcp__filesystem__list_directory",
                "mcp__filesystem__search_files",
            ],
            "git": [
                "mcp__git__git_status",
                "mcp__git__git_diff",
                "mcp__git__git_commit",
                "mcp__git__git_log",
            ],
        }


# =============================================================================
# CACHED INSTANCES (Singleton pattern for performance)
# Prevents redundant object creation on every convenience function call
# =============================================================================

# Module-level cached instances (lazy initialization)
_cached_enforcer: TaskEnforcer | None = None
_cached_detector: "SmartTaskDetector | None" = None


def get_task_enforcer(global_conventions: list[str] | None = None) -> TaskEnforcer:
    """
    Get cached TaskEnforcer instance for performance.

    Creates instance on first call, reuses on subsequent calls.
    Pass global_conventions only on first call or when conventions change.

    Args:
        global_conventions: Optional list of conventions (only used on first call)

    Returns:
        Cached TaskEnforcer instance
    """
    global _cached_enforcer
    if _cached_enforcer is None:
        _cached_enforcer = TaskEnforcer(global_conventions)
    return _cached_enforcer


def get_smart_detector() -> "SmartTaskDetector":
    """
    Get cached SmartTaskDetector instance for performance.

    Creates instance on first call (compiles all regex patterns),
    reuses on subsequent calls to avoid redundant compilation.

    Returns:
        Cached SmartTaskDetector instance
    """
    global _cached_detector
    if _cached_detector is None:
        _cached_detector = SmartTaskDetector()
    return _cached_detector


def reset_cached_instances() -> None:
    """
    Reset cached instances.

    Call this if you need to recreate instances with different settings,
    such as updated global conventions.
    """
    global _cached_enforcer, _cached_detector
    _cached_enforcer = None
    _cached_detector = None
    logger.debug("Reset cached TaskEnforcer and SmartTaskDetector instances")


# Convenience function for quick task type detection (uses cached instance)
def detect_task_type(description: str) -> TaskType:
    """Quick function to detect task type from description."""
    enforcer = get_task_enforcer()
    return enforcer.detect_task_type(description)


# =============================================================================
# SMART TASK DETECTOR
# Enhanced detection with confidence scoring and intent patterns
# =============================================================================

# Mapping from intent pattern keys to TaskType enum values
INTENT_TO_TASK_TYPE: dict[str, TaskType] = {
    "debug": TaskType.DEBUG,
    "code_write": TaskType.CODE_WRITE,
    "code_refactor": TaskType.CODE_REFACTOR,
    "research": TaskType.RESEARCH,
    "ui_test": TaskType.UI_TEST,
    "api_test": TaskType.API_TEST,
    "database": TaskType.DATABASE,
    "docker": TaskType.DOCKER,
    "browser_work": TaskType.BROWSER_WORK,
}


class SmartTaskDetector:
    """
    Enhanced task type detector with confidence scoring.

    Uses multiple detection strategies:
    1. Intent patterns (highest priority) - Regex patterns with confidence scores
    2. Keyword matching (fallback) - Simple keyword detection
    3. Default (lowest priority) - Returns GENERAL with low confidence

    The confidence score helps decide when to ask GLM for clarification.
    """

    def __init__(self):
        """Initialize detector with pre-compiled patterns."""
        # Pre-compile all intent patterns for performance
        self._compiled_patterns: dict[str, list[tuple[re.Pattern, float]]] = {}
        for intent_key, patterns in INTENT_PATTERNS.items():
            self._compiled_patterns[intent_key] = [
                (re.compile(pattern, re.IGNORECASE), confidence)
                for pattern, confidence in patterns
            ]

    def detect(self, description: str) -> TaskTypeDetection:
        """
        Detect task type with confidence scoring.

        Uses intent patterns first (higher confidence), then falls back
        to keyword matching (medium confidence), then defaults to GENERAL.

        Args:
            description: The task description to analyze

        Returns:
            TaskTypeDetection with type, confidence, and method used
        """
        # Strategy 1: Intent pattern matching (highest confidence)
        best_match: TaskTypeDetection | None = None

        for intent_key, compiled_patterns in self._compiled_patterns.items():
            task_type = INTENT_TO_TASK_TYPE.get(intent_key, TaskType.GENERAL)

            for pattern, confidence in compiled_patterns:
                match = pattern.search(description)
                if match:
                    current = TaskTypeDetection(
                        task_type=task_type,
                        confidence=confidence,
                        matched_pattern=pattern.pattern,
                        detection_method="intent_pattern",
                    )
                    # Keep highest confidence match
                    if best_match is None or current.confidence > best_match.confidence:
                        best_match = current
                        logger.debug(
                            f"Intent pattern match: {task_type.value} "
                            f"(confidence={confidence:.2f})"
                        )

        if best_match is not None:
            return best_match

        # Strategy 2: Keyword matching (medium confidence)
        keyword_result = self._detect_by_keyword(description)
        if keyword_result.task_type != TaskType.GENERAL:
            return keyword_result

        # Strategy 3: Default to GENERAL with low confidence
        return TaskTypeDetection(
            task_type=TaskType.GENERAL,
            confidence=0.3,
            matched_pattern="",
            detection_method="default",
        )

    def _detect_by_keyword(self, description: str) -> TaskTypeDetection:
        """
        Fall back to keyword-based detection.

        Uses the existing TASK_TYPE_KEYWORDS dictionary for matching.
        Returns medium confidence (0.5-0.6) for keyword matches.

        Args:
            description: The task description

        Returns:
            TaskTypeDetection with keyword-based result
        """
        description_lower = description.lower()

        for task_type, keywords in TASK_TYPE_KEYWORDS.items():
            for keyword in keywords:
                if keyword in description_lower:
                    return TaskTypeDetection(
                        task_type=task_type,
                        confidence=0.55,  # Medium confidence for keyword match
                        matched_pattern=keyword,
                        detection_method="keyword",
                    )

        return TaskTypeDetection(
            task_type=TaskType.GENERAL,
            confidence=0.3,
            matched_pattern="",
            detection_method="default",
        )

    def get_all_matches(self, description: str) -> list[TaskTypeDetection]:
        """
        Get all matching task types with their confidence scores.

        Useful for debugging or when GLM needs to choose between options.

        Args:
            description: The task description

        Returns:
            List of all matching TaskTypeDetection, sorted by confidence
        """
        matches = []

        # Check all intent patterns
        for intent_key, compiled_patterns in self._compiled_patterns.items():
            task_type = INTENT_TO_TASK_TYPE.get(intent_key, TaskType.GENERAL)

            for pattern, confidence in compiled_patterns:
                if pattern.search(description):
                    matches.append(TaskTypeDetection(
                        task_type=task_type,
                        confidence=confidence,
                        matched_pattern=pattern.pattern,
                        detection_method="intent_pattern",
                    ))

        # Check keyword matches
        description_lower = description.lower()
        for task_type, keywords in TASK_TYPE_KEYWORDS.items():
            for keyword in keywords:
                if keyword in description_lower:
                    matches.append(TaskTypeDetection(
                        task_type=task_type,
                        confidence=0.55,
                        matched_pattern=keyword,
                        detection_method="keyword",
                    ))
                    break  # Only one match per task type from keywords

        # Sort by confidence (highest first)
        matches.sort(key=lambda x: x.confidence, reverse=True)
        return matches


# Convenience function for smart detection (uses cached instance)
def smart_detect_task_type(description: str) -> TaskTypeDetection:
    """
    Quick function to detect task type with confidence scoring.

    Uses cached SmartTaskDetector instance for performance.

    Args:
        description: Task description

    Returns:
        TaskTypeDetection with type, confidence, and method
    """
    detector = get_smart_detector()
    return detector.detect(description)
