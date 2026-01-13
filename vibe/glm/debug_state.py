"""
Debug State - Data structures for GLM-tracked debugging sessions.

GLM maintains full context across Claude iterations:
- Tracks every iteration of Claude's work
- Formats history for Claude's next iteration
- Formats reviews for GLM decision making
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class ClaudeIteration:
    """One iteration of Claude's debugging work."""

    iteration_num: int
    task_given: str
    output: str  # Claude's raw response
    structured_findings: dict[str, Any] = field(default_factory=dict)
    files_changed: list[str] = field(default_factory=list)
    files_examined: list[str] = field(default_factory=list)
    duration_ms: int = 0
    timestamp: datetime = field(default_factory=datetime.now)

    def format_summary(self) -> str:
        """Format iteration for history display."""
        files_str = ", ".join(self.files_changed) if self.files_changed else "none"
        return (
            f"### Iteration {self.iteration_num}\n"
            f"**Task:** {self.task_given}\n"
            f"**Files Changed:** {files_str}\n"
            f"**Duration:** {self.duration_ms}ms\n"
            f"**Output:**\n{self.output[:1000]}{'...' if len(self.output) > 1000 else ''}\n"
        )


@dataclass
class GLMReview:
    """GLM's review of a Claude iteration."""

    iteration_num: int
    approved: bool
    is_problem_solved: bool
    feedback: str
    next_task: str | None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class DebugContext:
    """
    Full context GLM tracks across debugging iterations.

    This is the "brain" state - everything GLM knows about the debugging session.
    """

    problem: str
    iterations: list[ClaudeIteration] = field(default_factory=list)
    reviews: list[GLMReview] = field(default_factory=list)
    hypothesis: str | None = None
    must_preserve: list[str] = field(default_factory=list)
    is_complete: bool = False
    started_at: datetime = field(default_factory=datetime.now)

    def add_iteration(
        self,
        task: str,
        output: str,
        files_changed: list[str] | None = None,
        duration_ms: int = 0,
    ) -> ClaudeIteration:
        """Add a new Claude iteration to the context."""
        iteration = ClaudeIteration(
            iteration_num=len(self.iterations) + 1,
            task_given=task,
            output=output,
            files_changed=files_changed or [],
            duration_ms=duration_ms,
        )
        self.iterations.append(iteration)
        return iteration

    def add_review(
        self,
        approved: bool,
        is_problem_solved: bool,
        feedback: str,
        next_task: str | None = None,
    ) -> GLMReview:
        """Add GLM's review of the latest iteration."""
        review = GLMReview(
            iteration_num=len(self.iterations),
            approved=approved,
            is_problem_solved=is_problem_solved,
            feedback=feedback,
            next_task=next_task,
        )
        self.reviews.append(review)

        if is_problem_solved:
            self.is_complete = True

        return review

    def get_latest_iteration(self) -> ClaudeIteration | None:
        """Get the most recent Claude iteration."""
        return self.iterations[-1] if self.iterations else None

    def get_latest_review(self) -> GLMReview | None:
        """Get the most recent GLM review."""
        return self.reviews[-1] if self.reviews else None

    def format_for_claude(self) -> str:
        """
        Format complete history for Claude's next iteration.

        This gives Claude FULL context of what's been tried.
        """
        parts = [
            "## DEBUGGING SESSION CONTEXT",
            "",
            f"**Problem:** {self.problem}",
            "",
        ]

        if self.hypothesis:
            parts.append(f"**Current Hypothesis:** {self.hypothesis}")
            parts.append("")

        if self.must_preserve:
            parts.append("**Features That Must Keep Working:**")
            for feature in self.must_preserve:
                parts.append(f"- {feature}")
            parts.append("")

        if self.iterations:
            parts.append("## PREVIOUS ITERATIONS")
            parts.append("")
            for iteration in self.iterations:
                parts.append(iteration.format_summary())
                # Add corresponding review if exists
                matching_reviews = [
                    r for r in self.reviews if r.iteration_num == iteration.iteration_num
                ]
                if matching_reviews:
                    review = matching_reviews[0]
                    status = "APPROVED" if review.approved else "NEEDS WORK"
                    parts.append(f"**GLM Review:** {status}")
                    parts.append(f"**Feedback:** {review.feedback}")
                parts.append("")

        parts.append("---")
        parts.append("")

        return "\n".join(parts)

    def format_for_glm_review(self) -> str:
        """
        Format latest iteration for GLM review.

        This is what GLM sees when deciding approve/reject.
        """
        latest = self.get_latest_iteration()
        if not latest:
            return "No iterations to review."

        parts = [
            "## REVIEW CLAUDE'S WORK",
            "",
            f"**Original Problem:** {self.problem}",
            "",
            f"**Task Given to Claude:** {latest.task_given}",
            "",
            "**Claude's Output:**",
            latest.output,
            "",
        ]

        if latest.files_changed:
            parts.append(f"**Files Modified:** {', '.join(latest.files_changed)}")
            parts.append("")

        if self.must_preserve:
            parts.append("**Must Verify These Still Work:**")
            for feature in self.must_preserve:
                parts.append(f"- {feature}")
            parts.append("")

        # Show iteration history summary
        if len(self.iterations) > 1:
            parts.append(f"**This is iteration {len(self.iterations)}** - previous attempts:")
            for prev in self.iterations[:-1]:
                matching_reviews = [
                    r for r in self.reviews if r.iteration_num == prev.iteration_num
                ]
                status = (
                    "approved" if matching_reviews and matching_reviews[0].approved else "rejected"
                )
                parts.append(f"  - Iteration {prev.iteration_num}: {status}")
            parts.append("")

        return "\n".join(parts)

    def format_iterations_summary(self) -> str:
        """Brief summary of all iterations for task generation."""
        if not self.iterations:
            return "No previous attempts."

        summaries = []
        for iteration in self.iterations:
            matching_reviews = [
                r for r in self.reviews if r.iteration_num == iteration.iteration_num
            ]
            if matching_reviews:
                review = matching_reviews[0]
                result = "succeeded" if review.approved else f"failed: {review.feedback[:100]}"
            else:
                result = "pending review"
            summaries.append(
                f"- Attempt {iteration.iteration_num}: {iteration.task_given[:80]}... ({result})"
            )

        return "\n".join(summaries)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for storage."""
        return {
            "problem": self.problem,
            "hypothesis": self.hypothesis,
            "must_preserve": self.must_preserve,
            "is_complete": self.is_complete,
            "started_at": self.started_at.isoformat(),
            "iterations": [
                {
                    "iteration_num": it.iteration_num,
                    "task_given": it.task_given,
                    "output": it.output,
                    "files_changed": it.files_changed,
                    "duration_ms": it.duration_ms,
                    "timestamp": it.timestamp.isoformat(),
                }
                for it in self.iterations
            ],
            "reviews": [
                {
                    "iteration_num": r.iteration_num,
                    "approved": r.approved,
                    "is_problem_solved": r.is_problem_solved,
                    "feedback": r.feedback,
                    "next_task": r.next_task,
                    "timestamp": r.timestamp.isoformat(),
                }
                for r in self.reviews
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DebugContext":
        """Deserialize from storage."""
        context = cls(
            problem=data["problem"],
            hypothesis=data.get("hypothesis"),
            must_preserve=data.get("must_preserve", []),
            is_complete=data.get("is_complete", False),
        )

        if "started_at" in data:
            context.started_at = datetime.fromisoformat(data["started_at"])

        for it_data in data.get("iterations", []):
            iteration = ClaudeIteration(
                iteration_num=it_data["iteration_num"],
                task_given=it_data["task_given"],
                output=it_data["output"],
                files_changed=it_data.get("files_changed", []),
                duration_ms=it_data.get("duration_ms", 0),
            )
            if "timestamp" in it_data:
                iteration.timestamp = datetime.fromisoformat(it_data["timestamp"])
            context.iterations.append(iteration)

        for r_data in data.get("reviews", []):
            review = GLMReview(
                iteration_num=r_data["iteration_num"],
                approved=r_data["approved"],
                is_problem_solved=r_data["is_problem_solved"],
                feedback=r_data["feedback"],
                next_task=r_data.get("next_task"),
            )
            if "timestamp" in r_data:
                review.timestamp = datetime.fromisoformat(r_data["timestamp"])
            context.reviews.append(review)

        return context
