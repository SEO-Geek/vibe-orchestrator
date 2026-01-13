"""
Vibe Pricing - Cost calculation for GLM and Claude API calls.

Provides utilities to calculate API costs based on token usage.
"""

from typing import Any

# GLM pricing via OpenRouter (per 1K tokens)
GLM_COSTS: dict[str, dict[str, float]] = {
    "z-ai/glm-4.7": {"input": 0.0003, "output": 0.0006},
    "z-ai/glm-4-plus": {"input": 0.0005, "output": 0.0010},
    "z-ai/glm-4": {"input": 0.0003, "output": 0.0006},
}

# Claude pricing (per 1K tokens) - estimates for Claude Code
CLAUDE_COSTS: dict[str, dict[str, float]] = {
    "claude-3-opus": {"input": 0.015, "output": 0.075},
    "claude-3-sonnet": {"input": 0.003, "output": 0.015},
    "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
    "claude-code": {"input": 0.003, "output": 0.015},  # Assume Sonnet-level pricing
}

# Default costs if model not found
DEFAULT_GLM_COST = {"input": 0.0003, "output": 0.0006}
DEFAULT_CLAUDE_COST = {"input": 0.003, "output": 0.015}


def calculate_glm_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> float:
    """
    Calculate cost for GLM API call.

    Args:
        model: Model identifier (e.g., 'z-ai/glm-4.7')
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens

    Returns:
        Cost in USD
    """
    costs = GLM_COSTS.get(model, DEFAULT_GLM_COST)
    return (input_tokens * costs["input"] + output_tokens * costs["output"]) / 1000


def calculate_claude_cost(
    input_tokens: int,
    output_tokens: int,
    model: str = "claude-code",
) -> float:
    """
    Calculate cost for Claude API call.

    Args:
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        model: Model identifier

    Returns:
        Cost in USD
    """
    costs = CLAUDE_COSTS.get(model, DEFAULT_CLAUDE_COST)
    return (input_tokens * costs["input"] + output_tokens * costs["output"]) / 1000


def estimate_tokens(text: str) -> int:
    """
    Estimate token count from text.

    Simple approximation: ~4 characters per token for English text.

    Args:
        text: Input text

    Returns:
        Estimated token count
    """
    return len(text) // 4


def format_cost(cost: float) -> str:
    """
    Format cost for display.

    Args:
        cost: Cost in USD

    Returns:
        Formatted string (e.g., '$0.0123')
    """
    if cost < 0.01:
        return f"${cost:.4f}"
    elif cost < 1.00:
        return f"${cost:.3f}"
    else:
        return f"${cost:.2f}"


class CostTracker:
    """Track cumulative costs across a session."""

    def __init__(self) -> None:
        self.glm_cost: float = 0.0
        self.claude_cost: float = 0.0
        self.glm_calls: int = 0
        self.claude_calls: int = 0
        self.glm_tokens: dict[str, int] = {"input": 0, "output": 0}
        self.claude_tokens: dict[str, int] = {"input": 0, "output": 0}

    def add_glm_usage(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """
        Add GLM usage and return cost.

        Args:
            model: GLM model identifier
            input_tokens: Input token count
            output_tokens: Output token count

        Returns:
            Cost for this call in USD
        """
        cost = calculate_glm_cost(model, input_tokens, output_tokens)
        self.glm_cost += cost
        self.glm_calls += 1
        self.glm_tokens["input"] += input_tokens
        self.glm_tokens["output"] += output_tokens
        return cost

    def add_claude_usage(
        self,
        input_tokens: int,
        output_tokens: int,
        model: str = "claude-code",
    ) -> float:
        """
        Add Claude usage and return cost.

        Args:
            input_tokens: Input token count
            output_tokens: Output token count
            model: Claude model identifier

        Returns:
            Cost for this call in USD
        """
        cost = calculate_claude_cost(input_tokens, output_tokens, model)
        self.claude_cost += cost
        self.claude_calls += 1
        self.claude_tokens["input"] += input_tokens
        self.claude_tokens["output"] += output_tokens
        return cost

    @property
    def total_cost(self) -> float:
        """Get total cost (GLM + Claude)."""
        return self.glm_cost + self.claude_cost

    def reset(self) -> None:
        """Reset all counters."""
        self.glm_cost = 0.0
        self.claude_cost = 0.0
        self.glm_calls = 0
        self.claude_calls = 0
        self.glm_tokens = {"input": 0, "output": 0}
        self.claude_tokens = {"input": 0, "output": 0}

    def summary(self) -> dict[str, Any]:
        """Get usage summary."""
        return {
            "glm": {
                "cost": self.glm_cost,
                "calls": self.glm_calls,
                "tokens": self.glm_tokens,
            },
            "claude": {
                "cost": self.claude_cost,
                "calls": self.claude_calls,
                "tokens": self.claude_tokens,
            },
            "total_cost": self.total_cost,
        }

    def __str__(self) -> str:
        """String representation."""
        return (
            f"GLM: {format_cost(self.glm_cost)} ({self.glm_calls} calls) + "
            f"Claude: {format_cost(self.claude_cost)} ({self.claude_calls} calls) = "
            f"{format_cost(self.total_cost)}"
        )
