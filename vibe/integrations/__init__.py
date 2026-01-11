"""External integrations - Perplexity research, GitHub operations."""

from vibe.integrations.research import PerplexityClient, ResearchResult
from vibe.integrations.github_ops import GitHubOps, IssueInfo, PRInfo

__all__ = [
    "PerplexityClient",
    "ResearchResult",
    "GitHubOps",
    "IssueInfo",
    "PRInfo",
]
