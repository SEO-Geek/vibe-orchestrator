"""External integrations - Perplexity research, GitHub operations."""

from vibe.integrations.github_ops import GitHubOps, IssueInfo, PRInfo
from vibe.integrations.research import PerplexityClient, ResearchResult

__all__ = [
    "PerplexityClient",
    "ResearchResult",
    "GitHubOps",
    "IssueInfo",
    "PRInfo",
]
