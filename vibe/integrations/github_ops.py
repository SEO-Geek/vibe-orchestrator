"""
GitHub Operations Integration

Provides GitHub operations via the gh CLI for:
- Issue management
- Pull request creation
- Branch operations
"""

import json
import logging
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from vibe.exceptions import GitHubError

logger = logging.getLogger(__name__)


@dataclass
class IssueInfo:
    """Information about a GitHub issue."""

    number: int
    title: str
    body: str = ""
    state: str = "open"
    labels: list[str] = field(default_factory=list)
    assignees: list[str] = field(default_factory=list)
    created_at: datetime | None = None
    url: str = ""


@dataclass
class PRInfo:
    """Information about a GitHub pull request."""

    number: int
    title: str
    body: str = ""
    state: str = "open"
    head: str = ""
    base: str = ""
    draft: bool = False
    mergeable: bool = True
    created_at: datetime | None = None
    url: str = ""


class GitHubOps:
    """
    GitHub operations via gh CLI.

    Uses the GitHub CLI (gh) for all operations,
    which handles authentication automatically.
    """

    def __init__(self, repo: str | None = None):
        """
        Initialize GitHub operations.

        Args:
            repo: Repository in owner/repo format (auto-detected if not provided)
        """
        self.repo = repo
        self._authenticated = False

    def check_auth(self) -> bool:
        """Check if gh CLI is authenticated."""
        try:
            result = subprocess.run(
                ["gh", "auth", "status"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            self._authenticated = result.returncode == 0
            return self._authenticated
        except FileNotFoundError:
            logger.error("gh CLI not found - install with: sudo apt install gh")
            return False
        except subprocess.TimeoutExpired:
            logger.error("gh auth status timed out")
            return False

    def _run_gh(
        self,
        args: list[str],
        timeout: int = 30,
        check: bool = True,
    ) -> subprocess.CompletedProcess:
        """Run a gh CLI command."""
        cmd = ["gh"] + args

        # Add repo flag if specified
        if self.repo and "--repo" not in args and "-R" not in args:
            cmd.extend(["--repo", self.repo])

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            if check and result.returncode != 0:
                raise GitHubError(f"gh command failed: {result.stderr}")

            return result

        except FileNotFoundError:
            raise GitHubError("gh CLI not found")
        except subprocess.TimeoutExpired:
            raise GitHubError(f"gh command timed out after {timeout}s")

    # Issue Operations

    def list_issues(
        self,
        state: str = "open",
        limit: int = 30,
        labels: list[str] | None = None,
    ) -> list[IssueInfo]:
        """
        List repository issues.

        Args:
            state: Issue state (open, closed, all)
            limit: Maximum issues to return
            labels: Filter by labels

        Returns:
            List of IssueInfo objects
        """
        args = [
            "issue",
            "list",
            "--state",
            state,
            "--limit",
            str(limit),
            "--json",
            "number,title,body,state,labels,assignees,createdAt,url",
        ]

        if labels:
            for label in labels:
                args.extend(["--label", label])

        result = self._run_gh(args)
        issues_data = json.loads(result.stdout) if result.stdout else []

        issues = []
        for data in issues_data:
            issues.append(
                IssueInfo(
                    number=data["number"],
                    title=data["title"],
                    body=data.get("body", ""),
                    state=data.get("state", "open"),
                    labels=[l["name"] for l in data.get("labels", [])],
                    assignees=[a["login"] for a in data.get("assignees", [])],
                    created_at=datetime.fromisoformat(data["createdAt"].replace("Z", "+00:00"))
                    if data.get("createdAt")
                    else None,
                    url=data.get("url", ""),
                )
            )

        return issues

    def get_issue(self, number: int) -> IssueInfo:
        """Get a specific issue by number."""
        args = [
            "issue",
            "view",
            str(number),
            "--json",
            "number,title,body,state,labels,assignees,createdAt,url",
        ]

        result = self._run_gh(args)
        data = json.loads(result.stdout)

        return IssueInfo(
            number=data["number"],
            title=data["title"],
            body=data.get("body", ""),
            state=data.get("state", "open"),
            labels=[l["name"] for l in data.get("labels", [])],
            assignees=[a["login"] for a in data.get("assignees", [])],
            created_at=datetime.fromisoformat(data["createdAt"].replace("Z", "+00:00"))
            if data.get("createdAt")
            else None,
            url=data.get("url", ""),
        )

    def create_issue(
        self,
        title: str,
        body: str = "",
        labels: list[str] | None = None,
        assignees: list[str] | None = None,
    ) -> IssueInfo:
        """
        Create a new issue.

        Args:
            title: Issue title
            body: Issue body (markdown)
            labels: Labels to add
            assignees: Users to assign

        Returns:
            Created IssueInfo
        """
        args = ["issue", "create", "--title", title]

        if body:
            args.extend(["--body", body])

        if labels:
            for label in labels:
                args.extend(["--label", label])

        if assignees:
            for assignee in assignees:
                args.extend(["--assignee", assignee])

        # Return JSON
        args.append("--json")
        args.append("number,title,url")

        result = self._run_gh(args)
        data = json.loads(result.stdout)

        return IssueInfo(
            number=data["number"],
            title=data["title"],
            url=data.get("url", ""),
        )

    def close_issue(self, number: int, comment: str | None = None) -> bool:
        """Close an issue."""
        if comment:
            # Add comment first
            self._run_gh(["issue", "comment", str(number), "--body", comment])

        result = self._run_gh(["issue", "close", str(number)], check=False)
        return result.returncode == 0

    # Pull Request Operations

    def list_prs(
        self,
        state: str = "open",
        limit: int = 30,
    ) -> list[PRInfo]:
        """
        List pull requests.

        Args:
            state: PR state (open, closed, merged, all)
            limit: Maximum PRs to return

        Returns:
            List of PRInfo objects
        """
        args = [
            "pr",
            "list",
            "--state",
            state,
            "--limit",
            str(limit),
            "--json",
            "number,title,body,state,headRefName,baseRefName,isDraft,mergeable,createdAt,url",
        ]

        result = self._run_gh(args)
        prs_data = json.loads(result.stdout) if result.stdout else []

        prs = []
        for data in prs_data:
            prs.append(
                PRInfo(
                    number=data["number"],
                    title=data["title"],
                    body=data.get("body", ""),
                    state=data.get("state", "open"),
                    head=data.get("headRefName", ""),
                    base=data.get("baseRefName", ""),
                    draft=data.get("isDraft", False),
                    mergeable=data.get("mergeable", "") == "MERGEABLE",
                    created_at=datetime.fromisoformat(data["createdAt"].replace("Z", "+00:00"))
                    if data.get("createdAt")
                    else None,
                    url=data.get("url", ""),
                )
            )

        return prs

    def get_pr(self, number: int) -> PRInfo:
        """Get a specific pull request by number."""
        args = [
            "pr",
            "view",
            str(number),
            "--json",
            "number,title,body,state,headRefName,baseRefName,isDraft,mergeable,createdAt,url",
        ]

        result = self._run_gh(args)
        data = json.loads(result.stdout)

        return PRInfo(
            number=data["number"],
            title=data["title"],
            body=data.get("body", ""),
            state=data.get("state", "open"),
            head=data.get("headRefName", ""),
            base=data.get("baseRefName", ""),
            draft=data.get("isDraft", False),
            mergeable=data.get("mergeable", "") == "MERGEABLE",
            created_at=datetime.fromisoformat(data["createdAt"].replace("Z", "+00:00"))
            if data.get("createdAt")
            else None,
            url=data.get("url", ""),
        )

    def create_pr(
        self,
        title: str,
        body: str = "",
        base: str = "main",
        head: str | None = None,
        draft: bool = False,
    ) -> PRInfo:
        """
        Create a pull request.

        Args:
            title: PR title
            body: PR body (markdown)
            base: Base branch (default: main)
            head: Head branch (default: current branch)
            draft: Create as draft PR

        Returns:
            Created PRInfo
        """
        args = ["pr", "create", "--title", title, "--base", base]

        if body:
            args.extend(["--body", body])

        if head:
            args.extend(["--head", head])

        if draft:
            args.append("--draft")

        # Return JSON
        args.append("--json")
        args.append("number,title,url")

        result = self._run_gh(args)
        data = json.loads(result.stdout)

        return PRInfo(
            number=data["number"],
            title=data["title"],
            url=data.get("url", ""),
            base=base,
            head=head or "",
            draft=draft,
        )

    def merge_pr(
        self,
        number: int,
        method: str = "squash",
        delete_branch: bool = True,
    ) -> bool:
        """
        Merge a pull request.

        Args:
            number: PR number
            method: Merge method (merge, squash, rebase)
            delete_branch: Delete branch after merge

        Returns:
            True if merged successfully
        """
        args = ["pr", "merge", str(number), f"--{method}"]

        if delete_branch:
            args.append("--delete-branch")

        result = self._run_gh(args, check=False)
        return result.returncode == 0

    # Branch Operations

    def create_branch(self, name: str, base: str | None = None) -> bool:
        """
        Create a new branch.

        Args:
            name: Branch name
            base: Base branch (default: current)

        Returns:
            True if created successfully
        """
        # Use git for branch creation (gh doesn't have this)
        cmd = ["git", "checkout", "-b", name]

        if base:
            cmd.append(base)

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result.returncode == 0
        except Exception as e:
            logger.error(f"Failed to create branch: {e}")
            return False

    def sync_repo(self) -> bool:
        """Sync the repository with remote."""
        result = self._run_gh(["repo", "sync"], check=False)
        return result.returncode == 0

    def get_repo_info(self) -> dict[str, Any]:
        """Get repository information."""
        args = [
            "repo",
            "view",
            "--json",
            "name,owner,description,url,defaultBranchRef,isPrivate",
        ]

        result = self._run_gh(args)
        return json.loads(result.stdout) if result.stdout else {}

    def get_stats(self) -> dict[str, Any]:
        """Get usage statistics."""
        return {
            "authenticated": self._authenticated,
            "repo": self.repo,
        }
