"""Tests for supervisor module."""

from unittest.mock import MagicMock

import pytest

from vibe.config import Project
from vibe.orchestrator.supervisor import Supervisor


class TestInvestigationTaskDetection:
    """Tests for _is_investigation_task method."""

    @pytest.fixture
    def supervisor(self):
        """Create a Supervisor with mock GLM client."""
        mock_client = MagicMock()
        project = Project(name="test", path="/test")
        return Supervisor(
            glm_client=mock_client,
            project=project,
        )

    # Question-based user_requests
    @pytest.mark.parametrize("user_request", [
        "What does this function do?",
        "How does the authentication system work?",
        "Why is this test failing?",
        "Where is the config defined?",
        "Which files contain the API endpoints?",
        "Who wrote this code?",
        "When was this last modified?",
    ])
    def test_question_user_requests_are_investigation(self, supervisor, user_request):
        """Questions should be detected as investigation tasks."""
        assert supervisor._is_investigation_task(user_request)

    # Investigation keywords
    @pytest.mark.parametrize("user_request", [
        "Find all usages of this function",
        "Search for TODO comments",
        "Investigate why the build is slow",
        "Analyze the performance bottleneck",
        "Explain this code to me",
        "Show me how the login works",
        "List all API endpoints",
        "Check if there are any memory leaks",
    ])
    def test_investigation_keywords_detected(self, supervisor, user_request):
        """Investigation keywords should trigger detection."""
        assert supervisor._is_investigation_task(user_request)

    # Implementation/change user_requests (NOT investigation)
    @pytest.mark.parametrize("user_request", [
        "Add a new login button",
        "Fix the bug in authentication",
        "Implement dark mode",
        "Refactor the user service",
        "Update the config to use new values",
        "Create a new endpoint for users",
        "Delete unused imports",
        "Rename the function to something better",
    ])
    def test_implementation_user_requests_not_investigation(self, supervisor, user_request):
        """Implementation user_requests should NOT be investigation tasks."""
        assert not supervisor._is_investigation_task(user_request)

    # Edge cases
    def test_question_mark_at_end(self, supervisor):
        """Any user_request ending with ? should be investigation."""
        assert supervisor._is_investigation_task("Does this work?")
        assert supervisor._is_investigation_task("Is the test passing?")

    def test_case_insensitive(self, supervisor):
        """Detection should be case insensitive."""
        assert supervisor._is_investigation_task("WHAT does this do")
        assert supervisor._is_investigation_task("How Does This Work")
        assert supervisor._is_investigation_task("FIND all files")

    def test_can_you_patterns(self, supervisor):
        """'Can you find/show/explain' patterns should be investigation."""
        assert supervisor._is_investigation_task("Can you find the issue")
        assert supervisor._is_investigation_task("Can you show me the logs")
        assert supervisor._is_investigation_task("Can you explain this")

    def test_please_patterns(self, supervisor):
        """'Please find/show/explain' patterns should be investigation."""
        assert supervisor._is_investigation_task("Please find all usages")
        assert supervisor._is_investigation_task("Please explain how this works")
        assert supervisor._is_investigation_task("Please investigate the bug")

    def test_readme_docs_mentions(self, supervisor):
        """Requests about documentation should be investigation."""
        assert supervisor._is_investigation_task("Show me the readme")
        assert supervisor._is_investigation_task("What does the documentation say")

    def test_mixed_user_requests_with_implementation(self, supervisor):
        """Mixed requests with implementation intent are NOT investigation."""
        # These have implementation intent (verbs like "Add", "Fix" take priority)
        assert not supervisor._is_investigation_task("Add a feature to display user stats")
        assert not supervisor._is_investigation_task("Fix the authentication bug")

    def test_empty_and_whitespace(self, supervisor):
        """Empty and whitespace-only user_requests should not crash."""
        assert not supervisor._is_investigation_task("")
        assert not supervisor._is_investigation_task("   ")
        assert not supervisor._is_investigation_task("\n\t")
