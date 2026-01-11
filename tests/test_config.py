"""Tests for config module."""

import json
import tempfile
from pathlib import Path

import pytest

from vibe.config import Project, VibeConfig, load_config
from vibe.exceptions import ProjectNotFoundError


class TestProject:
    """Tests for Project dataclass."""

    def test_project_creation(self):
        """Test creating a project."""
        project = Project(
            name="test",
            path="/home/brian/test",
            description="Test project",
        )
        assert project.name == "test"
        assert project.path == "/home/brian/test"
        assert project.starmap == "STARMAP.md"
        assert project.claude_md == "CLAUDE.md"

    def test_project_path_expansion(self):
        """Test that ~ is expanded in path."""
        project = Project(name="test", path="~/projects/test")
        assert not project.path.startswith("~")
        assert "projects/test" in project.path

    def test_project_to_dict(self):
        """Test serialization to dict."""
        project = Project(name="test", path="/home/test")
        data = project.to_dict()
        assert data["name"] == "test"
        assert data["path"] == "/home/test"

    def test_project_from_dict(self):
        """Test deserialization from dict."""
        data = {
            "name": "test",
            "path": "/home/test",
            "description": "A test",
        }
        project = Project.from_dict(data)
        assert project.name == "test"
        assert project.description == "A test"


class TestVibeConfig:
    """Tests for VibeConfig."""

    def test_get_project_by_index(self):
        """Test getting project by 1-based index."""
        config = VibeConfig(
            projects=[
                Project(name="first", path="/first"),
                Project(name="second", path="/second"),
            ]
        )

        assert config.get_project(1).name == "first"
        assert config.get_project(2).name == "second"

    def test_get_project_by_name(self):
        """Test getting project by name (case-insensitive)."""
        config = VibeConfig(
            projects=[
                Project(name="MyProject", path="/project"),
            ]
        )

        assert config.get_project("myproject").name == "MyProject"
        assert config.get_project("MYPROJECT").name == "MyProject"

    def test_get_project_not_found(self):
        """Test error when project not found."""
        config = VibeConfig(projects=[])

        with pytest.raises(ProjectNotFoundError):
            config.get_project("nonexistent")

        with pytest.raises(ProjectNotFoundError):
            config.get_project(1)

    def test_add_project(self):
        """Test adding a project."""
        config = VibeConfig()
        config.add_project(Project(name="new", path="/new"))

        assert len(config.projects) == 1
        assert config.projects[0].name == "new"

    def test_remove_project(self):
        """Test removing a project."""
        config = VibeConfig(
            projects=[Project(name="toremove", path="/remove")]
        )

        removed = config.remove_project("toremove")
        assert removed.name == "toremove"
        assert len(config.projects) == 0
