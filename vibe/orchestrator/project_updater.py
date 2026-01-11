"""
Project Updater - Automatic STARMAP.md and CHANGELOG.md updates

Keeps project documentation in sync after Claude makes changes.
Uses GLM to generate appropriate update text.
"""

import logging
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from vibe.glm.client import GLMClient

logger = logging.getLogger(__name__)


@dataclass
class FileChange:
    """Represents a file change made by Claude."""

    path: str
    action: str  # 'created', 'modified', 'deleted'
    description: str = ""


@dataclass
class ChangelogEntry:
    """An entry to add to the changelog."""

    category: str  # 'Added', 'Changed', 'Fixed', 'Removed'
    description: str
    details: list[str] | None = None


STARMAP_UPDATE_PROMPT = """Given the following file changes and the current STARMAP.md content,
suggest updates to the STARMAP.md file.

Current STARMAP.md:
```
{starmap_content}
```

File changes:
{file_changes}

Task that was completed:
{task_description}

Instructions:
1. If new files were created, add them to the appropriate section
2. If new components/modules were added, update the structure section
3. Keep the existing format and style
4. Only output the sections that need to be updated
5. If no updates are needed, respond with "NO_UPDATES_NEEDED"

Output the updated sections in markdown format."""

CHANGELOG_ENTRY_PROMPT = """Generate a changelog entry for the following completed task.

Task: {task_description}

Changes made:
{file_changes}

Claude's summary:
{claude_summary}

Instructions:
1. Write a concise one-line description
2. Use present tense (e.g., "Add", "Fix", "Update")
3. Focus on what changed from a user perspective
4. Include relevant file/component names if helpful

Output format:
CATEGORY: [Added|Changed|Fixed|Removed]
DESCRIPTION: [One line description]
DETAILS: (optional bullet points)
- Detail 1
- Detail 2"""


class ProjectUpdater:
    """
    Updates project documentation after Claude makes changes.

    Responsibilities:
    - Parse and update STARMAP.md with new files/components
    - Add changelog entries for completed tasks
    - Maintain documentation consistency
    """

    def __init__(
        self,
        project_path: str,
        glm_client: GLMClient | None = None,
    ):
        """
        Initialize project updater.

        Args:
            project_path: Path to project root
            glm_client: Optional GLM client for intelligent updates
        """
        self.project_path = Path(project_path)
        self.glm_client = glm_client

        # Standard file locations
        self.starmap_path = self.project_path / "STARMAP.md"
        self.changelog_path = self.project_path / "CHANGELOG.md"

    def get_file_changes(self, files: list[str]) -> list[FileChange]:
        """
        Determine the type of change for each file.

        Args:
            files: List of file paths that were modified

        Returns:
            List of FileChange objects with action type
        """
        changes = []

        for file_path in files:
            full_path = self.project_path / file_path
            if not file_path.startswith("/"):
                full_path = self.project_path / file_path
            else:
                full_path = Path(file_path)

            # Check git status for this file
            action = self._get_git_status(str(full_path))
            changes.append(FileChange(path=file_path, action=action))

        return changes

    def _get_git_status(self, file_path: str) -> str:
        """Get git status for a file (created, modified, deleted)."""
        try:
            # Check if file is new (untracked or newly staged)
            result = subprocess.run(
                ["git", "status", "--porcelain", file_path],
                cwd=self.project_path,
                capture_output=True,
                text=True,
                timeout=5,
            )

            status = result.stdout.strip()
            if not status:
                return "modified"  # Already committed

            status_code = status[:2]
            if "?" in status_code or "A" in status_code:
                return "created"
            elif "D" in status_code:
                return "deleted"
            else:
                return "modified"
        except Exception:
            return "modified"  # Default assumption

    def read_starmap(self) -> str | None:
        """Read current STARMAP.md content."""
        if self.starmap_path.exists():
            try:
                return self.starmap_path.read_text()
            except Exception as e:
                logger.warning(f"Could not read STARMAP.md: {e}")
        return None

    def read_changelog(self) -> str | None:
        """Read current CHANGELOG.md content."""
        if self.changelog_path.exists():
            try:
                return self.changelog_path.read_text()
            except Exception as e:
                logger.warning(f"Could not read CHANGELOG.md: {e}")
        return None

    async def update_starmap(
        self,
        file_changes: list[FileChange],
        task_description: str,
    ) -> bool:
        """
        Update STARMAP.md with new files/components.

        Args:
            file_changes: List of file changes
            task_description: Description of the completed task

        Returns:
            True if STARMAP was updated
        """
        current_content = self.read_starmap()
        if not current_content:
            logger.info("No STARMAP.md found, skipping update")
            return False

        # Only update if new files were created
        new_files = [c for c in file_changes if c.action == "created"]
        if not new_files:
            return False

        if self.glm_client:
            # Use GLM to generate intelligent update
            try:
                changes_text = "\n".join(
                    f"- {c.path} ({c.action})" for c in file_changes
                )

                prompt = STARMAP_UPDATE_PROMPT.format(
                    starmap_content=current_content[:2000],  # Limit context
                    file_changes=changes_text,
                    task_description=task_description,
                )

                response = await self.glm_client.chat(
                    system_prompt="You are a documentation updater. Keep responses concise.",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                )

                if "NO_UPDATES_NEEDED" in response.content:
                    return False

                # For now, just log the suggested update
                # Full integration would parse and apply the changes
                logger.info(f"STARMAP update suggested:\n{response.content[:200]}")
                return True

            except Exception as e:
                logger.warning(f"GLM STARMAP update failed: {e}")

        # Fallback: Simple file list append
        return self._append_to_starmap(new_files)

    def _append_to_starmap(self, new_files: list[FileChange]) -> bool:
        """Simple fallback: append new files to STARMAP."""
        if not new_files:
            return False

        current_content = self.read_starmap()
        if not current_content:
            return False

        # Add a "Recent Changes" section if not exists
        addition = "\n\n## Recent Files Added\n\n"
        addition += f"_Updated: {datetime.now().strftime('%Y-%m-%d')}_\n\n"
        for f in new_files:
            addition += f"- `{f.path}`\n"

        # Check if section already exists
        if "## Recent Files Added" in current_content:
            # Update existing section
            pattern = r"## Recent Files Added\n\n.*?(?=\n## |\Z)"
            new_section = "## Recent Files Added\n\n"
            new_section += f"_Updated: {datetime.now().strftime('%Y-%m-%d')}_\n\n"
            for f in new_files:
                new_section += f"- `{f.path}`\n"
            updated = re.sub(pattern, new_section, current_content, flags=re.DOTALL)
        else:
            updated = current_content + addition

        try:
            self.starmap_path.write_text(updated)
            logger.info(f"Updated STARMAP.md with {len(new_files)} new files")
            return True
        except Exception as e:
            logger.error(f"Failed to write STARMAP.md: {e}")
            return False

    async def add_changelog_entry(
        self,
        task_description: str,
        file_changes: list[FileChange],
        claude_summary: str,
        category: str | None = None,
    ) -> bool:
        """
        Add an entry to CHANGELOG.md.

        Args:
            task_description: The task that was completed
            file_changes: Files that were changed
            claude_summary: Claude's summary of what was done
            category: Override category (Added, Changed, Fixed, Removed)

        Returns:
            True if changelog was updated
        """
        current_content = self.read_changelog()

        # Generate entry
        entry = await self._generate_changelog_entry(
            task_description=task_description,
            file_changes=file_changes,
            claude_summary=claude_summary,
            category=category,
        )

        if not entry:
            return False

        # Insert entry into changelog
        return self._insert_changelog_entry(current_content, entry)

    async def _generate_changelog_entry(
        self,
        task_description: str,
        file_changes: list[FileChange],
        claude_summary: str,
        category: str | None = None,
    ) -> ChangelogEntry | None:
        """Generate a changelog entry using GLM or fallback."""

        if self.glm_client and not category:
            try:
                changes_text = "\n".join(
                    f"- {c.path} ({c.action})" for c in file_changes
                )

                prompt = CHANGELOG_ENTRY_PROMPT.format(
                    task_description=task_description,
                    file_changes=changes_text,
                    claude_summary=claude_summary,
                )

                response = await self.glm_client.chat(
                    system_prompt="You are a technical writer. Be concise.",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                )

                # Parse response
                content = response.content
                cat_match = re.search(r"CATEGORY:\s*(\w+)", content)
                desc_match = re.search(r"DESCRIPTION:\s*(.+?)(?:\n|$)", content)

                if cat_match and desc_match:
                    entry_category = cat_match.group(1).strip()
                    description = desc_match.group(1).strip()

                    # Extract optional details
                    details = []
                    detail_section = re.search(r"DETAILS:(.+)", content, re.DOTALL)
                    if detail_section:
                        for line in detail_section.group(1).split("\n"):
                            line = line.strip()
                            if line.startswith("- "):
                                details.append(line[2:])

                    return ChangelogEntry(
                        category=entry_category,
                        description=description,
                        details=details if details else None,
                    )

            except Exception as e:
                logger.warning(f"GLM changelog generation failed: {e}")

        # Fallback: Generate simple entry
        if category is None:
            # Infer category from file actions
            actions = {c.action for c in file_changes}
            if "created" in actions:
                category = "Added"
            elif "deleted" in actions:
                category = "Removed"
            else:
                category = "Changed"

        # Simple description from task
        description = task_description[:100]
        if len(task_description) > 100:
            description = description.rsplit(" ", 1)[0] + "..."

        return ChangelogEntry(category=category, description=description)

    def _insert_changelog_entry(
        self,
        current_content: str | None,
        entry: ChangelogEntry,
    ) -> bool:
        """Insert a changelog entry into the file."""

        today = datetime.now().strftime("%Y-%m-%d")

        if not current_content:
            # Create new changelog
            content = f"""# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### {entry.category}

- {entry.description}
"""
            if entry.details:
                for detail in entry.details:
                    content += f"  - {detail}\n"

        else:
            # Find [Unreleased] section and appropriate category
            unreleased_pattern = r"## \[Unreleased\]"
            unreleased_match = re.search(unreleased_pattern, current_content)

            if not unreleased_match:
                # No unreleased section, add one
                # Find first ## header
                first_header = re.search(r"\n## ", current_content)
                if first_header:
                    insert_pos = first_header.start()
                    new_section = f"\n## [Unreleased]\n\n### {entry.category}\n\n- {entry.description}\n"
                    if entry.details:
                        for detail in entry.details:
                            new_section += f"  - {detail}\n"
                    content = (
                        current_content[:insert_pos]
                        + new_section
                        + current_content[insert_pos:]
                    )
                else:
                    # Just append
                    content = current_content + f"\n\n## [Unreleased]\n\n### {entry.category}\n\n- {entry.description}\n"
            else:
                # Find or create category section under Unreleased
                category_pattern = rf"### {entry.category}\n"
                unreleased_end = unreleased_match.end()

                # Find next ## header to limit search
                next_version = re.search(r"\n## \[[\d\.]", current_content[unreleased_end:])
                section_end = unreleased_end + next_version.start() if next_version else len(current_content)

                category_match = re.search(
                    category_pattern,
                    current_content[unreleased_end:section_end]
                )

                if category_match:
                    # Add to existing category
                    insert_pos = unreleased_end + category_match.end()
                    new_entry = f"\n- {entry.description}"
                    if entry.details:
                        for detail in entry.details:
                            new_entry += f"\n  - {detail}"
                    content = (
                        current_content[:insert_pos]
                        + new_entry
                        + current_content[insert_pos:]
                    )
                else:
                    # Create new category section
                    # Insert after [Unreleased] header
                    insert_pos = unreleased_end
                    new_section = f"\n\n### {entry.category}\n\n- {entry.description}"
                    if entry.details:
                        for detail in entry.details:
                            new_section += f"\n  - {detail}"
                    content = (
                        current_content[:insert_pos]
                        + new_section
                        + current_content[insert_pos:]
                    )

        try:
            self.changelog_path.write_text(content)
            logger.info(f"Added changelog entry: [{entry.category}] {entry.description[:50]}")
            return True
        except Exception as e:
            logger.error(f"Failed to write CHANGELOG.md: {e}")
            return False

    async def update_after_task(
        self,
        task_description: str,
        files_changed: list[str],
        claude_summary: str,
        update_starmap: bool = True,
        update_changelog: bool = True,
    ) -> dict[str, bool]:
        """
        Update project documentation after a task completes.

        Args:
            task_description: The completed task
            files_changed: List of changed file paths
            claude_summary: Claude's summary
            update_starmap: Whether to update STARMAP.md
            update_changelog: Whether to update CHANGELOG.md

        Returns:
            Dict with 'starmap' and 'changelog' update status
        """
        results = {"starmap": False, "changelog": False}

        # Get file change details
        file_changes = self.get_file_changes(files_changed)

        if update_starmap:
            results["starmap"] = await self.update_starmap(
                file_changes=file_changes,
                task_description=task_description,
            )

        if update_changelog:
            results["changelog"] = await self.add_changelog_entry(
                task_description=task_description,
                file_changes=file_changes,
                claude_summary=claude_summary,
            )

        return results
