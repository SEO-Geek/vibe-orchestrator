"""
Vibe Memory Compaction - Compress old context items to reduce token usage.

Summarizes old context entries into compact summaries using GLM.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vibe.glm.client import GLMClient
    from vibe.memory.keeper import VibeMemory

logger = logging.getLogger(__name__)


def _parse_timestamp(value: Any) -> datetime | None:
    """
    Parse a timestamp from various formats.

    Args:
        value: Timestamp value (datetime, str, or None)

    Returns:
        datetime object or None if parsing fails
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


# Default thresholds for compaction decisions
# These are conservative defaults to avoid losing important recent context
DEFAULT_MAX_ITEMS = 50  # Only consider compaction if total exceeds this
DEFAULT_AGE_HOURS = 24  # Only compact items older than this (preserve recent work)
DEFAULT_ITEMS_PER_SUMMARY = 20  # Minimum items to justify a compaction pass


async def compact_context(
    glm_client: GLMClient,
    memory: VibeMemory,
    max_items: int = DEFAULT_MAX_ITEMS,
    age_hours: int = DEFAULT_AGE_HOURS,
) -> dict[str, Any]:
    """
    Compress old context items into summaries.

    This helps reduce token usage when loading project context by:
    1. Finding items older than age_hours
    2. Grouping by category
    3. Summarizing each group with GLM
    4. Saving summaries and deleting originals

    Args:
        glm_client: GLMClient for generating summaries
        memory: VibeMemory instance to compact
        max_items: Only compact if item count exceeds this
        age_hours: Compact items older than this many hours

    Returns:
        Dict with compaction statistics:
        - compacted: Number of items compressed
        - summaries: Number of summaries created
        - reason: Why compaction did/didn't happen
    """
    result = {"compacted": 0, "summaries": 0, "reason": ""}

    try:
        # Load all context items (note: limit=500 means only first 500 are considered)
        all_items = memory.load_project_context(limit=500)
        total_count = len(all_items)

        if total_count < max_items:
            result["reason"] = f"Below threshold ({total_count}/{max_items} items)"
            logger.info(f"Compaction skipped: {result['reason']}")
            return result

        # Find old items using helper function
        cutoff = datetime.now() - timedelta(hours=age_hours)
        old_items = []

        for item in all_items:
            created_at = _parse_timestamp(getattr(item, "created_at", None))
            if created_at is None:
                continue
            # Make cutoff timezone-aware if timestamp is
            if created_at.tzinfo is not None:
                cutoff_aware = cutoff.replace(tzinfo=UTC)
                if created_at < cutoff_aware:
                    old_items.append(item)
            elif created_at < cutoff:
                old_items.append(item)

        if len(old_items) < DEFAULT_ITEMS_PER_SUMMARY:
            result["reason"] = (
                f"Not enough old items ({len(old_items)} < {DEFAULT_ITEMS_PER_SUMMARY})"
            )
            logger.info(f"Compaction skipped: {result['reason']}")
            return result

        # Group by category
        by_category: dict[str, list] = {}
        for item in old_items:
            cat = getattr(item, "category", "note") or "note"
            by_category.setdefault(cat, []).append(item)

        # Summarize each category with enough items
        summaries_created = 0
        items_compacted = 0

        for category, items in by_category.items():
            if len(items) < 5:  # Skip small categories
                continue

            # Build text for summarization
            items_text = "\n".join(
                [
                    f"- {getattr(item, 'key', 'unknown')}: {str(getattr(item, 'value', ''))[:200]}"
                    for item in items[:50]  # Limit items per summary
                ]
            )

            # Generate summary
            try:
                summary = await summarize_context_items(glm_client, items_text, category)
            except Exception as e:
                logger.warning(f"Failed to summarize {category} items: {e}")
                continue

            # SAFETY-FIRST: Collect keys but don't delete until summary is saved
            # This prevents data loss if the save fails
            items_to_delete = []
            for item in items:
                item_key = getattr(item, "key", "")
                if item_key:
                    items_to_delete.append(item_key)

            # Save summary FIRST - only proceed to deletion if this succeeds
            # Key format: compacted-{category}-{date} for easy identification
            summary_key = f"compacted-{category}-{cutoff.strftime('%Y%m%d')}"
            memory.save(
                key=summary_key,
                value=summary,
                category="note",
                priority="low",  # Compacted items are less important than fresh context
            )
            summaries_created += 1

            # NOW safe to delete originals - we have the summary as backup
            for item_key in items_to_delete:
                try:
                    memory.delete(item_key)
                    items_compacted += 1
                except Exception as e:
                    logger.warning(f"Failed to delete {item_key}: {e}")

        result["compacted"] = items_compacted
        result["summaries"] = summaries_created
        result["reason"] = f"Compacted {items_compacted} items into {summaries_created} summaries"
        logger.info(f"Compaction complete: {result['reason']}")

        return result

    except Exception as e:
        result["reason"] = f"Error: {e}"
        logger.error(f"Compaction failed: {e}")
        return result


async def summarize_context_items(
    glm_client: GLMClient,
    items_text: str,
    category: str,
) -> str:
    """
    Summarize a list of context items into a concise paragraph.

    Args:
        glm_client: GLMClient instance
        items_text: Formatted text of items to summarize
        category: Category of items being summarized

    Returns:
        Concise summary string
    """
    prompt = f"""Summarize these {category} items into a concise paragraph (2-3 sentences).
Focus on the key information useful for understanding project state.

Items:
{items_text}

Provide ONLY the summary paragraph, no introduction or explanation."""

    response = await glm_client.chat(
        system_prompt="You are a context summarizer. Output only the summary.",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
        method="summarize_context",
    )

    return response.content.strip()


def get_compaction_stats(memory: VibeMemory) -> dict[str, Any]:
    """
    Get statistics about what would be compacted.

    Args:
        memory: VibeMemory instance

    Returns:
        Dict with item counts by age and category
    """
    try:
        all_items = memory.load_project_context(limit=500)
        now = datetime.now()

        stats: dict[str, Any] = {
            "total_items": len(all_items),
            "by_age": {
                "last_hour": 0,
                "last_day": 0,
                "older": 0,
            },
            "by_category": {},
        }

        hour_ago = now - timedelta(hours=1)
        day_ago = now - timedelta(hours=24)

        for item in all_items:
            # Count by category
            cat = getattr(item, "category", "unknown")
            stats["by_category"][cat] = stats["by_category"].get(cat, 0) + 1

            # Count by age using helper function
            created_at = _parse_timestamp(getattr(item, "created_at", None))
            if created_at:
                # Handle timezone-aware timestamps
                if created_at.tzinfo is not None:
                    hour_ago_aware = hour_ago.replace(tzinfo=UTC)
                    day_ago_aware = day_ago.replace(tzinfo=UTC)
                    if created_at > hour_ago_aware:
                        stats["by_age"]["last_hour"] += 1
                    elif created_at > day_ago_aware:
                        stats["by_age"]["last_day"] += 1
                    else:
                        stats["by_age"]["older"] += 1
                else:
                    if created_at > hour_ago:
                        stats["by_age"]["last_hour"] += 1
                    elif created_at > day_ago:
                        stats["by_age"]["last_day"] += 1
                    else:
                        stats["by_age"]["older"] += 1

        return stats

    except Exception as e:
        return {"error": str(e)}
