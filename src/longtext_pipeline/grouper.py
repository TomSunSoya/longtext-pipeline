"""Summary grouping logic for the longtext pipeline.

This module provides functionality to group summaries into stages
for hierarchical analysis.
"""

from typing import List

from .models import Summary


class SummaryGrouper:
    """Groups summaries into stages for analysis.

    Attributes:
        group_size: Number of summaries to combine in each group.
    """

    def __init__(self, group_size: int = 5):
        """Initialize the grouper with a configurable group size.

        Args:
            group_size: Number of summaries to group together. Must be >= 1.
                       Default is 5 as specified in CONFIG.md.

        Raises:
            ValueError: If group_size is less than 1.
        """
        if group_size < 1:
            raise ValueError("group_size must be at least 1")

        self.group_size = group_size

    def group_summaries(self, summaries: List[Summary]) -> List[List[Summary]]:
        """Group summaries into stages based on configurable group_size.

        Args:
            summaries: List of Summary objects to group.
            group_size: Optional override for group size. If None, uses self.group_size.

        Returns:
            List of groups, where each group is a list of Summary objects.
            The last group may be smaller if summaries don't divide evenly.
            If summaries is empty, returns an empty list.

        Examples:
            >>> grouper = SummaryGrouper(group_size=3)
            >>> summaries = [Summary(part_index=i, content=f"summary {i}") for i in range(7)]
            >>> groups = grouper.group_summaries(summaries)
            >>> len(groups)
            3
            >>> len(groups[0])
            3
            >>> len(groups[1])
            3
            >>> len(groups[2])
            1
        """
        if not summaries:
            return []

        groups = []
        for i in range(0, len(summaries), self.group_size):
            group = summaries[i : i + self.group_size]
            groups.append(group)

        return groups
