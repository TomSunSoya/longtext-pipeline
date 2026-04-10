"""Tests for the SummaryGrouper class."""

import pytest

from src.longtext_pipeline.grouper import SummaryGrouper
from src.longtext_pipeline.models import Summary


class TestSummaryGrouper:
    """Test cases for SummaryGrouper."""

    def test_normal_grouping_correct_number_of_groups(self):
        """Test that summaries are grouped into correct number of groups."""
        grouper = SummaryGrouper(group_size=3)
        summaries = [Summary(part_index=i, content=f"summary {i}") for i in range(9)]

        groups = grouper.group_summaries(summaries)

        # 9 summaries with group_size=3 should create 3 groups
        assert len(groups) == 3
        # Each group should have exactly 3 summaries
        for group in groups:
            assert len(group) == 3

    def test_group_size_1_each_summary_alone(self):
        """Test that group_size=1 puts each summary in its own group."""
        grouper = SummaryGrouper(group_size=1)
        summaries = [Summary(part_index=i, content=f"summary {i}") for i in range(5)]

        groups = grouper.group_summaries(summaries)

        # Should create 5 groups
        assert len(groups) == 5
        # Each group should have exactly 1 summary
        for group in groups:
            assert len(group) == 1

    def test_fewer_summaries_than_group_size_single_group(self):
        """Test that fewer summaries than group_size creates single group."""
        grouper = SummaryGrouper(group_size=10)
        summaries = [Summary(part_index=i, content=f"summary {i}") for i in range(3)]

        groups = grouper.group_summaries(summaries)

        # Should create exactly 1 group
        assert len(groups) == 1
        # Group should contain all 3 summaries
        assert len(groups[0]) == 3
        # Verify all summaries are in the group
        for i, summary in enumerate(groups[0]):
            assert summary.part_index == i

    def test_uneven_grouping_last_group_smaller(self):
        """Test that uneven grouping results in smaller last group."""
        grouper = SummaryGrouper(group_size=3)
        summaries = [Summary(part_index=i, content=f"summary {i}") for i in range(7)]

        groups = grouper.group_summaries(summaries)

        # 7 summaries with group_size=3 creates 3 groups: 3, 3, 1
        assert len(groups) == 3
        # First two groups should have 3 summaries each
        assert len(groups[0]) == 3
        assert len(groups[1]) == 3
        # Last group should have remaining 1 summary
        assert len(groups[2]) == 1

    def test_empty_summaries_list_empty_groups(self):
        """Test that empty summaries list returns empty groups."""
        grouper = SummaryGrouper(group_size=5)
        summaries = []

        groups = grouper.group_summaries(summaries)

        # Should return empty list (not list with one empty group)
        assert groups == []

    def test_group_size_configurable(self):
        """Test that group_size is configurable via constructor."""
        grouper1 = SummaryGrouper(group_size=2)
        grouper2 = SummaryGrouper(group_size=10)
        grouper3 = SummaryGrouper(group_size=1)

        summaries = [Summary(part_index=i, content=f"summary {i}") for i in range(10)]

        groups1 = grouper1.group_summaries(summaries)
        groups2 = grouper2.group_summaries(summaries)
        groups3 = grouper3.group_summaries(summaries)

        # group_size=2 should create 5 groups
        assert len(groups1) == 5
        # group_size=10 should create 1 group
        assert len(groups2) == 1
        # group_size=1 should create 10 groups
        assert len(groups3) == 10

    def test_group_size_validation_less_than_one_raises_error(self):
        """Test that group_size < 1 raises ValueError."""
        with pytest.raises(ValueError):
            SummaryGrouper(group_size=0)

        with pytest.raises(ValueError):
            SummaryGrouper(group_size=-1)

    def test_summary_order_preserved_in_groups(self):
        """Test that summary order is preserved within groups."""
        grouper = SummaryGrouper(group_size=3)
        summaries = [Summary(part_index=i, content=f"summary {i}") for i in range(10)]

        groups = grouper.group_summaries(summaries)

        # Flatten groups and verify order
        flattened = [summary for group in groups for summary in group]

        assert len(flattened) == len(summaries)
        for i, summary in enumerate(flattened):
            assert summary.part_index == i

    def test_group_boundary_alignment(self):
        """Test that groups align with group_size boundaries."""
        grouper = SummaryGrouper(group_size=4)
        summaries = [Summary(part_index=i, content=f"summary {i}") for i in range(10)]

        groups = grouper.group_summaries(summaries)

        # Verify first group has indices 0,1,2,3
        assert [s.part_index for s in groups[0]] == [0, 1, 2, 3]
        # Verify second group has indices 4,5,6,7
        assert [s.part_index for s in groups[1]] == [4, 5, 6, 7]
        # Verify last group has indices 8,9
        assert [s.part_index for s in groups[2]] == [8, 9]
