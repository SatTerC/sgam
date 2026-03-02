import numpy as np
import pytest
from sgam.utils import (
    rescale_to_unit_interval,
    compute_relative_changes,
    solve_recurrence,
    find_segments,
)


class TestRescaleToUnitInterval:
    def test_normalizes_to_unit_interval(self):
        x = np.array([0.0, 50.0, 100.0])
        result = rescale_to_unit_interval(x)
        assert result[0] == 0.0
        assert result[2] == 1.0

    def test_identical_values_returns_zeros(self):
        x = np.array([5.0, 5.0, 5.0])
        result = rescale_to_unit_interval(x)
        np.testing.assert_array_equal(result, np.zeros(3))

    def test_handles_negative_values(self):
        x = np.array([-10.0, 0.0, 10.0])
        result = rescale_to_unit_interval(x)
        assert result[0] == 0.0
        assert result[2] == 1.0


class TestComputeRelativeChanges:
    def test_first_element_is_zero(self):
        x = np.array([1.0, 2.0, 3.0])
        result = compute_relative_changes(x)
        assert result[0] == 0.0

    def test_computes_correct_relative_change(self):
        x = np.array([10.0, 20.0])
        result = compute_relative_changes(x)
        assert result[1] == pytest.approx(1.0)  # (20-10)/10

    def test_handles_zero_previous_value(self):
        x = np.array([0.0, 5.0, 10.0])
        result = compute_relative_changes(x)
        assert result[1] == pytest.approx(5.0 / 1e-6)  # large due to near-zero divisor


class TestSolveRecurrence:
    def test_simple_decay(self):
        a = np.array([0.5, 0.5, 0.5])
        b = np.array([0.0, 0.0, 0.0])
        initial = 8.0
        result = solve_recurrence(initial, a, b)
        assert result[0] == pytest.approx(4.0)  # 0.5 * 8
        assert result[1] == pytest.approx(2.0)  # 0.5 * 4
        assert result[2] == pytest.approx(1.0)  # 0.5 * 2

    def test_constant_addition(self):
        a = np.array([1.0, 1.0, 1.0])
        b = np.array([1.0, 1.0, 1.0])
        initial = 0.0
        result = solve_recurrence(initial, a, b)
        assert result[0] == 1.0
        assert result[1] == 2.0
        assert result[2] == 3.0


class TestFindSegments:
    def test_no_true_values_returns_full_segment(self):
        mask = np.array([False, False, False])
        result = find_segments(mask)
        assert result == [(0, 3)]

    def test_all_true_returns_segments(self):
        mask = np.array([True, True, True])
        result = find_segments(mask)
        assert result == [(1, 2), (2, 3)]

    def test_single_true_in_middle(self):
        mask = np.array([False, True, False, False])
        result = find_segments(mask)
        assert result == [(0, 2), (2, 4)]

    def test_multiple_segments(self):
        mask = np.array([False, False, True, False, False, True, False])
        result = find_segments(mask)
        assert result == [(0, 3), (3, 6), (6, 7)]
