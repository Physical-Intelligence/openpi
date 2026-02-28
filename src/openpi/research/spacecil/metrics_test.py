"""Tests for metrics module."""

import numpy as np
import pytest

from openpi.research.spacecil.metrics import average_success
from openpi.research.spacecil.metrics import backward_transfer
from openpi.research.spacecil.metrics import forgetting
from openpi.research.spacecil.metrics import operational_forgetting
from openpi.research.spacecil.metrics import routing_accuracy
from openpi.research.spacecil.metrics import routing_entropy

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def perfect_lower_triangular():
    """Perfect matrix: R[i][j]=1.0 for i>=j, 0 for i<j.

    3x3 example:
        [[1, 0, 0],
         [1, 1, 0],
         [1, 1, 1]]
    After training task i, tasks 0..i are perfectly solved.
    No forgetting occurs, no backward transfer.
    """
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
        ]
    )


@pytest.fixture
def zero_final_row():
    """Matrix where final row is all zeros — maximum catastrophic forgetting.

    3x3:
        [[0.8, 0.0, 0.0],
         [0.6, 0.9, 0.0],
         [0.0, 0.0, 0.7]]
    """
    return np.array(
        [
            [0.8, 0.0, 0.0],
            [0.6, 0.9, 0.0],
            [0.0, 0.0, 0.7],
        ]
    )


@pytest.fixture
def known_2x2():
    """2x2 matrix for hand-verified calculations.

    [[0.9, 0.0],
     [0.5, 0.8]]

    Task 0: trained first, achieves 0.9 on itself.
    Task 1: trained second, final row is [0.5, 0.8].
    """
    return np.array(
        [
            [0.9, 0.0],
            [0.5, 0.8],
        ]
    )


@pytest.fixture
def known_3x3():
    """3x3 matrix for verifying all matrix-based metrics.

    [[0.9, 0.0, 0.0],
     [0.7, 0.85, 0.0],
     [0.4, 0.6, 0.95]]

    Peaks:  task0=0.9 (row0), task1=0.85 (row1)
    Final:  task0=0.4, task1=0.6
    """
    return np.array(
        [
            [0.9, 0.0, 0.0],
            [0.7, 0.85, 0.0],
            [0.4, 0.6, 0.95],
        ]
    )


# ---------------------------------------------------------------------------
# average_success
# ---------------------------------------------------------------------------


class TestAverageSuccess:
    def test_perfect_lower_triangular(self, perfect_lower_triangular):
        # Final row = [1, 1, 1], mean = 1.0
        assert average_success(perfect_lower_triangular) == pytest.approx(1.0)

    def test_zero_final_row(self, zero_final_row):
        # Final row = [0, 0, 0.7], mean = 0.7/3
        assert average_success(zero_final_row) == pytest.approx(0.7 / 3)

    def test_known_2x2(self, known_2x2):
        # Final row = [0.5, 0.8], mean = 0.65
        assert average_success(known_2x2) == pytest.approx(0.65)

    def test_known_3x3(self, known_3x3):
        # Final row = [0.4, 0.6, 0.95], mean = 1.95/3 = 0.65
        assert average_success(known_3x3) == pytest.approx(1.95 / 3)

    def test_single_task(self):
        r = np.array([[0.75]])
        assert average_success(r) == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# backward_transfer
# ---------------------------------------------------------------------------


class TestBackwardTransfer:
    def test_perfect_lower_triangular(self, perfect_lower_triangular):
        # Tasks 0,1: R[0][0]-R[2][0]=1-1=0, R[1][1]-R[2][1]=1-1=0 → mean=0
        assert backward_transfer(perfect_lower_triangular) == pytest.approx(0.0)

    def test_zero_final_row(self, zero_final_row):
        # Task 0: R[0][0]-R[2][0]=0.8-0.0=0.8
        # Task 1: R[1][1]-R[2][1]=0.9-0.0=0.9
        # mean = 0.85
        assert backward_transfer(zero_final_row) == pytest.approx(0.85)

    def test_known_2x2(self, known_2x2):
        # Task 0: R[0][0]-R[1][0] = 0.9-0.5 = 0.4
        assert backward_transfer(known_2x2) == pytest.approx(0.4)

    def test_known_3x3(self, known_3x3):
        # Task 0: R[0][0]-R[2][0] = 0.9-0.4 = 0.5
        # Task 1: R[1][1]-R[2][1] = 0.85-0.6 = 0.25
        # mean = 0.375
        assert backward_transfer(known_3x3) == pytest.approx(0.375)

    def test_single_task(self):
        r = np.array([[0.9]])
        assert backward_transfer(r) == pytest.approx(0.0)

    def test_negative_transfer(self):
        """When final perf exceeds just-trained perf → negative backward_transfer."""
        r = np.array(
            [
                [0.5, 0.0],
                [0.8, 0.9],
            ]
        )
        # Task 0: R[0][0]-R[1][0] = 0.5-0.8 = -0.3 (improvement!)
        assert backward_transfer(r) == pytest.approx(-0.3)


# ---------------------------------------------------------------------------
# forgetting
# ---------------------------------------------------------------------------


class TestForgetting:
    def test_perfect_lower_triangular(self, perfect_lower_triangular):
        # Peaks: task0=1, task1=1. Final: task0=1, task1=1. Drops: 0,0 → mean=0
        assert forgetting(perfect_lower_triangular) == pytest.approx(0.0)

    def test_zero_final_row(self, zero_final_row):
        # Peaks: task0=max(0.8,0.6,0.0)=0.8, task1=max(0.0,0.9,0.0)=0.9
        # Final: task0=0.0, task1=0.0
        # Drops: 0.8, 0.9 → mean=0.85
        assert forgetting(zero_final_row) == pytest.approx(0.85)

    def test_known_2x2(self, known_2x2):
        # Task 0: peak=max(0.9,0.5)=0.9, final=0.5, drop=0.4
        assert forgetting(known_2x2) == pytest.approx(0.4)

    def test_known_3x3(self, known_3x3):
        # Task 0: peak=max(0.9,0.7,0.4)=0.9, final=0.4, drop=0.5
        # Task 1: peak=max(0.0,0.85,0.6)=0.85, final=0.6, drop=0.25
        # mean = 0.375
        assert forgetting(known_3x3) == pytest.approx(0.375)

    def test_single_task(self):
        r = np.array([[0.9]])
        assert forgetting(r) == pytest.approx(0.0)

    def test_always_nonnegative(self):
        """Forgetting is always >= 0 since peak >= final."""
        r = np.array(
            [
                [0.5, 0.0],
                [0.8, 0.9],
            ]
        )
        # Task 0: peak=max(0.5,0.8)=0.8, final=0.8, drop=0.0
        assert forgetting(r) == pytest.approx(0.0)
        assert forgetting(r) >= 0.0


# ---------------------------------------------------------------------------
# operational_forgetting
# ---------------------------------------------------------------------------


class TestOperationalForgetting:
    def test_equal_weights_equals_forgetting(self, known_3x3):
        """With equal weights, operational_forgetting == forgetting."""
        w = np.ones(3)
        assert operational_forgetting(known_3x3, w) == pytest.approx(forgetting(known_3x3))

    def test_zero_weights_returns_zero(self, known_3x3):
        w = np.zeros(3)
        assert operational_forgetting(known_3x3, w) == pytest.approx(0.0)

    def test_known_2x2(self, known_2x2):
        # Task 0: peak=0.9, final=0.5, drop=0.4
        # With w=[3.0, 1.0], only w[0]=3.0 is used (T-1=1 task).
        # result = 3.0*0.4 / 3.0 = 0.4
        w = np.array([3.0, 1.0])
        assert operational_forgetting(known_2x2, w) == pytest.approx(0.4)

    def test_known_3x3_unequal_weights(self, known_3x3):
        # Drops: task0=0.5, task1=0.25
        # Weights for tasks 0,1: w=[2.0, 1.0]
        # result = (2.0*0.5 + 1.0*0.25) / (2.0+1.0) = 1.25/3.0
        w = np.array([2.0, 1.0, 0.5])
        assert operational_forgetting(known_3x3, w) == pytest.approx(1.25 / 3.0)

    def test_single_task(self):
        r = np.array([[0.9]])
        w = np.array([1.0])
        assert operational_forgetting(r, w) == pytest.approx(0.0)

    def test_single_zero_weight_task(self):
        """When only one task's weight is zero, it is excluded from weighting."""
        # 3x3 matrix, drops: task0=0.5, task1=0.25
        r = np.array(
            [
                [0.9, 0.0, 0.0],
                [0.7, 0.85, 0.0],
                [0.4, 0.6, 0.95],
            ]
        )
        w = np.array([0.0, 2.0, 1.0])
        # Only task1 contributes: 2.0*0.25 / 2.0 = 0.25
        assert operational_forgetting(r, w) == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# routing_entropy
# ---------------------------------------------------------------------------


class TestRoutingEntropy:
    def test_uniform_distribution(self):
        """Entropy of uniform distribution = log(K)."""
        k = 5
        probs = np.ones(k) / k
        expected = np.log(k)
        result = routing_entropy(probs)
        assert float(result) == pytest.approx(expected, abs=1e-8)

    def test_one_hot(self):
        """Entropy of one-hot distribution = 0."""
        probs = np.array([0.0, 0.0, 1.0, 0.0])
        result = routing_entropy(probs)
        assert float(result) == pytest.approx(0.0, abs=1e-8)

    def test_batched(self):
        """Batched input: shape (B, K) → shape (B,)."""
        probs = np.array(
            [
                [0.5, 0.5],
                [1.0, 0.0],
                [0.25, 0.75],
            ]
        )
        result = routing_entropy(probs)
        assert result.shape == (3,)
        # Row 0: uniform over 2 → log(2)
        assert float(result[0]) == pytest.approx(np.log(2), abs=1e-8)
        # Row 1: one-hot → 0
        assert float(result[1]) == pytest.approx(0.0, abs=1e-8)
        # Row 2: -0.25*log(0.25) - 0.75*log(0.75)
        expected_2 = -(0.25 * np.log(0.25) + 0.75 * np.log(0.75))
        assert float(result[2]) == pytest.approx(expected_2, abs=1e-8)

    def test_binary_distribution(self):
        """Known binary entropy: H(0.3, 0.7)."""
        probs = np.array([0.3, 0.7])
        expected = -(0.3 * np.log(0.3) + 0.7 * np.log(0.7))
        assert float(routing_entropy(probs)) == pytest.approx(expected, abs=1e-8)

    def test_single_class(self):
        """Single-class distribution: entropy = 0."""
        probs = np.array([1.0])
        assert float(routing_entropy(probs)) == pytest.approx(0.0, abs=1e-8)


# ---------------------------------------------------------------------------
# routing_accuracy
# ---------------------------------------------------------------------------


class TestRoutingAccuracy:
    def test_perfect_predictions(self):
        pred = np.array([0, 1, 2, 3])
        true = np.array([0, 1, 2, 3])
        assert routing_accuracy(pred, true) == pytest.approx(1.0)

    def test_all_wrong(self):
        pred = np.array([0, 0, 0, 0])
        true = np.array([1, 1, 1, 1])
        assert routing_accuracy(pred, true) == pytest.approx(0.0)

    def test_half_correct(self):
        pred = np.array([0, 1, 0, 1])
        true = np.array([0, 1, 1, 0])
        assert routing_accuracy(pred, true) == pytest.approx(0.5)

    def test_single_element(self):
        assert routing_accuracy(np.array([2]), np.array([2])) == pytest.approx(1.0)
        assert routing_accuracy(np.array([2]), np.array([3])) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Cross-metric consistency checks
# ---------------------------------------------------------------------------


class TestCrossMetricConsistency:
    def test_3x3_all_metrics(self, known_3x3):
        """Verify all 4 matrix-based metrics on the same 3x3 matrix."""
        # Final row: [0.4, 0.6, 0.95]
        assert average_success(known_3x3) == pytest.approx(1.95 / 3)

        # BWT: (0.9-0.4 + 0.85-0.6) / 2 = (0.5+0.25)/2 = 0.375
        assert backward_transfer(known_3x3) == pytest.approx(0.375)

        # Forgetting: peaks[0]=0.9, peaks[1]=0.85
        # drops: 0.9-0.4=0.5, 0.85-0.6=0.25 → mean=0.375
        assert forgetting(known_3x3) == pytest.approx(0.375)

        # Operational forgetting with equal weights = forgetting
        w = np.ones(3)
        assert operational_forgetting(known_3x3, w) == pytest.approx(0.375)

    def test_forgetting_geq_backward_transfer(self):
        """Forgetting >= backward_transfer always holds (since peak >= just-trained)."""
        rng = np.random.RandomState(42)
        for _ in range(10):
            t = rng.randint(2, 6)
            r = rng.rand(t, t)
            f = forgetting(r)
            bt = backward_transfer(r)
            assert f >= bt - 1e-12, f"forgetting={f} < backward_transfer={bt}"

    def test_perfect_lower_tri_all_metrics(self, perfect_lower_triangular):
        """Perfect lower-triangular: no forgetting, no backward transfer."""
        assert average_success(perfect_lower_triangular) == pytest.approx(1.0)
        assert backward_transfer(perfect_lower_triangular) == pytest.approx(0.0)
        assert forgetting(perfect_lower_triangular) == pytest.approx(0.0)
        w = np.ones(3)
        assert operational_forgetting(perfect_lower_triangular, w) == pytest.approx(0.0)
