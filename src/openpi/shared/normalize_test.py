import numpy as np
import pytest

import openpi.shared.normalize as normalize


def test_normalize_update():
    arr = np.arange(12).reshape(4, 3)  # 4 vectors of length 3

    stats = normalize.RunningStats()
    for i in range(len(arr)):
        stats.update(arr[i : i + 1])  # Update with one vector at a time
    results = stats.get_statistics()

    assert np.allclose(results.mean, np.mean(arr, axis=0))
    assert np.allclose(results.std, np.std(arr, axis=0))


@pytest.mark.parametrize(
    "new_extrema", [(2.0,), (-1.0,), (-1.0, 2.0), (2.0, 3.0), (-1.0, -2.0), (2.0, -1.0, 3.0, -2.0)]
)
def test_normalize_quantiles_with_expanding_range(new_extrema):
    # Only the first feature expands. The others retain their range or remain constant.
    arr = np.repeat([[0.0, 0.0, 5.0], [1.0, 1.0, 5.0]], 200, axis=0)
    stats = normalize.RunningStats()
    stats.update(arr)
    for value in new_extrema:
        batch = np.array([[value, 0.5, 5.0]])
        stats.update(batch)
        arr = np.concatenate([arr, batch])
        results = stats.get_statistics()

        # Allow for histogram quantization across repeated range expansions.
        np.testing.assert_allclose(results.q01, np.quantile(arr, 0.01, axis=0), atol=0.005)
        np.testing.assert_allclose(results.q99, np.quantile(arr, 0.99, axis=0), atol=0.005)
        np.testing.assert_allclose(results.mean, np.mean(arr, axis=0))
        np.testing.assert_allclose(results.std, np.std(arr, axis=0), atol=1e-7)


def test_normalize_quantiles_after_constant_batch():
    stats = normalize.RunningStats()
    stats.update(np.zeros((100, 2)))
    stats.update(np.array([[1.0, 0.0]]))
    results = stats.get_statistics()

    np.testing.assert_allclose(results.q01, [0.0, 0.0], atol=0.001)
    np.testing.assert_allclose(results.q99, [0.0, 0.0], atol=0.001)


def test_serialize_deserialize():
    stats = normalize.RunningStats()
    stats.update(np.arange(12).reshape(4, 3))  # 4 vectors of length 3

    norm_stats = {"test": stats.get_statistics()}
    norm_stats2 = normalize.deserialize_json(normalize.serialize_json(norm_stats))
    assert np.allclose(norm_stats["test"].mean, norm_stats2["test"].mean)
    assert np.allclose(norm_stats["test"].std, norm_stats2["test"].std)


def test_multiple_batch_dimensions():
    # Test with multiple batch dimensions: (2, 3, 4) where 4 is vector dimension
    batch_shape = (2, 3, 4)
    arr = np.random.rand(*batch_shape)

    stats = normalize.RunningStats()
    stats.update(arr)  # Should handle (2, 3, 4) -> reshape to (6, 4)
    results = stats.get_statistics()

    # Flatten batch dimensions and compute expected stats
    flattened = arr.reshape(-1, arr.shape[-1])  # (6, 4)
    expected_mean = np.mean(flattened, axis=0)
    expected_std = np.std(flattened, axis=0)

    assert np.allclose(results.mean, expected_mean)
    assert np.allclose(results.std, expected_std)
