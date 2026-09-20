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


@pytest.mark.parametrize("batch_size", [1, 7, 100])
@pytest.mark.parametrize(
    ("dtype", "offset", "spread"),
    [(np.float32, 1.0, 1e-4), (np.float32, 100.0, 0.01), (np.float64, 1e8, 1.0)],
)
def test_small_variance_with_nonzero_mean(dtype, offset, spread, batch_size):
    arr = np.array([offset - spread, offset + spread] * 50, dtype=dtype)[:, None]
    stats = normalize.RunningStats()
    for start in range(0, len(arr), batch_size):
        stats.update(arr[start : start + batch_size])
    result = stats.get_statistics()
    reference = arr.astype(np.float64)
    np.testing.assert_allclose(result.mean, reference.mean(axis=0), rtol=1e-12)
    np.testing.assert_allclose(result.std, reference.std(axis=0), rtol=1e-7)


@pytest.mark.parametrize("dtype", [np.uint8, np.int16, np.float16])
def test_moments_do_not_overflow_in_input_dtype(dtype):
    arr = np.array([[200, 100], [220, 100], [240, 100]], dtype=dtype)
    stats = normalize.RunningStats()
    stats.update(arr)
    reference = arr.astype(np.float64)
    np.testing.assert_allclose(stats.get_statistics().std, reference.std(axis=0), atol=1e-12)


def test_merging_batches_includes_difference_between_batch_means():
    arr = np.array([[100, 7]] * 3 + [[102, 7]] * 17, dtype=np.float32)
    stats = normalize.RunningStats()
    stats.update(arr[:3])
    stats.update(arr[3:])
    result = stats.get_statistics()
    np.testing.assert_allclose(result.mean, arr.astype(np.float64).mean(axis=0))
    np.testing.assert_allclose(result.std, arr.astype(np.float64).std(axis=0), atol=1e-12)
