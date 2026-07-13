import numpy as np
import pytest

from openpi_client import msgpack_numpy


@pytest.mark.parametrize(
    "dtype",
    [
        np.dtype("float32"),
        np.dtype("float64"),
        np.dtype("int32"),
        np.dtype("bool"),
    ],
)
def test_msgpack_roundtrip_common_dtypes(dtype):
    arr = np.array([[1, 2], [3, 4]], dtype=dtype)
    roundtripped = msgpack_numpy.unpackb(msgpack_numpy.packb(arr))
    assert roundtripped.dtype == arr.dtype
    assert np.array_equal(roundtripped, arr)


def test_msgpack_roundtrip_nested_structure():
    payload = {
        "actions": np.arange(12, dtype=np.float32).reshape(4, 3),
        "state": np.array([1.0, 2.0], dtype=np.float64),
        "mask": np.array([True, False, True]),
    }
    roundtripped = msgpack_numpy.unpackb(msgpack_numpy.packb(payload))
    assert np.allclose(roundtripped["actions"], payload["actions"])
    assert np.allclose(roundtripped["state"], payload["state"])
    assert np.array_equal(roundtripped["mask"], payload["mask"])


@pytest.mark.skipif(int(np.__version__.split(".")[0]) < 2, reason="numpy 2.x only")
def test_numpy2_string_dtype_scalar_roundtrip():
    scalar = np.str_("robot")
    roundtripped = msgpack_numpy.unpackb(msgpack_numpy.packb(scalar))
    assert str(roundtripped) == "robot"
