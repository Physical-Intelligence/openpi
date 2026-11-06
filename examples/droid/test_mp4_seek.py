from convert_droid_data_to_lerobot import MP4Reader
import cv2
import numpy as np
import pytest


def test_backward_seek_reads_requested_camera_frame(tmp_path):
    path = str(tmp_path / "camera.avi")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 10, (32, 24))
    if not writer.isOpened():
        pytest.skip("MJPEG encoder unavailable")
    try:
        for value in (20, 60, 100, 140):
            writer.write(np.full((24, 32, 3), value, dtype=np.uint8))
    finally:
        writer.release()

    reader = MP4Reader(path, "camera")
    reader.set_reading_parameters()
    try:
        for index in (3, 1, 2, 0):
            reader.set_frame_index(index)
            frame = reader.read_camera()["image"]["camera"]
            assert frame.mean() == pytest.approx((20, 60, 100, 140)[index], abs=2)
    finally:
        reader.disable_camera()
