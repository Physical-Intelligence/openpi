import pathlib

import pytest

import openpi.shared.download as download


def test_download_local(tmp_path: pathlib.Path):
    local_path = tmp_path / "local"
    local_path.touch()

    result = download.download(str(local_path))
    assert result == local_path

    with pytest.raises(FileNotFoundError):
        download.download("bogus")


def test_download_s3():
    remote_path = "s3://openpi-assets-internal/checkpoints/pi0_base"

    local_path = download.download(remote_path)
    assert local_path.exists()

    new_local_path = download.download(remote_path)
    assert new_local_path == local_path


def test_download_fsspec():
    remote_path = "gs://big_vision/paligemma_tokenizer.model"

    local_path = download.download(remote_path, gs={"token": "anon"})
    assert local_path.exists()

    new_local_path = download.download(remote_path, gs={"token": "anon"})
    assert new_local_path == local_path
