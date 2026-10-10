import pathlib
from unittest import mock

import pytest

import openpi.shared.download as download


@pytest.fixture(scope="session", autouse=True)
def set_openpi_data_home(tmp_path_factory):
    temp_dir = tmp_path_factory.mktemp("openpi_data")
    with pytest.MonkeyPatch().context() as mp:
        mp.setenv("OPENPI_DATA_HOME", str(temp_dir))
        yield


def test_download_local(tmp_path: pathlib.Path):
    local_path = tmp_path / "local"
    local_path.touch()

    result = download.maybe_download(str(local_path))
    assert result == local_path

    with pytest.raises(FileNotFoundError):
        download.maybe_download("bogus")


@pytest.mark.parametrize("is_directory", [False, True])
@pytest.mark.parametrize("write_partial", [False, True])
def test_download_failure_is_not_cached(tmp_path: pathlib.Path, monkeypatch, is_directory, write_partial):
    monkeypatch.setenv("OPENPI_DATA_HOME", str(tmp_path))
    remote_path = "memory://test/model"
    local_path = tmp_path / "test/model"
    filesystem = mock.Mock()
    filesystem.info.return_value = {
        "name": remote_path,
        "type": "directory" if is_directory else "file",
        "size": 8,
    }
    filesystem.du.return_value = 8
    error = OSError("Download interrupted")

    def get(url, destination, *, recursive):
        assert url == remote_path
        assert recursive == is_directory
        first_attempt = filesystem.get.call_count == 1
        if not first_attempt or write_partial:
            destination = pathlib.Path(destination)
            if is_directory:
                destination.mkdir(parents=True, exist_ok=True)
                destination /= "weights.bin"
            destination.write_bytes(b"partial" if first_attempt else b"complete")
        if first_attempt:
            raise error

    filesystem.get.side_effect = get
    monkeypatch.setattr(download.fsspec.core, "url_to_fs", lambda *args, **kwargs: (filesystem, remote_path))

    with pytest.raises(OSError, match="Download interrupted") as exc_info:
        download.maybe_download(remote_path)
    assert exc_info.value is error
    assert not local_path.exists()

    result = download.maybe_download(remote_path)
    assert result == local_path
    payload_path = result / "weights.bin" if is_directory else result
    assert payload_path.read_bytes() == b"complete"
    assert filesystem.get.call_count == 2

    assert download.maybe_download(remote_path) == local_path
    assert filesystem.get.call_count == 2


def test_download_gs_dir():
    remote_path = "gs://openpi-assets/testdata/random"

    local_path = download.maybe_download(remote_path)
    assert local_path.exists()

    new_local_path = download.maybe_download(remote_path)
    assert new_local_path == local_path


def test_download_gs():
    remote_path = "gs://openpi-assets/testdata/random/random_512kb.bin"

    local_path = download.maybe_download(remote_path)
    assert local_path.exists()

    new_local_path = download.maybe_download(remote_path)
    assert new_local_path == local_path


def test_download_fsspec():
    remote_path = "gs://big_vision/paligemma_tokenizer.model"

    local_path = download.maybe_download(remote_path, gs={"token": "anon"})
    assert local_path.exists()

    new_local_path = download.maybe_download(remote_path, gs={"token": "anon"})
    assert new_local_path == local_path
