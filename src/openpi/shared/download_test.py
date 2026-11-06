import pathlib
import subprocess

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


def test_download_gs_dir():
    remote_path = "gs://openpi-assets/testdata/random"

    local_path = download.maybe_download(remote_path)
    assert local_path.exists()

    new_local_path = download.maybe_download(remote_path)
    assert new_local_path == local_path


def test_download_gs():
    remote_path = "gs://openpi-assets/testdata/random/random_512kb.bin"

    local_path = download.maybe_download(remote_path)
    assert local_path.is_file()

    new_local_path = download.maybe_download(remote_path)
    assert new_local_path == local_path


def test_download_fsspec():
    remote_path = "gs://big_vision/paligemma_tokenizer.model"

    local_path = download.maybe_download(remote_path, gs={"token": "anon"})
    assert local_path.exists()

    new_local_path = download.maybe_download(remote_path, gs={"token": "anon"})
    assert new_local_path == local_path


@pytest.mark.parametrize("suffix", ["droid/ranges.json", "checkpoints/model", "checkpoints/model/"])
def test_gsutil_download_preserves_storage_type(tmp_path, monkeypatch, suffix):
    monkeypatch.setenv("OPENPI_DATA_HOME", str(tmp_path))
    monkeypatch.setattr(download.shutil, "which", lambda name: "gsutil")
    url = "gs://openpi-assets/" + suffix
    is_file = suffix.endswith(".json")
    payload = b'{"episode": [0, 8]}'
    # A failed earlier recursive attempt can leave a directory at the old staging path.
    (tmp_path / "openpi-assets" / suffix.rstrip("/")).with_suffix(".partial").mkdir(parents=True)

    def run(cmd, *, check):
        if cmd[1:3] == ["-q", "stat"]:
            assert cmd[3] == url
            return subprocess.CompletedProcess(cmd, 0 if is_file else 1)
        destination = pathlib.Path(cmd[-1])
        if is_file:
            assert cmd[1:-1] == ["cp", url]
            assert not destination.exists()
            destination.write_bytes(payload)
        else:
            assert cmd[1:-1] == ["-m", "cp", "-r", url.rstrip("/") + "/*"]
            (destination / "weights").write_bytes(payload)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(download.subprocess, "run", run)
    path = download.maybe_download(url)
    assert (path if is_file else path / "weights").read_bytes() == payload


def test_gsutil_retries_interrupted_object_copy(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENPI_DATA_HOME", str(tmp_path))
    monkeypatch.setattr(download.shutil, "which", lambda name: "gsutil")
    url = "gs://openpi-assets/droid/ranges.json"
    destinations = []

    def run(cmd, *, check):
        if cmd[1:3] == ["-q", "stat"]:
            return subprocess.CompletedProcess(cmd, 0)
        destination = pathlib.Path(cmd[-1])
        destinations.append(destination)
        if len(destinations) == 1:
            destination.write_bytes(b"partial")
            raise subprocess.CalledProcessError(1, cmd)
        assert not destination.exists()
        destination.write_bytes(b"complete")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(download.subprocess, "run", run)
    with pytest.raises(subprocess.CalledProcessError):
        download.maybe_download(url)
    assert not (tmp_path / "openpi-assets" / "droid" / "ranges.json").exists()
    assert not destinations[0].exists()
    assert download.maybe_download(url).read_bytes() == b"complete"
