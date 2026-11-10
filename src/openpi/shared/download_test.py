import http.server
import pathlib
import threading
import urllib.parse

import fsspec
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


@pytest.fixture
def http_origin():
    requests = []

    class Origin(http.server.BaseHTTPRequestHandler):
        def body(self):
            query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            return ("model-version:" + query.get("version", ["plain"])[0]).encode()

        def do_HEAD(self):  # noqa: N802
            self.send_response(200)
            self.send_header("Content-Length", str(len(self.body())))
            self.end_headers()

        def do_GET(self):  # noqa: N802
            requests.append(self.path)
            self.do_HEAD()
            self.wfile.write(self.body())

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Origin)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_download_http_without_query(http_origin):
    origin, requests = http_origin
    path = download.maybe_download(origin + "/weights.bin")
    assert path.read_bytes() == b"model-version:plain"
    assert path == download.get_cache_dir() / urllib.parse.urlparse(origin).netloc / "weights.bin"
    count = len(requests)
    assert download.maybe_download(origin + "/weights.bin") == path
    assert len(requests) == count


@pytest.mark.parametrize("prefill", [False, True])
def test_download_http_query_versions(http_origin, prefill):
    origin, requests = http_origin
    if prefill:
        plain = download.maybe_download(origin + "/weights.bin")
        assert plain.read_bytes() == b"model-version:plain"

    first = download.maybe_download(origin + "/weights.bin?version=first")
    assert first.read_bytes() == b"model-version:first"
    second = download.maybe_download(origin + "/weights.bin?version=second")
    assert second.read_bytes() == b"model-version:second"
    assert first != second
    assert first.suffix == second.suffix == ".bin"
    count = len(requests)
    assert download.maybe_download(origin + "/weights.bin?version=first") == first
    assert len(requests) == count
    assert first.read_bytes() == b"model-version:first"
    if prefill:
        assert plain.read_bytes() == b"model-version:plain"


def test_download_memory_file_and_directory(tmp_path):
    origin = f"memory://{tmp_path.name}"
    fs = fsspec.filesystem("memory")
    fs.pipe(origin + "/single.bin", b"single")
    fs.pipe(origin + "/directory/weights.bin", b"weights")
    file = download.maybe_download(origin + "/single.bin")
    directory = download.maybe_download(origin + "/directory")
    assert file.read_bytes() == b"single"
    assert (directory / "weights.bin").read_bytes() == b"weights"


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
