"""Private framed-pickle HDF5 worker used by the legacy collection adapter."""

from __future__ import annotations

import argparse
import pickle
import struct
import sys

from examples.ur5_twinwrist.record_hdf5 import EpisodeWriter
from examples.ur5_twinwrist.record_hdf5 import recover_stale_episodes


def _read(stream):
    header = stream.read(8)
    if not header:
        return None
    if len(header) != 8:
        raise EOFError("truncated writer message header")
    size = struct.unpack("!Q", header)[0]
    payload = stream.read(size)
    if len(payload) != size:
        raise EOFError("truncated writer message")
    return pickle.loads(payload)


def _write(stream, value) -> None:
    payload = pickle.dumps(value, protocol=5)
    stream.write(struct.pack("!Q", len(payload)))
    stream.write(payload)
    stream.flush()


def serve(root: str) -> int:
    reader, writer = sys.stdin.buffer, sys.stdout.buffer
    episode = None
    recovered = recover_stale_episodes(root)
    while True:
        try:
            message = _read(reader)
            if message is None:
                break
            operation = message["op"]
            if operation == "ping":
                result = [str(path) for path in recovered]
                recovered = []
            elif operation == "start":
                if episode is not None:
                    raise RuntimeError("an episode is already active")
                episode = EpisodeWriter(
                    root,
                    int(message["episode_id"]),
                    tuple(message["image_shape"]),
                    dict(message["attrs"]),
                )
                result = None
            elif operation == "append":
                if episode is None:
                    raise RuntimeError("no active episode")
                episode.append(message["observation"], message["action"])
                result = None
            elif operation == "finish":
                if episode is None:
                    raise RuntimeError("no active episode")
                result = str(episode.finish(success=True))
                episode = None
            elif operation == "reject":
                if episode is None:
                    raise RuntimeError("no active episode")
                result = str(episode.reject(str(message["reason"])))
                episode = None
            elif operation == "close":
                if episode is not None:
                    episode.reject("writer closed with active episode")
                    episode = None
                _write(writer, {"ok": True, "result": None})
                return 0
            else:
                raise ValueError(f"unknown writer operation: {operation}")
            _write(writer, {"ok": True, "result": result})
        except BaseException as exc:
            _write(writer, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})
            return 1
    if episode is not None:
        episode.reject("writer input closed with active episode")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    return serve(parser.parse_args().root)


if __name__ == "__main__":
    raise SystemExit(main())
