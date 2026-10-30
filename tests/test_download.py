"""Parallel byte-range downloads (scripts/download_datasets.py) against a simulated flaky server."""
import importlib.util
import io
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("download_datasets", ROOT / "scripts" / "download_datasets.py")
dd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dd)

PIECE = 100_000
PAYLOAD = np.random.default_rng(0).bytes(30 * PIECE)


class FakeResponse:
    def __init__(self, data: bytes):
        self.buf, self.status = io.BytesIO(data), 206

    def read(self, n):
        return self.buf.read(n)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def fake_server(behaviour):
    """urlopen stand-in serving PAYLOAD by byte range. behaviour(piece, call_number) returns None (serve the
    range), "error" (raise, like a timeout) or "cut" (serve half the range, like a dropped connection)."""
    calls = {}

    def urlopen(req, timeout=None):
        start, end = map(int, req.get_header("Range").split("=")[1].split("-"))
        piece = start // PIECE
        calls[piece] = calls.get(piece, 0) + 1
        mode = behaviour(piece, calls[piece])
        if mode == "error":
            raise urllib.error.URLError("simulated timeout")
        data = PAYLOAD[start:end + 1]
        return FakeResponse(data[: len(data) // 2] if mode == "cut" else data)

    return urlopen


@pytest.fixture
def target(monkeypatch, tmp_path):
    monkeypatch.setattr(dd, "PIECE_MIN", PIECE)
    monkeypatch.setattr(dd, "PIECE_MAX", PIECE)
    monkeypatch.setattr(dd.time, "sleep", lambda s: None)  # no real back-off waits
    item = {"url": "https://example.org/file.zip", "name": "file.zip", "size": len(PAYLOAD), "md5": None}
    return item, tmp_path / "file.zip.part"


def test_flaky_connections_still_complete(target, monkeypatch):
    item, part = target

    def flaky(piece, n):
        if piece % 3 == 0 and n <= 2:
            return "error"
        if piece % 4 == 1 and n == 1:
            return "cut"
        return None

    monkeypatch.setattr(urllib.request, "urlopen", fake_server(flaky))
    dd.fetch_parallel(item, part, 4, print, True, "file.zip")
    assert part.read_bytes() == PAYLOAD


def test_one_dead_piece_does_not_stop_the_others_and_rerun_resumes(target, monkeypatch):
    item, part = target
    monkeypatch.setattr(urllib.request, "urlopen", fake_server(lambda piece, n: "error" if piece == 5 else None))
    with pytest.raises(IOError, match="still missing"):
        dd.fetch_parallel(item, part, 4, print, True, "file.zip")
    got = part.read_bytes()
    assert got[: 5 * PIECE] == PAYLOAD[: 5 * PIECE] and got[6 * PIECE:] == PAYLOAD[6 * PIECE:]

    fetched = []
    monkeypatch.setattr(urllib.request, "urlopen", fake_server(lambda piece, n: fetched.append(piece)))
    dd.fetch_parallel(item, part, 4, print, True, "file.zip")
    assert part.read_bytes() == PAYLOAD
    assert set(fetched) == {5}  # only the missing piece is downloaded again


def test_fetch_skips_a_removed_archive_whose_folder_exists(tmp_path, monkeypatch):
    monkeypatch.setattr(dd, "RAW", tmp_path)
    dest = tmp_path / "src"
    (dest / "data").mkdir(parents=True)  # the extracted folder of data.zip
    rec = {"path": "src/data.zip", "size": 10, "md5": "x", "archive_removed": "2026-10-12"}
    item = {"url": "http://invalid.example/data.zip", "name": "data.zip", "size": 10, "md5": "x"}
    out = dd.fetch(item, dest, {"src/data.zip": rec}, log=lambda m: None)
    assert out["status"] == "extracted; archive removed"
    assert not (dest / "data.zip").exists() and not (dest / "data.zip.part").exists()
