"""Library discovery checks; do not require CUDA or a BitBLAS installation."""
from types import SimpleNamespace

import pytest

from vitqat import w2a8_bitblas as ops


@pytest.fixture(autouse=True)
def clear_nvrtc_cache():
    ops._load_nvrtc.cache_clear()
    yield
    ops._load_nvrtc.cache_clear()


def test_pip_nvrtc_absolute_path_and_cached_handle(monkeypatch):
    calls = []
    handle = object()
    monkeypatch.setattr(ops, "distribution", lambda name: SimpleNamespace(
        locate_file=lambda path: "/venv/site-packages/" + path))

    def load(path, mode):
        calls.append((path, mode))
        return handle

    monkeypatch.setattr(ops.ctypes, "CDLL", load)
    assert ops._load_nvrtc() is handle
    assert ops._load_nvrtc() is handle
    assert calls == [("/venv/site-packages/nvidia/cuda_nvrtc/lib/libnvrtc.so.12", ops.ctypes.RTLD_GLOBAL)]


def test_system_nvrtc_without_pip_package(monkeypatch):
    def missing(name):
        raise ops.PackageNotFoundError(name)

    monkeypatch.setattr(ops, "distribution", missing)
    monkeypatch.setattr(ops.ctypes, "CDLL", lambda path, mode: path)
    assert ops._load_nvrtc() == "libnvrtc.so.12"


def test_nvrtc_error_identifies_attempted_path(monkeypatch):
    monkeypatch.setattr(ops, "distribution", lambda name: SimpleNamespace(locate_file=lambda path: "/broken/" + path))

    def fail(path, mode):
        raise OSError("missing dependency")

    monkeypatch.setattr(ops.ctypes, "CDLL", fail)
    with pytest.raises(RuntimeError, match="/broken/.*missing dependency"):
        ops._load_nvrtc()
