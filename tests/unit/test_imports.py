"""All modules can be imported (function bodies are not yet implemented, but signatures and dependency directions must be correct)."""

import importlib
import pkgutil

import callkit


def test_all_modules_import():
    names = [m.name for m in pkgutil.walk_packages(callkit.__path__, prefix="callkit.")]
    assert "callkit.record" in names
    for name in names:
        importlib.import_module(name)


def test_public_api():
    assert set(callkit.__all__) >= {"Client", "Endpoint", "Record", "extend", "fork"}
