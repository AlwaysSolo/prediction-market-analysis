from __future__ import annotations

import importlib


def test_main_module_imports():
    module = importlib.import_module("main")
    assert hasattr(module, "analyze")
    assert hasattr(module, "index")
    assert hasattr(module, "setup")
