"""
Tests for src/utils/io.py (atomic writes) and src/utils/keys.py
(API key validation) -- both ported from an external repository audit.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.io import atomic_write_text, save_json, load_json, sha256_file
from src.utils.keys import get_key


def test_atomic_write_creates_file_with_correct_content():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "sub" / "file.txt"
        atomic_write_text(path, "hello world")
        assert path.read_text() == "hello world"


def test_atomic_write_creates_parent_directories():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "a" / "b" / "c" / "file.txt"
        atomic_write_text(path, "nested")
        assert path.exists()


def test_atomic_write_leaves_no_temp_files_behind():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "file.txt"
        atomic_write_text(path, "content")
        remaining = list(Path(d).iterdir())
        assert remaining == [path]


def test_save_and_load_json_round_trip():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "results.json"
        payload = {"run_id": "abc123", "rmse": 0.364, "nested": {"a": [1, 2, 3]}}
        save_json(path, payload)
        loaded = load_json(path)
        assert loaded == payload


def test_sha256_file_matches_known_hash():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "data.txt"
        path.write_text("hello world")
        import hashlib
        expected = hashlib.sha256(b"hello world").hexdigest()
        assert sha256_file(path) == expected


def test_get_key_returns_real_value():
    os.environ["TEST_KEY_XYZ"] = "a-real-looking-value"
    try:
        assert get_key("TEST_KEY_XYZ") == "a-real-looking-value"
    finally:
        del os.environ["TEST_KEY_XYZ"]


def test_get_key_raises_on_missing_required_key():
    os.environ.pop("TEST_KEY_MISSING", None)
    with pytest.raises(RuntimeError, match="Missing required key"):
        get_key("TEST_KEY_MISSING")


def test_get_key_raises_on_placeholder_value():
    os.environ["TEST_KEY_PLACEHOLDER"] = "sk-placeholder-do-not-use"
    try:
        with pytest.raises(RuntimeError, match="Missing required key"):
            get_key("TEST_KEY_PLACEHOLDER")
    finally:
        del os.environ["TEST_KEY_PLACEHOLDER"]


def test_get_key_optional_returns_default_when_missing():
    os.environ.pop("TEST_KEY_OPTIONAL", None)
    assert get_key("TEST_KEY_OPTIONAL", required=False, default="") == ""


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
