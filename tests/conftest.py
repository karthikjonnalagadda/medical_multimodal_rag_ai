from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def tmp_path():
    """
    Workspace-local replacement for pytest's default tmp_path.

    Some Windows environments lock the shared system temp directory, which causes
    PermissionError during fixture setup. This keeps test scratch space inside the
    repository and cleans it up after each test.
    """
    base_dir = Path(__file__).resolve().parent.parent / ".pytest_runtime"
    base_dir.mkdir(parents=True, exist_ok=True)
    path = Path(tempfile.mkdtemp(prefix="case_", dir=base_dir))
    yield path
    shutil.rmtree(path, ignore_errors=True)
