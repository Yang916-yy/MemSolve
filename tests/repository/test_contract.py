from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


pytestmark = pytest.mark.repository


def test_repository_contract() -> None:
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, str(root / "tools" / "check_repository.py")],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
