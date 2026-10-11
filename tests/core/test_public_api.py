from __future__ import annotations

import pytest

import memsolve
from memsolve import CoreMode, MemSolve, MemSolveConfig
from memsolve.ball import CoreMode as BallCoreMode
from memsolve.ball import MemSolve as BallMemSolve
from memsolve.ball import MemSolveConfig as BallConfig


pytestmark = pytest.mark.core


def test_public_api_is_narrow() -> None:
    assert CoreMode is BallCoreMode
    assert MemSolve is BallMemSolve
    assert MemSolveConfig is BallConfig
    assert set(memsolve.__all__) == {
        "CoreMode",
        "MemSolve",
        "MemSolveConfig",
        "__version__",
    }
    assert memsolve.__version__ == "0.15.0"
