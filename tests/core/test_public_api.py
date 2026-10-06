import memsolve
from memsolve import MemSolve, MemSolveConfig
from memsolve.ball import MemSolve as BallMemSolve, MemSolveConfig as BallConfig


def test_public_api_is_narrow():
    assert MemSolve is BallMemSolve and MemSolveConfig is BallConfig
    assert set(memsolve.__all__) == {"MemSolve", "MemSolveConfig", "__version__"}
    assert memsolve.__version__ == "0.14.0"
