import ridgon
from ridgon import Ridgon, RidgonConfig
from ridgon.ball import Ridgon as BallRidgon, RidgonConfig as BallConfig


def test_public_api_is_narrow():
    assert Ridgon is BallRidgon and RidgonConfig is BallConfig
    assert set(ridgon.__all__) == {"Ridgon", "RidgonConfig", "__version__"}
    assert ridgon.__version__ == "0.11.0"
