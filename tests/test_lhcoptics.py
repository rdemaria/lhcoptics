import pytest
from lhcoptics import LHC

@pytest.fixture
def lhc_optics(request):
    lhc = LHC()
    opt= lhc.y2025.pp.ramp[0]
    return opt

def test_lhcoptics(lhc_optics):
    opt = lhc_optics
    assert opt is not None
