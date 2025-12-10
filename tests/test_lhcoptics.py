from pathlib import Path

import pytest
import xtrack as xt
from lhcoptics import LHCOptics

DATA_DIR = Path(__file__).resolve().parent.parent / "examples" / "data" / "hllhc"


@pytest.fixture
def lhc_model():
    return xt.load(str(DATA_DIR / "lhc.json"))

def test_lhcoptics_from_xsuite(lhc_model):
    opt = LHCOptics.from_xsuite(
        lhc_model,
        variant="hl",
        knob_structure=str(DATA_DIR / "knobs.yaml"),
        circuits=str(DATA_DIR / "lhccircuits.json"),
    )
    assert opt is not None
