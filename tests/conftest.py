from pathlib import Path

import pytest
import xtrack as xt

from lhcoptics import LHCOptics
from lhcoptics.circuits import LHCCircuits

DATA_DIR = Path(__file__).resolve().parent.parent / "examples" / "data" / "hllhc"


def load_optics(data_dir):
    return LHCOptics.from_xsuite(
        xt.load(str(data_dir / "lhc.json")),
        variant="hl",
        knob_structure=str(data_dir / "knobs.yaml"),
        circuits=str(data_dir / "lhccircuits.json"),
    )


@pytest.fixture(scope="session")
def hllhc_data_dir():
    return DATA_DIR


@pytest.fixture(scope="session")
def circuits_hl(hllhc_data_dir):
    return LHCCircuits.from_json(str(hllhc_data_dir / "lhccircuits.json"))


@pytest.fixture(scope="session")
def optics_hl(hllhc_data_dir):
    return load_optics(hllhc_data_dir)


@pytest.fixture
def fresh_optics_hl(hllhc_data_dir):
    return load_optics(hllhc_data_dir)
