from pathlib import Path

import pytest
import xtrack as xt

from lhcoptics import LHCOptics, LHCXsuiteModel
from lhcoptics.circuits import LHCCircuits

HL_DATA_DIR = Path(__file__).resolve().parent.parent / "examples" / "data" / "hllhc"
LHC_DATA_DIR = Path(__file__).resolve().parent.parent / "examples" / "data" / "lhc"
LHC_CIRCUITS_FILE = (
    Path(__file__).resolve().parent.parent / "examples" / "data" / "lhccircuits.json"
)


def pytest_addoption(parser):
    parser.addoption(
        "--run-long",
        action="store_true",
        default=False,
        help="run tests marked as long",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-long"):
        return

    deselected = []
    selected = []
    for item in items:
        if "long" in item.keywords:
            deselected.append(item)
        else:
            selected.append(item)

    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = selected


@pytest.fixture(scope="session")
def circuits_hl():
    return LHCCircuits.from_json(str(HL_DATA_DIR / "lhccircuits.json"))


@pytest.fixture(scope="session")
def circuits_lhc():
    return LHCCircuits.from_json(str(LHC_CIRCUITS_FILE))


@pytest.fixture(scope="session")
def xsuite_model_hl():
    lhc = xt.load(str(HL_DATA_DIR / "lhc.json"))
    lhc.vars.load(str(HL_DATA_DIR / "optics.madx"))
    return LHCXsuiteModel(lhc)


@pytest.fixture(scope="session")
def xsuite_model_lhc():
    return LHCXsuiteModel.from_madx_scripts(
        LHC_DATA_DIR / "lhc.seq",
        LHC_DATA_DIR / "optics.madx",
        stdout=False,
    )


@pytest.fixture(scope="session")
def optics_hl(xsuite_model_hl, circuits_hl):
    return LHCOptics.from_model(xsuite_model_hl.copy(), circuits=circuits_hl)


@pytest.fixture(scope="session")
def optics_lhc(xsuite_model_lhc, circuits_lhc):
    return LHCOptics.from_model(xsuite_model_lhc.copy(), circuits=circuits_lhc)


@pytest.fixture(scope="session")
def madx_optics_hl():
    return HL_DATA_DIR / "optics.madx"
