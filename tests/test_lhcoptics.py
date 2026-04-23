import pytest
from lhcoptics import LHCOptics


def test_lhcoptics_from_model_hllhc(xsuite_model_hl):
    model=xsuite_model_hl
    optics = LHCOptics.from_model(model)

    assert optics.variant == "hl"
    assert len(optics.irs) == 8
    assert len(optics.arcs) == 8
    assert optics.params["qxb1"] == model["qxb1"]
    assert optics.ir1.variant == "hl"
    assert optics.ir1.strengths["kqx1.l1"] == model["kqx1.l1"]
    assert len(optics.knobs) > 0


def test_lhcoptics_from_madx_optics(madx_optics_hl):
    opt=LHCOptics.from_madx_optics(madx_optics_hl)
    assert opt.variant == "hl"
    assert opt.params["qxb1"] == 62.31
    assert opt.get_params_from_variables()==opt.params


