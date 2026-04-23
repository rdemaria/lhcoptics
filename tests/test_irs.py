import pytest

from lhcoptics import LHCIR1


def test_ir1_from_xsuite_model_hllhc(xsuite_model_hl):
    model=xsuite_model_hl
    ir = LHCIR1.from_model(model)

    assert ir.name == "ir1"
    assert ir.variant == "hl"
    assert ir.strengths["kqx1.l1"] == model["kqx1.l1"]
    assert "betxip1b1" in ir.params
    assert len(ir.knobs) > 0
