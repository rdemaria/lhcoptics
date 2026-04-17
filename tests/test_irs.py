from lhcoptics import LHCIR1


def test_ir1_from_xsuite_model_hllhc(model_xsuite_hl):
    ir = LHCIR1.from_model(model_xsuite_hl)

    assert ir.name == "ir1"
    assert ir.variant == "hl"
    assert ir.strengths["kqx1.l1"] == model_xsuite_hl["kqx1.l1"]
    assert "betxip1b1" in ir.params
    assert len(ir.knobs) > 0
