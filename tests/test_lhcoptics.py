from lhcoptics import LHCOptics


def test_lhcoptics_from_model_hllhc(model_xsuite_hl):
    optics = LHCOptics.from_model(model_xsuite_hl)

    assert optics.variant == "hl"
    assert len(optics.irs) == 8
    assert len(optics.arcs) == 8
    assert optics.params["qxb1"] == model_xsuite_hl["qxb1"]
    assert optics.ir1.variant == "hl"
    assert optics.ir1.strengths["kqx1.l1"] == model_xsuite_hl["kqx1.l1"]
    assert len(optics.knobs) > 0
