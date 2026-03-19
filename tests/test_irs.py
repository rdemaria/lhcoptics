def test_ir_hllhc_triplet_access(optics_hl):
    ir = optics_hl.ir1

    assert ir.variant == "hl"
    assert ir["kqx1.l1"] == ir.strengths["kqx1.l1"]
    assert ir["kqx2.l1"] == ir.strengths["kqx2a.l1"]
    assert ir["kqx3.l1"] == ir.strengths["kqx3.l1"]


def test_ir_params_from_variables_and_roundtrip(optics_hl):
    ir = optics_hl.ir1
    params = ir.get_params(mode="from_variables")
    restored = ir.__class__.from_dict(ir.to_dict())

    assert "betxip1b1" in params
    assert "muxip1b1" in params
    assert restored.variant == "hl"


def test_ir_match_smoke_solve_false(fresh_optics_hl):
    match = fresh_optics_hl.ir1.match(solve=False, verbose=False)

    assert len(match.targets) > 0
    assert len(match.vary) > 0
