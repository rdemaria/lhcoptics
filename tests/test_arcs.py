#def test_arc_params_from_variables_on_hllhc_model(optics_hl):
#    arc = optics_hl.a12
#    params = arc.get_params(mode="from_variables")
#
#    assert arc.variant == "hl"
#    assert set(arc.phase_names) <= set(params)
#
#
#def test_arc_roundtrip_preserves_variant(optics_hl):
#    arc = optics_hl.a12
#
#    restored = arc.__class__.from_dict(arc.to_dict())
#
#    assert restored.name == arc.name
#    assert restored.variant == "hl"
#
#
#def test_arc_match_smoke_solve_false(fresh_optics_hl):
#    match = fresh_optics_hl.a12.match(solve=False, verbose=False)
#
#    assert len(match.targets) == 4
#    assert len(match.vary) >= 2
