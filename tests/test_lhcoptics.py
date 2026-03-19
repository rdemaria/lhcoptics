def test_lhcoptics_from_xsuite_uses_hllhc_data(optics_hl):
    assert optics_hl.variant == "hl"
    assert len(optics_hl.irs) == 8
    assert len(optics_hl.arcs) == 8
    assert optics_hl.circuits is not None


def test_lhcoptics_copy_preserves_variant(optics_hl):
    copied = optics_hl.copy(name="copy")

    assert copied.variant == "hl"
    assert all(ss.variant == "hl" for ss in copied.irs + copied.arcs)


def test_lhcoptics_table_interp_preserves_variant(optics_hl):
    interp = optics_hl.to_table(optics_hl.copy(name="copy")).interp(0, order=0)

    assert interp.variant == "hl"
    assert interp.ir1.variant == "hl"
    assert interp.a12.variant == "hl"
