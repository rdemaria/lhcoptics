import pytest


def test_circuits_from_hllhc_json(circuits_hl):
    circuit = circuits_hl.madname["kqf.a12"]

    assert len(circuits_hl.pcname) > 0
    assert "kqf.a12" in circuits_hl.madname
    assert circuit.calibname in circuits_hl.calibrations


#def test_circuit_limits_cover_loaded_arc_strength(circuits_hl, optics_hl):
#    kval = optics_hl.a12.strengths["kqf.a12"]
#    p0c = optics_hl.params["p0c"]
#
#    lower, upper = circuits_hl.get_klimits("kqf.a12", p0c)
#
#    assert lower < kval < upper
#
#
#def test_circuit_field_current_roundtrip(circuits_hl, optics_hl):
#    circuit = circuits_hl.madname["kqf.a12"]
#    kval = optics_hl.a12.strengths["kqf.a12"]
#    p0c = optics_hl.params["p0c"]
#    brho = p0c / 299792458
#
#    current = circuit.get_current(kval, p0c)
#    field = circuit.get_field(current)
#
#    assert field / brho == pytest.approx(kval)
#
#
#def test_triplet_limits_use_hllhc_special_case(circuits_hl, optics_hl):
#    lower, upper = circuits_hl.get_klimits("kqx1.l1", optics_hl.params["p0c"])
#
#    assert lower < 0
#    assert upper == 0
