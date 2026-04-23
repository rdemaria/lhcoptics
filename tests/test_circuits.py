import pytest


def test_circuits_from_hllhc_json(circuits_hl):
    circuit = circuits_hl.madname["kqf.a12"]

    assert len(circuits_hl.pcname) > 0
    assert "kqf.a12" in circuits_hl.madname
    assert circuit.calibname in circuits_hl.calibrations

