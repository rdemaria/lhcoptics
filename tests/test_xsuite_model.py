import pytest
import numpy as np

from lhcoptics.model_xsuite import LHCXsuiteModel


@pytest.mark.long
def test_from_madx_sequence(hllhc_data_dir):
    model = LHCXsuiteModel.from_madx_sequence(hllhc_data_dir / "lhc.seq")

    assert set(model.env.lines) == {"b1", "b2"}
    assert model.sequence[1] is model.env.b1
    assert model.sequence[2] is model.env.b2
    assert model.p0c == 450e9

def test_twiss_from_model(xsuite_model_hl):
    tw1 = xsuite_model_hl.env.b1.twiss()

    assert tw1.s[0] == 0
    assert np.isclose(tw1.betx[0], 0.15)
    assert np.isclose(tw1.bety[0], 0.15)

def test_get_knob(xsuite_model_hl):
    model=xsuite_model_hl
    name="dqx.b1"
    knob = model.get_knob_by_xdeps(name)

    assert knob is not None
    assert knob.name == name
    assert len(knob.weights) > 0
    assert np.all(ww!=0 for ww in knob.weights.values())

    knob2 = model.get_knob_by_probing(name)
    assert np.all(knob.weights[kk]==knob2.weights[kk] for kk in knob.weights)

    model.create_knob(knob)
    knob3 = model.get_knob_by_weight_names(name)
    assert np.all(knob.weights[kk]==knob3.weights[kk] for kk in knob.weights)

    knob4=  model.get_knob(knob)
    assert np.all(knob.weights[kk]==knob4.weights[kk] for kk in knob.weights)
