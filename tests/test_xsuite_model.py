
import pytest

from lhcoptics.model_xsuite import LHCXsuiteModel


@pytest.mark.long
def test_from_madx_sequence(hllhc_data_dir):
    model = LHCXsuiteModel.from_madx_sequence(hllhc_data_dir / "lhc.seq")

    assert set(model.env.lines) == {"b1", "b2"}
    assert model.sequence[1] is model.env.b1
    assert model.sequence[2] is model.env.b2
    assert model.p0c == 450e9
