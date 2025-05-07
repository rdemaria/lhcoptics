from lhcoptics import LHCOptics

def test_open_lhcoptics():
    LHCOptics.set_repository("2024")

    inj = LHCOptics.from_madxfile("examples/data/model.madx", make_model="xsuite")

    assert inj.model is not None


