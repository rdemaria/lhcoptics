from lhcoptics import LHCCircuits, LHCOptics

LHCOptics.set_repository("2024")

inj = LHCOptics.from_madxfile("model.madx", make_model="xsuite")
inj.model.to_json("lhc.json")
inj.set_params()
inj.to_json("opt_inj.json")
inj = LHCOptics.from_madxfile("model.madx", make_model="xsuite", sliced=True)
inj.model.to_json("lhc_sliced.json")
