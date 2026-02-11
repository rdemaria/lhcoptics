from lhcoptics import LHCCircuits, LHCOptics

LHCOptics.set_repository("2025")

inj = LHCOptics.from_madxfile("000_vdm_model.madx", make_model="xsuite")
inj.model.to_json("data/lhc.json")
inj.set_params()
inj.to_json("data/opt_inj2025.json")

inj = LHCOptics.from_madxfile("000_vdm_model.madx", make_model="xsuite", sliced=True)
inj.model.to_json("lhc_sliced.json")

inj = LHCOptics.from_madxfile("001_vdm_model.madx", make_model="xsuite")
inj.model.to_json("lhc.json")
inj.set_params()
inj.to_json("data/opt_2025_50m.json")

inj = LHCOptics.from_madxfile("001_vdm_model.madx", make_model="xsuite", sliced=True)
inj.model.to_json("opt_2025_50m.json")



