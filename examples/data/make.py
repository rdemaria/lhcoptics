from lhcoptics import LHCCircuits, LHCOptics

inj = LHCOptics.from_madx_scripts("model.madx", attach_model=True)
inj.model.to_json("lhc.json")
inj.set_params()
inj.to_json("opt_inj.json")

inj.model.make_thin_model()
inj.model.to_json("lhc_sliced.json")


