from lhcoptics import LHCOptics

inj=LHCOptics.from_json("data/opt_inj.json")
inj.set_xsuite_model("data/lhc.json") # set an xsuite model
inj.set_circuits_from_json("data/lhccircuits.json")
inj.get_params()


