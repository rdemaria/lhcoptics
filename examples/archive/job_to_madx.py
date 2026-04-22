from lhcoptics import LHCMadxModel, LHCOptics

inj = LHCOptics.from_json("data/opt_inj.json")
inj.set_madx_model("acc-models-lhc/lhc.seq")
