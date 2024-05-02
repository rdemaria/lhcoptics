from lhcoptics import LHCOptics, LHCMadModel


inj = LHCOptics.from_json("data/opt_inj.json")
inj.set_madx_model("acc-models-lhc/lhc.seq")
