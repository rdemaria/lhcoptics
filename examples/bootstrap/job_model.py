from lhcoptics import LHCOptics

LHCOptics.set_repository("2024")

repo_model="acc-models-lhc/scenarios/pp_lumi/inj/model.madx"
inj=LHCOptics.from_madxfile(repo_model,gen_model='xsuite')
inj.model.to_json("lhc.json")
inj.to_json("opt_inj.json")

inj=LHCOptics.from_json("opt_inj.json",xsuite_model="lhc.json")
inj.ir5.plot()

from lhcoptics import LHC
inj=LHC(2024).pp_lumi.inj.optics.ir5.plot()

repo=LHCOptics.set_repository("2024")
repo_model="acc-models-lhc/scenarios/pp_lumi/inj/model.madx"
inj=LHCOptics.from_madxfile(repo_model,xsuite_model="lhc.json")
inj.ir5.plot()

repo_model="acc-models-lhc/scenarios/pp_lumi/endlev/model.madx"
inj=LHCOptics.from_madxfile(repo_model,xsuite_model="lhc.json")
inj.ir5.plot()