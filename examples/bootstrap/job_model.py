from lhcoptics import LHCOptics

LHCOptics.set_repository("2024")

repo_model="acc-models-lhc/scenarios/pp_lumi/inj/model.madx"
inj=LHCOptics.from_madxfile(repo_model,xsuite_model="acc-models-lhc/xsuite/lhc.json")
inj.ir5.plot()

repo_model="acc-models-lhc/scenarios/pp_lumi/endoframp/model.madx"
eor=LHCOptics.from_madxfile(repo_model,xsuite_model="acc-models-lhc/xsuite/lhc.json")
eor.ir5.plot()

eor2=LHCOptics.from_madxfile(repo_model,make_model="xsuite")
eor2.ir5.plot()


repo_model="acc-models-lhc/scenarios/pp_lumi/endoframp/model.madx"
eor=LHCOptics.from_madxfile(repo_model,xsuite_model="acc-models-lhc/xsuite/lhc.json")
eor.ir5.plot()
