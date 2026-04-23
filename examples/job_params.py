from lhcoptics import LHCOptics

opt=LHCOptics.from_madx_optics("data/hllhc/optics.madx")
opt.set_xsuite_model("data/hllhc/lhc.json")
opt.get_params_from_variables()
opt.get_params_from_variables(full=True)
opt.get_params_from_twiss()
opt.get_params_from_twiss(full=True)
opt.get_params_from_twiss(mode="full")
opt.get_params_from_twiss(full=True,mode="full")


mx=LHCXsuiteModel.from_json("data/hllhc/lhc.json")
mx.load("data/hllhc/optics.madx")

mm=LHCMadxModel.from_madx_scripts("data/lhc/lhc.seq","data/lhc/optics.madx")

opt=LHCOptics.from_model(mx)
