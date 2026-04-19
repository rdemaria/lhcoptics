from lhcoptics import LHCOptics, LHCXsuiteModel, LHCMadxModel

mx=LHCXsuiteModel.from_json("data/hllhc/lhc.json")
mx.load("data/hllhc/optics.madx")

mm=LHCMadxModel.from_madx_scripts("data/lhc/lhc.seq","data/lhc/optics.madx")

opt=LHCOptics.from_model(mx)
