from lhcoptics import LHCOptics, LHCXsuiteModel, LHCMadxModel

mx=LHCXsuiteModel.from_json("data/hllhc/lhc.json")
mx.load("data/hllhc/optics.madx")

mm=LHCMadxModel.from_madx_scripts("data/lhc/lhc.seq","data/lhc/optics.madx")


for IR in LHCOptics._irs:
    ir=IR.from_model(mx)

for Arc in LHCOptics._arcs:
    arc=Arc.from_model(mx)


ir1=LHCOptics._irs[0].from_model(mx)
ir1=LHCOptics._irs[0].from_model(mx)


ir1=LHCOptics._irs[0].from_madx_optics("data/lhc/optics.madx")
ir1=LHCOptics._irs[0].from_madx_optics("data/hllhc/optics.madx")


opt=LHCOptics.from_model(mx)
