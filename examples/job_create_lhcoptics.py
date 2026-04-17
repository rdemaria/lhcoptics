from lhcoptics import LHCOptics

mx=LHCXsuiteModel.from_json("data/hllhc/lhc.json")
mm=LHCMadxModel.from_madxfile("data/lhc/lhc.seq","data/lhc/optics.madx")

LHCOptics._irs[0].from_madx_optics("data/lhc/optics.madx")


