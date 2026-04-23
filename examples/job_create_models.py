from lhcoptics import LHCXsuiteModel


model=LHCXsuiteModel.from_json("data/hllhc/lhc.json")
model.update_from_madx_optics("data/hllhc/optics.madx")
model.env.b1.twiss()

knob=model.get_knob_by_xdeps("dqx.b1")


