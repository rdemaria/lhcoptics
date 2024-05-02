from lhcoptics import LHCOptics, LHCMadModel

# test round trip
model = LHCMadModel.from_madxfile("flat_model.madx")
tw1 = model.b1.twiss()
tw1.x
tw1.header["q2"]

inj = LHCOptics.from_madxfile("flat_model.madx", make_model="madx")
inj.to_json("opt_test.json")

inj.model.diff(model)

inj = LHCOptics.from_json("opt_test.json")
inj.set_madx_model("acc-models-lhc/lhc.seq")

tw1 = inj.model.b1.twiss()
tw1.x
tw1.header["q2"]

inj.model.diff(model)
