from lhcoptics import LHCOptics

inj = LHCOptics.from_json("data/opt_inj.json", xsuite_model="data/lhc.json")
inj.name = "inj"
inj.params["match_inj"] = True
inj.set_circuits_from_json("data/lhccircuits.json")
inj.knobs_off()
inj.set_params()
inj.update()

# IR4 (from hl)
hlopt = LHCOptics.from_madxfile("opt_ramp_2000_1500.madx")
inj.ir4.update(hlopt.ir4, verbose=True).update_model()
inj.ir4.match().solve()

inj.check()

inj.update().to_json("inj_nophaseknob.json")

inj.model["phase_change.b1"] = 0
inj.model["phase_change.b2"] = 0
oldphases = inj.get_phase_arcs()
inj.model["phase_change.b1"] = 1
inj.model["phase_change.b2"] = 1
newphases = inj.get_phase_arcs()
inj.model["phase_change.b1"] = 0
inj.model["phase_change.b2"] = 0
inj.match_phase_arcs(newphases)
inj.match_tune()


inj.update().to_json("inj_newphase.json")

inj2 = LHCOptics.from_json(
    "inj_nophaseknob.json", xsuite_model="data/lhc.json"
)
inj2 = LHCOptics.from_json("inj_newphase.json", xsuite_model="data/lhc.json")
inj2.check()
