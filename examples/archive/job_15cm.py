import xtrack as xt
from lhcoptics import LHCOptics

lhc = xt.load("acc-models-lhc/xsuite/lhc.json")
opt0 = LHCOptics.from_xsuite(lhc)
lhc.vars.load("acc-models-lhc/strengths/round/opt_round_150_1500_optphases.madx")
lhc.set_particle_ref(p0c=6800e9)
lhc.b1.twiss().cols["betx bety"].rows["ip.*"]
lhc.b2.twiss().cols["betx bety"].rows["ip.*"]

opt = LHCOptics.from_xsuite(
    lhc,
    knob_structure="acc-models-lhc/xsuite/knobs.yaml",
    variant="hl",
    circuits="lhccircuits.json",
    params_mode="from_variables"
)
opt.check()
opt.ir1.set_params() # set betxip1b1=0.5
opt.ir5.set_params() # set betyip1b1=0.5
opt.ir1.check_match()
opt.ir1.match().solve()
opt.ir2.match().solve()
opt.ir3.match().solve()
opt.ir4.match().solve()
opt.ir5.match().solve()
opt.ir6.match().solve()
opt.ir7.match().solve()
opt.ir8.match().solve()
for ir in opt.ir1,opt.ir2,opt.ir5,opt.ir8:
   ir.match_knobs()

opt.check()
opt.update() # uptate opt from model
opt.update_model() # uptate model from opt
opt.to_madx("opt_round_150_1500.madx")



