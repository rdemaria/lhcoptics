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
opt.ir1.set_params()
opt.ir5.set_params()
opt.ir1.check_match()
opt.ir1.match().solve()
opt.ir2.match().solve()
opt.ir3.match().solve()
opt.ir4.match().solve()
opt.ir5.match().solve()
opt.ir6.match().solve()
opt.ir7.match().solve()
opt.ir8.match().solve()
opt.check()
opt.update()
opt.to_madx("opt_round_150_1500.madx")



