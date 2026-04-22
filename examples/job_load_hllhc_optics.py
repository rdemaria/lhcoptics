from lhcoptics import LHCDev, LHCOptics
from lhcoptics import LHCMadxModel, LHCXsuiteModel

lhc=LHCDev()

# Injection hl19
fn=lhc.hl19.basedir/"strengths/cycle_round_v0/opt_6000.madx"
opt=LHCOptics.from_madx_optics(fn)
opt.set_xsuite_model(lhc.hl19.get_xsuite_json())
opt.check()
opt.set_circuits(lhc.hl19.get_circuits_json())
opt.match()


# End of levelling hl19
fn=lhc.hl19.basedir/"strengths/cycle_round_v0/opt_150.madx"
opt=LHCOptics.from_madx_optics(fn)
opt.set_xsuite_model(lhc.hl19.get_xsuite_json())
opt.check()
opt.set_circuits(lhc.hl19.get_circuits_json())
opt.match()
opt.update(params=False,verbose=True)
opt.to_madx("opt_150_v0.madx")






# End of levelling thin hl19
lhc=LHCDev()
fn=lhc.hl19.basedir/"strengths/cycle_round_v0/opt_150_thin.madx"
opt=LHCOptics.from_madx_optics(fn)
opt.set_xsuite_model(lhc.hl19.get_xsuite_json(thin=True))
opt.check()


mx=lhc.hl19.get_xsuite_model()
fn=lhc.hl19.basedir/"strengths/cycle_round_v0/opt_150.madx"
mx.update_from_madx_optics(fn)
mx.b1.twiss().show("ip.*","betx bety px py")
mx.b2.twiss().show("ip.*","betx bety px py")
opt2=LHCOptics.from_model(mx)
h
