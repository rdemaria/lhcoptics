from lhcoptics import LHCDev, LHCOptics
from lhcoptics import LHCDev, LHCOptics, LHCMadxModel, LHCXsuiteModel
lhc=LHCDev()

# Injection 2025
madxfile=lhc.y2025.pp.ramp.get_madx_model_file(0)
madxdir=lhc.y2025.pp.ramp.get_madx_model_basedir(0)
opt=LHCOptics.from_madx_scripts(madxfile,basedir=madxdir)
"""
ValueError: Knob on_ssep1_h still has dependencies after deletion
"""
opt.check()
opt.check_params()
opt.plot(yl='x y')

# End of levelling
madxfile=lhc.y2025.pp.levelling.get_madx_model_file(-1)
madxdir=lhc.y2025.pp.levelling.get_madx_model_basedir(-1)
opt=LHCOptics.from_madx_scripts(madxfile,basedir=madxdir)
opt.check()
opt.check_params()
opt.plot(yl='x y')

# Shortcut
madxfile=lhc.y2025.pp.levelling.get_madx_model_file(-1)
madxdir=lhc.y2025.pp.levelling.get_madx_model_basedir(-1)
mm=LHCMadxModel.from_madx_scripts(madxfile,basedir=madxdir)
mx=LHCXsuiteModel.from_json(lhc.y2025.get_xsuite_json())

opt=LHCOptics.from_model(mm,attach_model=True)
opt.set_xsuite_model(mx)
#opt.check()
#opt.check_params()
opt.check_data()
#opt.plot(yl='x y')

#
mxm=LHCXsuiteModel.from_cpymad(mm.madx)
mxm2=LHCXsuiteModel(mxm.env.copy())

opt=LHCOptics.from_model(mm,attach_model=False)
opt.set_xsuite_model(mxm2)


