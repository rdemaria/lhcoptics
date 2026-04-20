from lhcoptics import LHCDev, LHCOptics
from lhcoptics import LHCMadxModel, LHCXsuiteModel

mm=LHCMadxModel.from_madx_scripts(madxfile,basedir=madxdir)
mx=LHCXsuiteModel.from_json(lhc.y2025.get_xsuite_json())



# Injection 2025
lhc=LHCDev()
madxfile=lhc.y2025.pp.ramp.get_madx_model_file(0)
madxdir=lhc.y2025.pp.ramp.get_madx_model_basedir(0)
opt=LHCOptics.from_madx_scripts(madxfile,basedir=madxdir)
opt.check()
opt.check_params()
opt.plot(yl='x y')

# End of levelling
madxfile=lhc.y2025.pp.levelling.get_madx_model_file(-1)
madxdir=lhc.y2025.pp.levelling.get_madx_model_basedir(-1)
opt=LHCOptics.from_madx_scripts(madxfile,basedir=madxdir, attach_model=True)
opt=LHCOptics.from_madx_scripts(madxfile,basedir=madxdir)
opt.check()
opt.check_params()
opt.plot(yl='x y')

