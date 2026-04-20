from lhcoptics import LHCDev, LHCOptics

lhc=LHCDev()
madxfile=lhc.y2025.pp.ramp.get_madx_model_file(0)
madxdir=lhc.y2025.pp.ramp.get_madx_model_basedir(0)
opt=LHCOptics.from_madx_scripts(madxfile,basedir=madxdir, attach_model=False)
opt.set_xsuite_model(lhc.y2025.get_xsuite_json())
opt.plot()


