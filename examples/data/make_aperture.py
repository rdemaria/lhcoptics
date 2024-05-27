from lhcoptics import LHCOptics

inj=LHCOptics.from_json("opt_inj.json", xsuite_model="lhc.json")
inj.model.make_aperture().to_json("lhcaperture.json")
