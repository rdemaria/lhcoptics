from lhcoptics import LHCOptics

inj=LHCOptics.from_json("data/opt_inj2025.json", xsuite_model="data/lhc.json")
inj.model.make_aperture().to_json("data/lhcaperture.json")
