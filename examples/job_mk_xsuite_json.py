from lhcoptics import LHCOptics

lhc=LHCOptics.from_madxfile("job_model.madx",model='xsuite')
lhc.model.to_json("lhc.json")
lhc=LHCOptics.from_madxfile("job_model.madx",model='xsuite',sliced=True)
lhc.model.to_json("lhc_sliced.json")

