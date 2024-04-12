from lhcoptics import LHCOptics

LHCOptics.set_repository("2024")

lhc=LHCOptics.from_madxfile("job_model.madx",model='xsuite')
lhc.model.to_json("lhc.json")
lhc=LHCOptics.from_madxfile("job_model.madx",model='xsuite',sliced=True)
lhc.model.to_json("lhc_sliced.json")

import os
os.system("cp *.json ../data/")

