from lhcoptics import LHCOptics

LHCOptics.set_repository("2024")

inj=LHCOptics.from_madxfile("model.madx",model='xsuite')
inj.model.to_json("lhc.json")
inj.to_json("opt_inj.json")
inj=LHCOptics.from_madxfile("model.madx",model='xsuite',sliced=True)
inj.model.to_json("lhc_sliced.json")


import os
os.system("mv *.json ../data/")

