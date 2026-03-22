from lhcoptics import LHCCircuits

cir=LHCCircuits.from_json('https://acc-models.web.cern.ch/acc-models/lhc/hl19/xsuite/lhccircuits.json')
print(cir.madname['kco.a23b2'].get_klimits(pc=6800e9))