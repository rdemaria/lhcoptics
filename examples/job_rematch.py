"""
Flat top optics: [mon]
    STD Match IRs [x]
    STD Match Arcs [x]
    Rematch IR3, IR7
    IR2, IR8 from 2016
    IR4 from HL
    Rematch phase
    Save optics

Inj Optics: [tue]
    Get phase with phase knobs
    Compute RDT w and w/o
    Set new phases to inj
    Check apertures

Make squeeze: [wed]
    make squeeze:  k(idx)
    make t[idx] and ksmooth idx(t)
    from idx(t) measure current


kqx1.l1     = kqx.l1+ktqx1.l1;
kqx2.l1     = -kqx.l1-ktqx2.l1;
kqx3.l1     = kqx.l1;
kqx.l1     := kqx3.l1;
ktqx2.l1   := -kqx3.l1 -kqx2.l1;
ktqx1.l1   := kqx3.l1-ktqx1.l1;
"""

from lhcoptics import LHCOptics

inj = LHCOptics.from_json("data/opt_inj.json", xsuite_model="data/lhc.json")
inj.set_circuits_from_json("data/lhccircuits.json")
inj.model["phase_change.b1"] = 0
inj.model["phase_change.b2"] = 0
inj.set_bumps_off()
inj.update()

# IR1
mtc = inj.ir1.match()
mtc.vary_status()
mtc.target_status()

# ARC
mtc = inj.a12.match()
mtc.vary_status()
mtc.target_status()
