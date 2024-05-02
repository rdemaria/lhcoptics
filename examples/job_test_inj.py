from lhcoptics import LHCOptics

inja=LHCOptics.from_json("data/opt_inj.json",xsuite_model="data/lhc.json")
injb=LHCOptics.from_json("inj_newphase.json",xsuite_model="data/lhc.json")

tw1a,tw2a=inja.twiss()
tw1b,tw2b=injb.twiss()

clf()
plot(tw1b.s, tw1b.betx,label="new")
plot(tw1a.s, tw1a.betx,label="old")
legend()

clf()
plot(tw1b.s, tw1b.bety,label="new")
plot(tw1a.s, tw1a.bety,label="old")
axhline(180,color='k')
legend()


clf()
plot(tw2b.s, tw2b.betx,label="new")
plot(tw2a.s, tw2a.betx,label="old")
axhline(180,color='k')
legend()

clf()
plot(tw2b.s, tw2b.bety,label="new")
plot(tw2a.s, tw2a.bety,label="old")
axhline(180,color='k')
legend()














