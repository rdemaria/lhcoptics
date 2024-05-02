from lhcoptics import LHCOptics

eor=LHCOptics.from_json("data/opt_inj.json",xsuite_model="data/lhc.json")
eor.name="eor"
eor.params["match_inj"]=False
eor.set_circuits_from_json("data/lhccircuits.json")
eor.knobs_off()
eor.model.p0c=6.8e12
eor.set_params()
eor.update()

#IR2 (from hl)
hlopt=LHCOptics.from_madxfile("opt_ramp_2000_1500.madx")
eor.ir2.update(hlopt.ir2,verbose=True,knobs=False).update_model()
eor.ir2.match().solve()

#IR4 (from hl)
eor.ir4.update(hlopt.ir4,verbose=True).update_model()
eor.ir4.match().solve()

#IR8 (rematch but need to optimize beta)
eor.params["p0c"]=7e12
eor.ir8.strengths["ktqx1.l8"]=0
eor.ir8.strengths["ktqx1.r8"]=0
eor.ir8.strengths["ktqx2.l8"]=0
eor.ir8.strengths["ktqx2.r8"]=0
eor.ir8.update_model()
eor.ir8.match().disable(vary_name="ktqx.*").solve()
eor.params["p0c"]=6.8e12

#IR7 (from bjorn)
eor.ir7.update_model(src=eor.ir7.knobsRematched12c6b)
opt=eor.ir7.match()
opt.disable(target=["ip_","mu"])
opt.solve()

#IR3 (from bjorn)
eor.ir3.update_model(src=eor.ir3.knobsRematched13b_mu)
opt=eor.ir3.match()
opt.disable(target=["ip_","mu"])
opt.solve()


#IR1 (from 2024, check ATS phases)
ats=LHCOptics.from_madxfile("acc-models-lhc/strengths/ATS_Nominal/2024/ats_250cm.madx")
eor.ir1.update(ats.ir1,knobs=False,verbose=True).update_model()
eor.ir1.set_betastar(2.5)
opt=eor.ir1.match()
opt.solve()

#IR5 (from 2024, check ATS phases)
ats=LHCOptics.from_madxfile("acc-models-lhc/strengths/ATS_Nominal/2024/ats_250cm.madx")
eor.ir5.update(ats.ir5,knobs=False,verbose=True).update_model()
eor.ir5.set_betastar(2.5)
opt=eor.ir5.match()
opt.solve()

#IR6 (small adjust)
eor.ir6.match().solve()

eor.update()
eor.set_params()
eor.check()
eor.to_json("eor_detuned.json")

eor.match_tune()
eor.update()
eor.check()
eor.to_json("eor_tuned.json")

eor2=LHCOptics.from_json("eor_tuned.json",xsuite_model="data/lhc.json")
eor2.check()
