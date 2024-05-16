from lhcoptics import LHCOptics

inj = LHCOptics.from_json("data/opt_inj.json", xsuite_model="data/lhc.json")
inj.ir7.get_params()


inj.set_circuits_from_json("data/lhccircuits.json")

mtc = inj.ir7.match()
mtc.vary_status()
mtc.target_status()
mtc.step(10)
inj.ir7.to_json("ir7_inj.json")

inj.ir7.update_model(src=inj.ir7.knobsRematched12c6b)
inj.ir7.match().vary_status()
inj.ir7.match().target_status()
opt = inj.ir7.match()
opt.disable(target="ipcond")
opt.disable(target="phase")
opt.step(20)
inj.ir7.plot()

inj.ir7.update_strengths().set_params().to_json("ir7_imp.json")

ir7t = inj.ir7.copy("ir7_inj.json").to_table(inj.ir7.copy("ir7_imp.json"))

out = []
for n in np.linspace(0, 1, 10):
    inj.ir7.update_model(ir7t.interp(n))
    inj.ir7.params["n"] = n
    opt = inj.ir7.match()
    opt.disable(target="ipcond")
    opt.disable(target="phase")
    opt.step(20)
    print(n, opt.log().penalty[-1])
    out.append(inj.ir7.update_strengths().update_params().copy())

ir7t2 = inj.ir7.to_table(*out)
