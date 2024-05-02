from lhcoptics import LHCOptics
import numpy as np

inj = LHCOptics.from_json("inj_newphase.json")
inj.params["match_inj"] = False
eor = LHCOptics.from_json("eor_tuned.json")

ramp = inj.to_table(eor)
ramp["p0c"]

cur = inj.copy()
cur.params["match_inj"] = False
cur.set_xsuite_model("data/lhc.json")
cur.set_circuits_from_json("data/lhccircuits.json")
cur.knobs_off()
cur.check()

cur.update(ramp.interp(0.1)).update_model().check()
cur.update(ramp.interp(0.0)).update_model().check()
cur.update(ramp.interp(1.0)).update_model().check()


# overall matching

optt = inj.to_table().clear()
for n in np.linspace(0, 1, 11):
    cur.update(ramp.interp(n)).update_model()
    [arc.match().solve() for arc in cur.arcs]
    cur.ir1.match().disable(vary_name="kq10.r1b1").solve()
    cur.ir2.match().solve()
    cur.ir3.match().solve()
    cur.ir4.match().solve()
    cur.ir5.match().solve()
    cur.ir6.match().solve()
    cur.ir7.match().solve()
    cur.ir8.match().solve()
    cur.check()
    cur.update()
    cur.to_json(f"opt_{n}.json")
    optt.append(cur.copy())

optt.plot_quads()


# refine

ir7t = optt.ir7  # only first time
ir7t = ir7t2

ir7t2 = inj.ir7.to_table().clear()
for n in np.linspace(0, 10, 11):
    cur.update(optt.interp(n))
    cur.ir7.update_strengths(ir7t.interp(n, order=4))
    cur.update_model()
    # cur.check()
    cur.ir7.match().solve()
    cur.check()
    cur.update()
    cur.to_json(f"ramp_{n}.json")
    ir7t2.append(cur.ir7.copy())
    print(n)

ir7t2.plot_quads()


# IR1
ir1t = inj.ir1.to_table().clear()
for n in np.linspace(0, 1, 11):
    cur.update(ramp.interp(n)).update_model()
    print(cur.get_phase_arcs()["muxa12b1"])
    print(cur.get_phase_arcs()["muxa81b1"])
    cur.a12.match()
    cur.a81.match()
    print(cur.get_phase_arcs()["muxa12b1"])
    print(cur.get_phase_arcs()["muxa81b1"])
    cur.ir1.match().disable(vary_name="kq10.r1b1").solve()
    cur.ir1.match().solve()
    cur.update()
    # cur.ir1.plot(1)
    # plt.savefig('ir1b1.png')
    # cur.ir1.plot(2)
    # plt.savefig('ir1b2.png')
    ir1t.append(cur.ir1.copy())
    print(cur.ir1["betxip1b1"])

ir1t.plot_quads(xaxis="betxip1b1")


# IR2
ir2t = inj.ir2.to_table().clear()
for n in np.linspace(0, 1, 11):
    cur.update(ramp.interp(n)).update_model()
    cur.a12.match()
    cur.a23.match()
    cur.ir2.match().solve()
    cur.update()
    ir2t.append(cur.ir2.copy())
    print(cur.ir2["kqx.l2"])

ir2t.plot_quads(xaxis="kqx.l2")


# IR3
ir3t = inj.ir3.to_table().clear()
for n in np.linspace(0, 1, 11):
    cur.update(ramp.interp(n)).update_model()
    cur.ir3.match().solve()
    cur.update()
    ir3t.append(cur.ir3.copy())
    print(n)

ir3t.plot_quads()


# IR4
ir4t = inj.ir4.to_table().clear()
for n in np.linspace(0, 1, 11):
    cur.update(ramp.interp(n)).update_model()
    cur.ir4.match().solve()
    cur.update()
    ir4t.append(cur.ir4.copy())
    print(n)

ir4t.plot_quads()

# IR5
ir5t = inj.ir5.to_table().clear()
for n in np.linspace(0, 1, 11):
    cur.update(ramp.interp(n)).update_model()
    cur.ir5.match().solve()
    cur.update()
    ir5t.append(cur.ir5.copy())
    print(n)

ir5t.plot_quads()

# IR6
ir6t = inj.ir6.to_table().clear()
for n in np.linspace(0, 1, 11):
    cur.update(ramp.interp(n)).update_model()
    cur.ir6.match().solve()
    cur.update()
    ir6t.append(cur.ir6.copy())
    print(n)

ir6t.plot_quads()

# IR7
ir7t = inj.ir7.to_table().clear()
for n in np.linspace(0, 1, 11):
    cur.update(ramp.interp(n)).update_model()
    cur.ir7.match().solve()
    cur.update()
    ir7t.append(cur.ir7.copy())
    print(n)

ir7t.plot_quads()


# IR8
ir8t = inj.ir8.to_table().clear()
for n in np.linspace(0, 1, 11):
    cur.update(ramp.interp(n)).update_model()
    cur.ir8.match().solve()
    cur.update()
    ir8t.append(cur.ir8.copy())
    print(n)

ir8t.plot_quads()
