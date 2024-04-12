from .irs import LHCIR


class LHCIR7(LHCIR):
    name = "ir7"

    collimators = [
        "tcp.c6l7.b1",
        "tcp.d6l7.b1",
        "tcp.c6r7.b2",
        "tcp.d6r7.b2",
        "tcsg.a4l7.b1",
        "tcsg.a4l7.b1",
        "tcsg.a4r7.b2",
        "tcsg.a4r7.b2",
        "tcsg.d5r7.b1",
        "tcsg.d5r7.b1",
        "tcsg.d5l7.b2",
        "tcsg.d5l7.b2",
        "tcspm.6r7.b1",
        "tcspm.6r7.b1",
        "tcspm.6l7.b2",
        "tcspm.6l7.b2",
        "tcla.d6r7.b1",
        "tcla.d6r7.b1",
        "tcla.d6l7.b2",
        "tcla.d6l7.b2",
        "tcsg.a5l7.b1",
        "tcsg.a5l7.b1",
        "tcsg.a5r7.b2",
        "tcsg.a5r7.b2",
    ]

    def update_from_model(self):
        for beam, tw in enumerate(self.twiss):
            self.params[f"betxip7b{beam+1}"] = tw["betxip"]
            self.params[f"betyip7b{beam+1}"] = tw["betyip"]
        self.params = {
            "betxip7b1": 0.8,
        }


"""
class SinglePassDispersion(xd.Action):
    def __init__(self, line, ele_start, ele_stop, backtrack=False, delta=1e-3):
        self.line = line
        self.ele_start = ele_start
        self.ele_stop = ele_stop
        self.delta = delta
        self.backtrack = backtrack
        self._pp = line.build_particles(delta=delta)

    def run(self):
        for nn in ["x", "px", "y", "py", "zeta", "delta", "at_element"]:
            setattr(self._pp, nn, 0)
        self._pp.delta = self.delta
        self.line.track(
            self._pp,
            ele_start=self.ele_start,
            ele_stop=self.ele_stop,
            backtrack=self.backtrack,
        )
        return {
            "d" + nn: getattr(self._pp, nn)[0] / self.delta
            for nn in ["x", "px", "y", "py"]
        }


act_sp1 = SinglePassDispersion(
    lhc.lhcb1, ele_start="tcp.d6l7.b1", ele_stop="tcspm.6r7.b1"
)
act_sp2 = SinglePassDispersion(
    lhc.lhcb2, ele_start="tcp.d6r7.b2", ele_stop="tcspm.6l7.b2", backtrack=False
)


dxb1=act_sp1.run()['dx'] 
dxb2=act_sp2.run()['dx'] 
opt3 = lhc.match(solve=False,
                default_tol={None: 5e-8},
                twiss_init='preserve_start',
                table_for_twiss_init=(tw.lhcb1,tw.lhcb2),
                ele_start=('s.ds.l3.b1','s.ds.l3.b2'),
                ele_stop=('e.ds.r3.b1','e.ds.r3.b2'),
                targets=[
                    xt.TargetSet(['alfx','alfy','betx','bety','dx','dpx'],value=tw.lhcb1,line='lhcb1',at=xt.END),
                    xt.TargetSet(['alfx','alfy','betx','bety','dx','dpx'],value=tw.lhcb2,line='lhcb2',at=xt.END),
                    xt.Target('bety',xt.LessThan((1+margin)*bety0),line='lhcb1',at='tcsg.5l3.b1',tol=1e-1,tag='tcsg0b1'),
                    xt.Target('bety',xt.LessThan((1+margin)*bety1),line='lhcb1',at='tcsg.4r3.b1',tol=1e-1,tag='tcsg1b1'),
                    xt.Target('bety',xt.LessThan((1+margin)*bety2),line='lhcb1',at='tcsg.a5r3.b1',tol=1e-1,tag='tcsg2b1'),
                    xt.Target('bety',xt.LessThan((1+margin)*bety3),line='lhcb1',at='tcsg.b5r3.b1',tol=1e-1,tag='tcsg3b1'),

                    xt.Target('bety',xt.LessThan((1+margin)*bety4),line='lhcb2',at='tcsg.5r3.b2',tol=1e-1,tag='tcsg0b2'),
                    xt.Target('bety',xt.LessThan((1+margin)*bety5),line='lhcb2',at='tcsg.4l3.b2',tol=1e-1,tag='tcsg1b2'),
                    xt.Target('bety',xt.LessThan((1+margin)*bety6),line='lhcb2',at='tcsg.a5l3.b2',tol=1e-1,tag='tcsg2b2'),
                    xt.Target('bety',xt.LessThan((1+margin)*bety7),line='lhcb2',at='tcsg.b5l3.b2',tol=1e-1,tag='tcsg3b2'),
                ]+ir3targets+[
                    xt.Target(action=act_pa3, tar='mux', value=nomMux, tol=1e-3, tag='mu'),
                    xt.Target(action=act_pa4, tar='muy', value=nomMuy, tol=1e-3, tag='mu'),
                ],
                vary=(xt.VaryList(list(strDictIR3.keys())))
)
opt3.assert_within_tol=False


bety0=tw1.lhcb1.rows['tcsg.5l3.b1'].bety[0]
bety1=tw1.lhcb1.rows['tcsg.4r3.b1'].bety[0]
bety2=tw1.lhcb1.rows['tcsg.a5r3.b1'].bety[0]
bety3=tw1.lhcb1.rows['tcsg.b5r3.b1'].bety[0]
bety4=tw1.lhcb2.rows['tcsg.5r3.b2'].bety[0]
bety5=tw1.lhcb2.rows['tcsg.4l3.b2'].bety[0]
bety6=tw1.lhcb2.rows['tcsg.a5l3.b2'].bety[0]
bety7=tw1.lhcb2.rows['tcsg.b5l3.b2'].bety[0]

opt7 = lhc.match(solve=False,
                default_tol={None: 5e-8},
                solver_options=dict(max_rel_penalty_increase=2.),
                twiss_init='preserve_start',
                table_for_twiss_init=(tw.lhcb1,tw.lhcb2),
                ele_start=('s.ds.l7.b1','s.ds.l7.b2'),
                ele_stop=('e.ds.r7.b1','e.ds.r7.b2'),
                targets=[
                    xt.TargetSet(['alfx','alfy','betx','bety','dx','dpx'],value=tw.lhcb1,line='lhcb1',at=xt.END),
                    xt.TargetSet(['alfx','alfy','betx','bety','dx','dpx'],value=tw.lhcb2,line='lhcb2',at=xt.END),
                    xt.Target(action=act_sp1, tar='dx', value=dxb1, tol=1e-2, tag='dx'),
                    xt.Target(action=act_sp2, tar='dx', value=dxb2, tol=1e-2, tag='dx'),
                    xt.Target('betx',xt.GreaterThan((1-margin)*betx1),line='lhcb1',at='tcp.c6l7.b1',tol=1e-1,tag='tcp'),
                    xt.Target('bety',xt.GreaterThan((1-margin)*bety1),line='lhcb1',at='tcp.d6l7.b1',tol=1e-1,tag='tcp'),
                    xt.Target('betx',xt.GreaterThan((1-margin)*betx2),line='lhcb2',at='tcp.c6r7.b2',tol=1e-1,tag='tcp'),
                    xt.Target('bety',xt.GreaterThan((1-margin)*bety2),line='lhcb2',at='tcp.d6r7.b2',tol=1e-1,tag='tcp'),

                    xt.Target('betx',xt.LessThan((1+margin)*betx3),line='lhcb1',at='tcsg.a4l7.b1',tol=1e-1,tag='tcsga4'),
                    xt.Target('bety',xt.LessThan((1+margin)*bety3),line='lhcb1',at='tcsg.a4l7.b1',tol=1e-1,tag='tcsga4'),
                    xt.Target('betx',xt.LessThan((1+margin)*betx4),line='lhcb2',at='tcsg.a4r7.b2',tol=1e-1,tag='tcsga4'),
                    xt.Target('bety',xt.LessThan((1+margin)*bety4),line='lhcb2',at='tcsg.a4r7.b2',tol=1e-1,tag='tcsga4'),

                    xt.Target('betx',xt.LessThan((1+margin)*betx5),line='lhcb1',at='tcsg.d5r7.b1',tol=1e-1,tag='tcsgd'),
                    xt.Target('bety',xt.LessThan((1+margin)*bety5),line='lhcb1',at='tcsg.d5r7.b1',tol=1e-1,tag='tcsgd'),
                    xt.Target('betx',xt.LessThan((1+margin)*betx6),line='lhcb2',at='tcsg.d5l7.b2',tol=1e-1,tag='tcsgd'),
                    xt.Target('bety',xt.LessThan((1+margin)*bety6),line='lhcb2',at='tcsg.d5l7.b2',tol=1e-1,tag='tcsgd'),

                    xt.Target('betx',xt.LessThan((1+margin)*betx7),line='lhcb1',at='tcspm.6r7.b1',tol=1e-1,tag='tcspm'),
                    xt.Target('bety',xt.LessThan((1+margin)*bety7),line='lhcb1',at='tcspm.6r7.b1',tol=1e-1,tag='tcspm'),
                    xt.Target('betx',xt.LessThan((1+margin)*betx8),line='lhcb2',at='tcspm.6l7.b2',tol=1e-1,tag='tcspm'),
                    xt.Target('bety',xt.LessThan((1+margin)*bety8),line='lhcb2',at='tcspm.6l7.b2',tol=1e-1,tag='tcspm'),

                    xt.Target('betx',xt.LessThan((1+margin)*betx9),line='lhcb1',at='tcla.d6r7.b1',tol=1e-1,tag='tcla'),
                    xt.Target('bety',xt.LessThan((1+margin)*bety9),line='lhcb1',at='tcla.d6r7.b1',tol=1e-1,tag='tcla'),
                    xt.Target('betx',xt.LessThan((1+margin)*betxA),line='lhcb2',at='tcla.d6l7.b2',tol=1e-1,tag='tcla'),
                    xt.Target('bety',xt.LessThan((1+margin)*betyA),line='lhcb2',at='tcla.d6l7.b2',tol=1e-1,tag='tcla'),

                    xt.Target('betx',xt.LessThan((1+margin)*betxB),line='lhcb1',at='tcsg.a5l7.b1',tol=1e-1,tag='tcsga5'),
                    xt.Target('bety',xt.LessThan((1+margin)*betyB),line='lhcb1',at='tcsg.a5l7.b1',tol=1e-1,tag='tcsga5'),
                    xt.Target('betx',xt.LessThan((1+margin)*betxC),line='lhcb2',at='tcsg.a5r7.b2',tol=1e-1,tag='tcsga5'),
                    xt.Target('bety',xt.LessThan((1+margin)*betyC),line='lhcb2',at='tcsg.a5r7.b2',tol=1e-1,tag='tcsga5'),
                ],
                vary=(xt.VaryList(list(strDictIR7.keys())))
)


colls_ir3b1=['tcp.6l3.b1','tcsg.5l3.b1','tcsg.4r3.b1','tcsg.a5r3.b1','tcsg.b5r3.b1','tcla.a5r3.b1','tcla.b5r3.b1','tcla.6r3.b1','tcla.7r3.b1']
colls_ir3b2=['tcp.6r3.b2','tcsg.5r3.b2','tcsg.4l3.b2','tcsg.a5l3.b2','tcsg.b5l3.b2','tcla.a5l3.b2','tcla.b5l3.b2','tcla.6l3.b2','tcla.7l3.b2']

ir3targets=[]
for coll in colls_ir3b1:
    ir3targets.append(xt.TargetSet(['betx','dx'],tol=1e-1,value=tw.lhcb1,line='lhcb1',at=coll))
for coll in colls_ir3b2:
    ir3targets.append(xt.TargetSet(['betx','dx'],tol=1e-1,value=tw.lhcb2,line='lhcb2',at=coll))


knobsRematched12c6b = {
    "kqt4.l7": 0.0012257364160585084,
    "kqt4.r7": 0.0012659632628095638,
    "kqt13.l7b1": -0.0048823483573787445,
    "kqt12.l7b1": -0.004882279788343516,
    "kqtl11.l7b1": 0.0027739663492968103,
    "kqtl10.l7b1": 0.004623538857746193,
    "kqtl9.l7b1": -0.003372747954072591,
    "kqtl8.l7b1": -0.0023127417813640786,
    "kqtl7.l7b1": -0.002011344510772721,
    "kq6.l7b1": 0.0031173363410593766,
    "kq6.r7b1": -0.0031388056161611565,
    "kqtl7.r7b1": 0.0009532375359442739,
    "kqtl8.r7b1": 0.002688438505728887,
    "kqtl9.r7b1": 0.0033416607916765947,
    "kqtl10.r7b1": -0.003461273410884878,
    "kqtl11.r7b1": 0.0010531054411466265,
    "kqt12.r7b1": -0.0027831205556483702,
    "kqt13.r7b1": -0.0013509460856456692,
    "kqt13.l7b2": -0.004192310485204978,
    "kqt12.l7b2": -0.0035271197718106688,
    "kqtl11.l7b2": 0.0008993274235722462,
    "kqtl10.l7b2": -0.0035044843946580337,
    "kqtl9.l7b2": 0.003295485018957867,
    "kqtl8.l7b2": 0.002429071850457167,
    "kqtl7.l7b2": 0.0008310840304967491,
    "kq6.l7b2": -0.0031817725498278727,
    "kq6.r7b2": 0.003183554427942885,
    "kqtl7.r7b2": -0.0012886165853725183,
    "kqtl8.r7b2": -0.0037917967174795034,
    "kqtl9.r7b2": -0.0033703081873609005,
    "kqtl10.r7b2": 0.0049711605825101994,
    "kqtl11.r7b2": 0.002278252114016244,
    "kqt12.r7b2": -0.0048808187874553495,
    "kqt13.r7b2": -0.0048815559298144,
    "kq4.lr7": 0.0011653779946877393,
    "kq5.lr7": -0.001202569087048791,
}

knobsRematched13b_mu = {
    "kqt4.l3": 0.0006887129999999986,
    "kqt4.r3": 0.000688713,
    "kqt5.l3": 0.000972084,
    "kqt5.r3": 0.000972084,
    "kqt13.l3b1": -0.002328955907392481,
    "kqt12.l3b1": 0.002822813556121194,
    "kqtl11.l3b1": 0.0012986138594000976,
    "kqtl10.l3b1": 0.0010616412957247959,
    "kqtl9.l3b1": -0.005223865183101024,
    "kqtl8.l3b1": 0.00033781692792629684,
    "kqtl7.l3b1": -0.000876435629840312,
    "kq6.l3b1": 0.0025894410128743917,
    "kq6.r3b1": -0.002412918519504643,
    "kqtl7.r3b1": 0.0022028677895794004,
    "kqtl8.r3b1": 0.0035691450329306527,
    "kqtl9.r3b1": -7.37775306738355e-05,
    "kqtl10.r3b1": 0.004022882019207013,
    "kqtl11.r3b1": -0.0030302762162364503,
    "kqt12.r3b1": -0.005138992845888184,
    "kqt13.r3b1": -0.001896775114412516,
    "kqt13.l3b2": -0.0025197838172713494,
    "kqt12.l3b2": -0.003785001043390164,
    "kqtl11.l3b2": -0.0032679415485541703,
    "kqtl10.l3b2": 0.004726996198081681,
    "kqtl9.l3b2": -0.0006237278666374633,
    "kqtl8.l3b2": 0.0038112997328762573,
    "kqtl7.l3b2": 0.0005221529209823068,
    "kq6.l3b2": -0.002467855997059401,
    "kq6.r3b2": 0.0025687299038460276,
    "kqtl7.r3b2": 0.0007580546568790491,
    "kqtl8.r3b2": -0.0007870443115947539,
    "kqtl9.r3b2": -0.004254750086155878,
    "kqtl10.r3b2": 0.00041179336225102066,
    "kqtl11.r3b2": 0.0006593584215004978,
    "kqt12.r3b2": -9.48176531256308e-05,
    "kqt13.r3b2": -0.005098976136482916,
}


"""
