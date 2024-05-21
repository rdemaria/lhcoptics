from .irs import LHCIR
from .model_xsuite import SinglePassDispersion


class LHCIR3(LHCIR):
    name = "ir3"

    colls_ir3b1 = [
        "tcp.6l3.b1",
        "tcsg.5l3.b1",
        "tcsg.4r3.b1",
        "tcsg.a5r3.b1",
        "tcsg.b5r3.b1",
        "tcla.a5r3.b1",
        "tcla.b5r3.b1",
        "tcla.6r3.b1",
        "tcla.7r3.b1",
    ]
    colls_ir3b2 = [
        "tcp.6r3.b2",
        "tcsg.5r3.b2",
        "tcsg.4l3.b2",
        "tcsg.a5l3.b2",
        "tcsg.b5l3.b2",
        "tcla.a5l3.b2",
        "tcla.b5l3.b2",
        "tcla.6l3.b2",
        "tcla.7l3.b2",
    ]

    collimators = colls_ir3b1 + colls_ir3b2

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

    def get_params_from_twiss(self, tw1, tw2):
        params = LHCIR.get_params_from_twiss(self, tw1, tw2)
        for col_name in self.collimators:
            if "b1" in col_name:
                params[f"betx_{col_name}"] = tw1["betx", col_name]
                params[f"bety_{col_name}"] = tw1["bety", col_name]
            else:
                params[f"betx_{col_name}"] = tw2["betx", col_name]
                params[f"bety_{col_name}"] = tw2["bety", col_name]
        if self.parent.model is not None:
            self.action_sp1 = SinglePassDispersion(
                self.parent.model.b1,
                ele_start="tcp.d6l7.b1",
                ele_stop="tcspm.6r7.b1",
            )
            self.action_sp2 = SinglePassDispersion(
                self.parent.model.b2,
                ele_start="tcp.d6r7.b2",
                ele_stop="tcspm.6l7.b2",
            )
            params["dx_tcp_tcsb1"] = self.action_sp1.run()["dx"]
            params["dx_tcp_tcsb2"] = self.action_sp2.run()["dx"]
        return params


"""
    def match(self):
        lhc=self.multiline
        margin=0
        tw1, tw2 = lhc.twiss()
        bety0=tw1.lhcb1.rows['tcsg.5l3.b1'].bety[0]
        bety1=tw1.lhcb1.rows['tcsg.4r3.b1'].bety[0]
        bety2=tw1.lhcb1.rows['tcsg.a5r3.b1'].bety[0]
        bety3=tw1.lhcb1.rows['tcsg.b5r3.b1'].bety[0]
        bety4=tw1.lhcb2.rows['tcsg.5r3.b2'].bety[0]
        bety5=tw1.lhcb2.rows['tcsg.4l3.b2'].bety[0]
        bety6=tw1.lhcb2.rows['tcsg.a5l3.b2'].bety[0]
        bety7=tw1.lhcb2.rows['tcsg.b5l3.b2'].bety[0]
        ir3targets=[]
        for coll in colls_ir3b1:
            ir3targets.append(xt.TargetSet(['betx','dx'],tol=1e-1,value=tw1,line='lhcb1',at=coll))
        for coll in colls_ir3b2:
            ir3targets.append(xt.TargetSet(['betx','dx'],tol=1e-1,value=tw2,line='lhcb2',at=coll))
        opt3 = lhc.match(solve=False,
                        default_tol={None: 5e-8},
                        twiss_init='preserve_start',
                        table_for_twiss_init=(tw1,tw2),
                        ele_start=('s.ds.l3.b1','s.ds.l3.b2'),
                        ele_stop=('e.ds.r3.b1','e.ds.r3.b2'),
                        targets=[
                            xt.TargetSet(['alfx','alfy','betx','bety','dx','dpx'],value=tw1,line='lhcb1',at=xt.END),
                            xt.TargetSet(['alfx','alfy','betx','bety','dx','dpx'],value=tw2,line='lhcb2',at=xt.END),
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
"""
