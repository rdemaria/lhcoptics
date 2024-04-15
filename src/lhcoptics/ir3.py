import numpy as np
import xtrack as xt

from .irs import LHCIR

class LHCIR3(LHCIR):
    name = "ir3"

    colls_ir3b1=['tcp.6l3.b1','tcsg.5l3.b1','tcsg.4r3.b1','tcsg.a5r3.b1','tcsg.b5r3.b1','tcla.a5r3.b1','tcla.b5r3.b1','tcla.6r3.b1','tcla.7r3.b1']
    colls_ir3b2=['tcp.6r3.b2','tcsg.5r3.b2','tcsg.4l3.b2','tcsg.a5l3.b2','tcsg.b5l3.b2','tcla.a5l3.b2','tcla.b5l3.b2','tcla.6l3.b2','tcla.7l3.b2']


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