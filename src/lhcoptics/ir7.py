import xtrack as xt
import xdeps as xd

import numpy as np

from .irs import LHCIR


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

    def get_params_from_twiss(self, tw1, tw2):
        params = LHCIR.get_params_from_twiss(self, tw1, tw2)
        for col_name in self.collimators:
            if "b1" in col_name:
                params[f"betx_{col_name}"] = tw1["betx", col_name]
                params[f"bety_{col_name}"] = tw1["bety", col_name]
            else:
                params[f"betx_{col_name}"] = tw2["betx", col_name]
                params[f"bety_{col_name}"] = tw2["bety", col_name]
        return params

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

    def update_from_model(self):
        for beam, tw in enumerate(self.twiss):
            self.params[f"betxip7b{beam+1}"] = tw["betxip"]
            self.params[f"betyip7b{beam+1}"] = tw["betyip"]
        self.params = {
            "betxip7b1": 0.8,
        }

    def match(self):
        if self.init_left is None or self.init_right is None:
            self.set_init()
        if len(self.params) == 0:
            self.params = self.get_params()
        if self.parent.model is None:
            raise ValueError("Model not set for {self)")
        lhc = self.parent.model.multiline
        if lhc.b1.tracker is None:
            lhc.b1.build_tracker()
        if lhc.b2.tracker is None:
            lhc.b2.build_tracker()
        self.action_sp1 = SinglePassDispersion(
            lhc.b1, ele_start="tcp.d6l7.b1", ele_stop="tcspm.6r7.b1"
        )
        self.action_sp2 = SinglePassDispersion(
            lhc.b2, ele_start="tcp.d6r7.b2", ele_stop="tcspm.6l7.b2"
        )
        dxb1 = self.action_sp1.run()["dx"]
        dxb2 = self.action_sp2.run()["dx"]

        inits = [self.init_left[1], self.init_left[2]]
        starts = [self.startb12[1], self.startb12[2]]
        ends = [self.endb12[1], self.endb12[2]]
        lines = ["b1", "b2"]

        std_targets = LHCIR.get_match_targets(self)
        sp_targets = [
            xt.Target(
                action=self.action_sp1,
                tar="dx",
                value=dxb1,
                tol=1e-2,
                tag="spdx",
            ),
            xt.Target(
                action=self.action_sp2,
                tar="dx",
                value=dxb2,
                tol=1e-2,
                tag="spdx",
            ),
        ]
        colltargets = []
        for cn in self.collimators:
            line = "b1" if "b1" in cn else "b2"
            for tt in ["betx", "bety"]:
                vv = self.params[f"{tt}_{cn}"]
                tt = xt.Target(
                    tt,
                    xt.GreaterThan(vv),
                    line=line,
                    at=cn,
                    tol=1e-1,
                    tag="coll",
                )
                colltargets.append(tt)

        varylst = [
            xt.Vary(kk, limits=[-0.1, 0.1], step=1e-9)
            for kk in self.quads
        ]

        opt = lhc.match(
            solve=False,
            default_tol={None: 5e-8},
            solver_options=dict(max_rel_penalty_increase=2.0),
            lines=lines,
            start=starts,
            end=ends,
            init=inits,
            targets=(std_targets + sp_targets + colltargets),
            vary=varylst,
        )
        opt.disable_targets(tag="coll")
        opt.disable_targets(tag="spdx")
        self.opt = opt
        return opt
