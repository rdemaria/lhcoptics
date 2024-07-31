import xtrack as xt

from .irs import LHCIR
from .model_xsuite import SinglePassDispersion


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

    def match(
        self,
        kmin_marg=0.0,
        kmax_marg=0.0,
        collimation=False,
        b1=True,
        b2=True,
        common=True,
        hold_init=False,
    ):
        if self.parent.model is None:
            raise ValueError("Model not set for {self)")
        if self.parent.circuits is None:
            raise ValueError("Circuits not set for {self)")
        if not hold_init:
            self.set_init()
        if len(self.params) == 0:
            self.set_params()
        lhc = self.parent.model.multiline
        if lhc.b1.tracker is None:
            lhc.b1.build_tracker()
        if lhc.b2.tracker is None:
            lhc.b2.build_tracker()

        if collimation:
            self.action_sp1 = SinglePassDispersion(
                lhc.b1, ele_start="tcp.d6l7.b1", ele_stop="tcspm.6r7.b1"
            )
            self.action_sp2 = SinglePassDispersion(
                lhc.b2, ele_start="tcp.d6r7.b2", ele_stop="tcspm.6l7.b2"
            )
            dxb1 = self.action_sp1.run()["dx"]
            dxb2 = self.action_sp2.run()["dx"]

        inits = [self.init_left[1], self.init_left[2]]
        lines = ["b1", "b2"]

        targets = LHCIR.get_match_targets(self)
        if collimation:
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
            targets += sp_targets + colltargets

        varylst = LHCIR.get_match_vary(
            self,
            b1=b1,
            b2=b2,
            common=common,
            dkmin=kmin_marg,
            dkmax=kmax_marg,
        )

        opt = lhc.match(
            solve=False,
            default_tol={None: 5e-8},
            solver_options=dict(max_rel_penalty_increase=2.0),
            lines=lines,
            start=self.startb12,
            end=self.endb12,
            init=inits,
            targets=targets,
            vary=varylst,
            check_limits=False,
            strengths=False,
        )
        opt.disable(target="coll")
        opt.disable(target="spdx")
        opt.disable(target="mu.*_l")
        self.optmizer = opt
        return opt
