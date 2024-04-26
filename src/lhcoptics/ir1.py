from .irs import LHCIR


class LHCIR1(LHCIR):
    name = "ir1"
    knobs = [
        "on_x1_h",
        "on_sep1_h",
        "on_x1_v",
        "on_sep1_v",
        "on_xip1b1",
        "on_xip1b2",
        "on_oh1",
        "on_yip1b2",
        "on_yip1b1",
        "on_ov1",
    ]

    def match(
        self, kmin_marg=0.0, kmax_marg=0.0, b1=True, b2=True, common=True
    ):
        if self.parent.model is None:
            raise ValueError("Model not set for {self)")
        if self.parent.circuits is None:
            raise ValueError("Circuits not set for {self)")
        if self.init_left is None or self.init_right is None:
            self.set_init()
        if len(self.params) == 0:
            self.params = self.get_params()
        lhc = self.parent.model.multiline
        if lhc.b1.tracker is None:
            lhc.b1.build_tracker()
        if lhc.b2.tracker is None:
            lhc.b2.build_tracker()

        inits = [self.init_left[1], self.init_left[2]]
        starts = [self.startb12[1], self.startb12[2]]
        ends = [self.endb12[1], self.endb12[2]]
        lines = ["b1", "b2"]

        targets = LHCIR.get_match_targets(self, b1=b1, b2=b2)
        varylst = LHCIR.get_match_vary(
            self,
            b1=b1,
            b2=b2,
            common=common,
            kmin_marg=kmin_marg,
            kmax_marg=kmax_marg,
        )

        opt = lhc.match(
            solve=False,
            default_tol={None: 5e-8},
            solver_options=dict(max_rel_penalty_increase=2.0),
            lines=lines,
            start=starts,
            end=ends,
            init=inits,
            targets=targets,
            vary=varylst,
            check_limits=False,
            strengths=False,
        )
        opt.disable(target="coll")
        opt.disable(target="spdx")
        opt.disable(target="mu.*_l")
        self.opt = opt
        return opt
