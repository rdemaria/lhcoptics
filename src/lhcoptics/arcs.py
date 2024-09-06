import xtrack as xt

from .model_xsuite import LHCMadxModel
from .section import LHCSection


class ActionArcPhaseAdvance(xt.Action):
    def __init__(self, arc, beam):
        self.arc = arc
        self.beam = beam

    def run(self):
        tw_arc = self.arc.twiss(self.beam, strengths=False)

        return {
            "table": tw_arc,
            "mux": tw_arc["mux", -1] - tw_arc["mux", 0],
            "muy": tw_arc["muy", -1] - tw_arc["muy", 0],
        }


class LHCArc(LHCSection):

    default_twiss_method = "periodic"

    @classmethod
    def from_madx(cls, madx, name="a12"):
        madmodel = LHCMadxModel(madx)
        i1, i2 = int(name[1]), int(name[2])
        strength_names = []
        strength_names += madmodel.filter(f"kq[fd]\\.*{name}$")
        strength_names += madmodel.filter(f"kqt[fd]\\.*{name}b[12]$")
        strength_names += madmodel.filter(f"kqs\\.*{name}b[12]$")
        strength_names += madmodel.filter(f"ksq\\.*r{i1}b[12]$")
        strength_names += madmodel.filter(f"ksq\\.*l{i2}b[12]$")
        strength_names += madmodel.filter(f"ks[fd][12]\\.*{name}b[12]$")
        strength_names += madmodel.filter(f"ko[fd]\\.*{name}b[12]$")
        knobs = {}
        params = {}
        strengths = {st: madx.globals[st] for st in strength_names}
        return cls(name, strengths, params, knobs)

    def __init__(
        self,
        name=None,
        strengths=None,
        params=None,
        knobs=None,
        start=None,
        end=None,
        filename=None,
        parent=None,
    ):
        i1, i2 = int(name[1]), int(name[2])
        start = f"s.ds.r{i1}"
        end = f"e.ds.l{i2}"
        super().__init__(
            name,
            start,
            end,
            strengths,
            params,
            knobs,
            filename=filename,
            parent=parent,
        )
        self.i1 = i1
        self.i2 = i2
        self.start_cellb1 = f"s.cell.{i1}{i2}.b1"
        self.start_cellb2 = f"s.cell.{i1}{i2}.b2"
        self.end_cellb1 = f"e.cell.{i1}{i2}.b1"
        self.end_cellb2 = f"e.cell.{i1}{i2}.b2"
        self.start_cellb12 = (self.start_cellb1, self.start_cellb2)
        self.end_cellb12 = (self.end_cellb1, self.end_cellb2)
        self.startb1 = f"e.ds.r{i1}.b1"
        self.startb2 = f"e.ds.r{i1}.b2"
        self.endb1 = f"s.ds.l{i2}.b1"
        self.endb2 = f"s.ds.l{i2}.b2"
        self.startb12 = self.startb1, self.startb2
        self.endb12 = self.endb1, self.endb2
        self.phase_names = [
            f"mux{self.name}b1",
            f"muy{self.name}b1",
            f"mux{self.name}b2",
            f"muy{self.name}b2",
        ]

    def __repr__(self):
        if self.parent is None:
            return f"<LHCArc {self.name}>"
        else:
            return f"<LHCArc {self.name} in {self.parent.name!r}>"

    def get_init(self, beam):
        """Get twiss init at the beginning and end of the arc."""
        tw = self.twiss(beam, strengths=False)
        start = tw.get_twiss_init(self.startb12[beam - 1])
        end = tw.get_twiss_init(self.endb12[beam - 1])
        start.mux = 0
        start.muy = 0
        end.mux = 0
        end.muy = 0
        return [start, end]

    def get_init_cell(self, beam):
        """Get twiss init at the beginning of the cell."""
        sequence = self.model.sequence[beam]
        start_cell = self.start_cellb12[beam - 1]
        end_cell = self.end_cellb12[beam - 1]
        twinit_cell = sequence.twiss(
            start=start_cell,
            end=end_cell,
            init="periodic",
            only_twiss_init=True,
        )
        return twinit_cell

    def get_init_left(self, beam):
        """Get twiss init at the beginning of the arc."""
        tw = self.twiss(beam, strengths=False)
        start = tw.get_twiss_init(self.startb12[beam - 1])
        start.mux = 0
        start.muy = 0
        return start

    def get_init_right(self, beam):
        """Get twiss init at the end of the arc."""
        tw = self.twiss(beam, strengths=False)
        end = tw.get_twiss_init(self.endb12[beam - 1])
        end.mux = 0
        end.muy = 0
        return end

    def get_init_periodic(self, beam):
        """Get twiss init at the beginning and end of the arc."""
        tw = self.twiss_periodic(beam, strengths=False)
        start = tw.get_twiss_init(self.startb12[beam - 1])
        end = tw.get_twiss_init(self.endb12[beam - 1])
        start.mux = 0
        start.muy = 0
        end.mux = 0
        end.muy = 0
        return [start, end]

    def twiss_full(self, beam=None, strengths=True):
        """Get twiss table of full arc of the full LHC periodic solution"""
        if beam is None:
            return [
                self.twiss_full(beam=1, strengths=strengths),
                self.twiss_full(beam=2, strengths=strengths),
            ]
        else:
            sequence = self.model.sequence[beam]
            start = self.startb12[beam - 1]
            end = self.endb12[beam - 1]
            init = sequence.twiss(strengths=False).get_twiss_init(start)
            return sequence.twiss(
                start=start, end=end, init=init, strengths=strengths
            )

    def get_params_from_twiss(self, tw1, tw2):
        params = {
            f"mux{self.name}b1": tw1.mux[-1],
            f"muy{self.name}b1": tw1.muy[-1],
            f"mux{self.name}b2": tw2.mux[-1],
            f"muy{self.name}b2": tw2.muy[-1],
            f"muxcell{self.name[1:]}b1": tw1["mux", self.end_cellb1]
            - tw1["mux", self.start_cellb1],
            f"muxcell{self.name[1:]}b2": tw2["mux", self.end_cellb2]
            - tw2["mux", self.start_cellb2],
            f"muycell{self.name[1:]}b1": tw1["muy", self.end_cellb1]
            - tw1["muy", self.start_cellb1],
            f"muycell{self.name[1:]}b2": tw2["muy", self.end_cellb2]
            - tw2["muy", self.start_cellb2],
        }
        return params

    def get_params(self):
        """Get params from model"""
        tw1, tw2 = self.twiss(strengths=False)
        return self.get_params_from_twiss(tw1, tw2)

    @property
    def quads(self):
        """Get quads in the arc"""
        return [k for k in self.strengths if "kq" in k]

    def get_match_targets(self, b1=True, b2=True):
        """Get match targets for the arc"""
        targets = []
        beams = []
        if b1:
            beams.append(1)
        if b2:
            beams.append(2)
        for beam in beams:
            for mu in "mux", "muy":
                targets.append(
                    xt.Target(
                        action=ActionArcPhaseAdvance(self, beam),
                        tar=mu,
                        value=self.params[f"{mu}{self.name}b{beam}"],
                        tag=mu,
                    )
                )
        return targets

    def get_match_kq_vary(self, fd, beam="", kmax_marg=0.0):
        """Get match vary for the arc"""
        if beam:
            kname = f"kqt{fd}.{self.name}{beam}"
        else:
            kname = f"kq{fd}.{self.name}"
        limits = self.parent.circuits.get_klimits(
            kname, self.parent.params["p0c"]
        )
        limits[0] *= 1 + kmax_marg
        limits[1] *= 1 - kmax_marg
        tag = beam if beam else "common"
        return xt.Vary(name=kname, limits=limits, step=1e-8, tag=tag)

    def match(self, b1=True, b2=True):
        lhc = self.parent.model.multiline
        if lhc.b1.tracker is None:
            lhc.b1.build_tracker()
        if lhc.b2.tracker is None:
            lhc.b2.build_tracker()
        """Match the arc"""
        targets = self.get_match_targets(b1=b1, b2=b2)
        varylst = []
        for fd in ["f", "d"]:
            if b1 and b2:
                varylst.append(self.get_match_kq_vary(fd))
                varylst.append(self.get_match_kq_vary(fd, "b1"))
                self.model.vars[f"kqt{fd}.{self.name}b2"] = -self.model.vars[
                    f"kqt{fd}.{self.name}b1"
                ]
            elif b1:
                varylst.append(self.get_match_kq_vary(fd), "b1")
            elif b2:
                varylst.append(self.get_match_kq_vary(fd), "b2")

        opt = lhc.match(
            solve=False,
            default_tol={None: 5e-8},
            solver_options=dict(max_rel_penalty_increase=2.0),
            targets=targets,
            vary=varylst,
            check_limits=False,
            strengths=False,
        )
        return opt

    def get_close_irs(self):
        ira = getattr(self.parent, f"ir{self.i1}")
        irb = getattr(self.parent, f"ir{self.i2}")
        return ira, irb

    def shift_phase(
        self, dmuxb1=0, dmuyb1=0, dmuxb2=0, dmuyb2=0, rematch_irs=True
    ):
        arc = self.name
        self.params[f"mux{arc}b1"] += dmuxb1
        self.params[f"muy{arc}b1"] += dmuyb1
        self.params[f"mux{arc}b2"] += dmuxb2
        self.params[f"muy{arc}b2"] += dmuyb2
        self.match_phase(rematch_irs=rematch_irs)

    def match_phase(self, rematch_irs=True):
        print(f"Match {self}")
        self.match().solve()
        if rematch_irs:
            ira, irb = self.get_close_irs()
            print(f"Match {ira}")
            ira.match().solve()
            print(f"Match {irb}")
            irb.match().solve()

    def get_phase(self):
        params = self.get_params()
        return {k: params[k] for k in self.phase_names}

    def to_table(self, *rows):
        from .opttable import LHCArcTable
        return LHCArcTable([self] + list(rows))

    def twiss_cell(self, beam=None, strengths=True):
        """Get twiss table of periodic cell."""
        if beam is None:
            return [
                self.twiss_cell(beam=1, strengths=strengths),
                self.twiss_cell(beam=2, strengths=strengths),
            ]
        sequence = self.model.sequence[beam]
        start_cell = self.start_cellb12[beam - 1]
        end_cell = self.end_cellb12[beam - 1]
        return sequence.twiss(
            start=start_cell,
            end=end_cell,
            init="periodic",
            strengths=strengths,
        )

    def twiss_periodic(self, beam=None, strengths=True):
        """Get twiss table of matched arc."""
        if beam is None:
            return [
                self.twiss(beam=1, strengths=strengths),
                self.twiss(beam=2, strengths=strengths),
            ]
        else:
            twinit_cell = self.get_init_cell(beam)
            start_arc = self.startb12[beam - 1]
            end_arc = self.endb12[beam - 1]

            sequence = self.model.sequence[beam]
            res = sequence.twiss(
                start=start_arc,
                end=end_arc,
                init=twinit_cell,
                strengths=strengths,
            )

            res["mux"] = res["mux"] - res["mux", start_arc]
            res["muy"] = res["muy"] - res["muy", start_arc]

            return res
