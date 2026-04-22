import numpy as np

import xtrack as xt

from .model_xsuite import LHCMadxModel
from .section import LHCSection, gen_acb_alt_names
from .utils import match_compare_log


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
    def from_model(cls, model, name=None, variant=None):
        if name is None:
            name = cls.name
        if variant is None:
            variant = model.get_variant()
        i1, i2 = int(name[1]), int(name[2])
        arc = cls(name=name, strengths={}, params={}, knobs={}, variant=variant)
        knobs = model.make_and_set0_knobs(
            knob_names=arc.gen_knob_names(), variant=variant
        )
        arc.knobs.update(knobs)
        for strength in arc.gen_strength_names():
            if strength in model:
                arc.strengths[strength] = model[strength]
        for param in arc.gen_param_names():
            if param in model:
                arc.params[param] = model[param]
        return arc

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
        variant=None,
    ):
        if name is None:
            name = self.__class__.name
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
            variant=variant,
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
        self.cell_phase_names = [
            f"muxcell{self.name[1:]}b1",
            f"muycell{self.name[1:]}b1",
            f"muxcell{self.name[1:]}b2",
            f"muycell{self.name[1:]}b2",
        ]

    def __repr__(self):
        if self.parent is None:
            return f"<LHCArc {self.name}>"
        else:
            return f"<LHCArc {self.name} in {self.parent.name!r}>"

    def gen_skewquad_names(self):
        a = self.i1 % 2 + 1
        b = self.i2 % 2 + 1
        return [f"kqs.{self.name}b{a}", f"kqs.r{self.i1}b{b}", f"kqs.l{self.i2}b{b}"]

    def gen_acb_names(self):
        out = []
        if self.i1 % 2 == 1:
            out.extend(gen_acb_alt_names("", range(14, 34), 0, "r", self.i1))
            out.extend(gen_acb_alt_names("", range(34, 13, -1), 1, "l", self.i2))
        else:
            out.extend(gen_acb_alt_names("", range(14, 34), 1, "r", self.i1))
            out.extend(gen_acb_alt_names("", range(34, 13, -1), 0, "l", self.i2))
        return out

    def gen_bend_names(self):
        return [f"ab.{self.name}", f"kb.{self.name}"]

    def gen_quad_names(self):
        return [
            f"kqf.{self.name}",
            f"kqd.{self.name}",
            f"kqtf.{self.name}b1",
            f"kqtf.{self.name}b2",
            f"kqtd.{self.name}b1",
            f"kqtd.{self.name}b2",
        ]

    def gen_sext_names(self):
        return [
            f"ks{fd}{nn}.{self.name}b{bb}" for fd in "fd" for nn in "12" for bb in "12"
        ]

    def gen_oct_names(self):
        return [f"ko{fd}.{self.name}b{bb}" for fd in "fd" for bb in "12"]

    def gen_param_names(self):
        return self.phase_names + self.cell_phase_names

    def gen_strength_names(self):
        out = []
        out.extend(self.gen_quad_names())
        out.extend(self.gen_bend_names())
        out.extend(self.gen_acb_names())
        out.extend(self.gen_skewquad_names())
        out.extend(self.gen_sext_names())
        out.extend(self.gen_oct_names())
        return out

    def check_acb_names(self, verbose=True):
        gen = set(self.gen_acb_names())
        mod = set()
        for pattern in [
            f"mcb[hv].[23][0-9].*l{self.i2}.b[12]",
            f"mcb[hv].1[4-9].*l{self.i2}.b[12]",
            f"mcb[hv].[23][0-9].*r{self.i1}.b[12]",
            f"mcb[hv].1[4-9].*r{self.i1}.b[12]",
        ]:
            mod |= set(self.parent.model.get_acb_names(pattern).values())
        extra = gen - mod
        missing = mod - gen
        passed = not extra and not missing
        if verbose and not passed:
            print(f"Extra ACB names in {self.name}: {extra}")
            print(f"Missing ACB names in {self.name}: {missing}")
        return passed

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
            return sequence.twiss(start=start, end=end, init=init, strengths=strengths)

    def get_params_from_twiss(self, tw1=None, tw2=None):
        if tw1 is None or tw2 is None:
            tw1, tw2 = self.twiss(strengths=False)
        params = {
            f"mux{self.name}b1": tw1.mux[-1],
            f"mux{self.name}b2": tw2.mux[-1],
            f"muy{self.name}b1": tw1.muy[-1],
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

    def get_params_from_variables(self, model=None, verbose=False):
        """Get params from model variables"""
        if model is None:
            model = self.model
        if verbose:
            print(f"Getting parameters for Arc {self.name} from variables")
        params = {}
        for pr in ["a", "cell"]:
            for xy in "xy":
                for beam in [1, 2]:
                    key = f"mu{xy}{pr}{self.name[1:]}b{beam}"
                    if key in model:
                        key2 = key
                    # fall back in case of old HL optics
                    elif self.parent.variant.startswith("hl"):
                        key2 = key.replace("a", "")
                    params[key] = model[key2]
        return params

    @property
    def quads(self):
        """Get quads in the arc"""
        return {k: v for k, v in self.strengths.items() if "kqf" in k or "kqd" in k}

    @property
    def skew_quads(self):
        """Get skew quads in the arc"""
        return {k: v for k, v in self.strengths.items() if "kqs" in k}

    @property
    def sexts(self):
        """Get sextupoles in the arc"""
        return {k: v for k, v in self.strengths.items() if "ksf" in k or "ksd" in k}

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
                        tag=f"b{beam}",
                    )
                )
        return targets

    def get_match_kq_vary(self, fd, beam="", dkmax=0.0):
        """Get match vary for the arc"""
        if beam:
            kname = f"kqt{fd}.{self.name}{beam}"
        else:
            kname = f"kq{fd}.{self.name}"
        limits = self.parent.circuits.get_klimits(kname, self.parent.params["p0c"])
        limits[0] *= 1 + dkmax
        limits[1] *= 1 - dkmax
        tag = beam if beam else "common"
        return xt.Vary(name=kname, limits=limits, step=1e-10, tag=tag)

    def match(self, b1=True, b2=True, verbose=False, solve=True, tol=5e-10, fail=True):
        """Match the arc"""
        lhc = self.parent.model.env
        targets = self.get_match_targets(b1=b1, b2=b2)
        varylst = []
        for fd in ["f", "d"]:
            if b1 and b2:
                varylst.append(self.get_match_kq_vary(fd))
                varylst.append(self.get_match_kq_vary(fd, "b1"))
                self.model.ref[f"kqt{fd}.{self.name}b2"] = -self.model.ref[
                    f"kqt{fd}.{self.name}b1"
                ]
            elif b1:
                varylst.append(self.get_match_kq_vary(fd), "b1")
            elif b2:
                varylst.append(self.get_match_kq_vary(fd), "b2")

        mtc = lhc.match(
            solve=False,
            default_tol={None: tol},
            solver_options=dict(max_rel_penalty_increase=2.0),
            targets=targets,
            vary=varylst,
            check_limits=False,
            strengths=False,
        )
        if not verbose:
            mtc._err.show_call_counter = False
        if solve:
            if verbose:
                print(f"Matching phase of Arc {self.name}")
            try:
                mtc.solve(n_steps=10)
                params = self.get_params_from_twiss()
                cc = f"cell{self.name[1:]}"
                for beam in [1, 2]:
                    for xy in "xy":
                        nn = f"mu{xy}{cc}b{beam}"
                        if verbose:
                            print(
                                f"Update {nn}: from {lhc[nn]:.6f} to {params[nn]:.6f} diff {params[nn] - lhc[nn]:.2f}"
                            )
                        lhc[nn] = params[nn]
                if verbose:
                    match_compare_log(mtc)
            except Exception as e:
                if verbose:
                    print(f"Matching failed for Arc {self.name} with error: {e}")
                if fail:
                    raise ValueError(
                        f"Matching failed for Arc {self.name} with error: {e}"
                    )
        return mtc

    def get_close_irs(self):
        ira = getattr(self.parent, f"ir{self.i1}")
        irb = getattr(self.parent, f"ir{self.i2}")
        return ira, irb

    def shift_phase(self, dmuxb1=0, dmuyb1=0, dmuxb2=0, dmuyb2=0, rematch_irs=True):
        arc = self.name
        self.params[f"mux{arc}b1"] += dmuxb1
        self.params[f"muy{arc}b1"] += dmuyb1
        self.params[f"mux{arc}b2"] += dmuxb2
        self.params[f"muy{arc}b2"] += dmuyb2
        self.match_phase(rematch_irs=rematch_irs)

    def match_phase(self, rematch_irs=True):
        print(f"Match {self.name.upper()}")
        self.match().solve()
        if rematch_irs:
            ira, irb = self.get_close_irs()
            print(f"Match {ira.name.upper()}")
            ira.match().solve()
            print(f"Match {irb}")
            irb.match().solve()

    def get_phase(self):
        params = self.get_params_from_twiss()
        return {k: params[k] for k in self.phase_names}

    def round_params(self, verbose=False, dryrun=False):
        if dryrun:
            verbose = True
        for k in self.params:
            new_value = np.round(self.params[k], 9)
            if verbose and new_value != self.params[k]:
                print(f"Parameter {k} rounded to {new_value}")
            if not dryrun:
                self.params[k] = new_value

    def set_ats_phase(self):
        self.strengths[f"kqf.{self.name}"] = 0.0087032988458
        self.strengths[f"kqd.{self.name}"] = -0.0087047551604
        for beam in [1, 2]:
            for fd in "fd":
                kname = f"kqt{fd}.{self.name}b{beam}"
                self.strengths[kname] = 0
        self.update_model()
        self.set_params()

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


class LHCA12(LHCArc):
    name = "a12"


class LHCA12(LHCArc):
    name = "a12"


class LHCA23(LHCArc):
    name = "a23"


class LHCA34(LHCArc):
    name = "a34"


class LHCA45(LHCArc):
    name = "a45"


class LHCA56(LHCArc):
    name = "a56"


class LHCA67(LHCArc):
    name = "a67"


class LHCA78(LHCArc):
    name = "a78"


class LHCA81(LHCArc):
    name = "a81"
