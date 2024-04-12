from .section import LHCSection
from .model_xsuite import LHCMadModel
import xtrack as xt


class LHCArc(LHCSection):
    @classmethod
    def from_madx(cls, madx, name="a12"):
        madmodel = LHCMadModel(madx)
        i1, i2 = int(name[1]), int(name[2])
        strength_names = []
        strength_names += madmodel.filter(f"kq[fd]\.*{name}$")
        strength_names += madmodel.filter(f"kqt[fd]\.*{name}b[12]$")
        strength_names += madmodel.filter(f"kqs\.*{name}b[12]$")
        strength_names += madmodel.filter(f"ksq\.*r{i1}b[12]$")
        strength_names += madmodel.filter(f"ksq\.*l{i2}b[12]$")
        strength_names += madmodel.filter(f"ks[fd][12]\.*{name}b[12]$")
        strength_names += madmodel.filter(f"ko[fd]\.*{name}b[12]$")
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
    ):
        i1, i2 = int(name[1]), int(name[2])
        start = f"s.ds.r{i1}"
        end = f"e.ds.l{i2}"
        super().__init__(name, start, end, strengths, params, knobs)
        self.i1 = i1
        self.i2 = i2
        self.start_cell = {1: f"s.cell.{i1}{i2}.b1", 2: f"s.cell.{i1}{i2}.b2"}
        self.end_cell = {1: f"e.cell.{i1}{i2}.b1", 2: f"e.cell.{i1}{i2}.b2"}
        self.startb = {1: f"s.ds.r{i1}.b1", 2: f"s.ds.r{i1}.b2"}
        self.endb = {1: f"e.ds.l{i2}.b1", 2: f"e.ds.l{i2}.b2"}

    def twiss_init(self, beam):
        tw = self.twiss(beam)
        return [
            tw.get_twiss_init(self.startb[beam]),
            tw.get_twiss_init(self.endb[beam]),
        ]

    def twiss_cell(self, beam=None):
        if beam is None:
            return [self.twiss_cell(beam=1), self.twiss_cell(beam=2)]
        sequence = self.model.sequence[beam]
        start_cell = self.start_cell[beam]
        end_cell = self.end_cell[beam]
        return sequence.twiss(start=start_cell, end=end_cell, init="periodic")

    def twiss_cell_init(self, beam):
        sequence = self.model.sequence[beam]
        start_cell = self.start_cell[beam]
        end_cell = self.end_cell[beam]
        twinit_cell = sequence.twiss(
            start=start_cell,
            end=end_cell,
            init="periodic",
            only_twiss_init=True,
        )
        return twinit_cell

    def twiss_full(self, beam=None):
        if beam is None:
            return [self.twiss_full(beam=1), self.twiss_full(beam=2)]
        else:
            sequence = self.model.sequence[beam]
            start = self.startb12[beam]
            end = self.endb12[beam]
            init = sequence.twiss().get_twiss_init(start)
            return sequence.twiss(start=start, end=end, init=init)

    def twiss(self, beam=None):
        if beam is None:
            return [self.twiss(beam=1), self.twiss(beam=2)]
        else:
            twinit_cell = self.twiss_cell_init(beam)
            start_arc = self.startb[beam]
            end_arc = self.endb[beam]

            sequence = self.model.sequence[beam]
            res = sequence.twiss(
                start=start_arc, end=end_arc, init=twinit_cell
            )

            res["mux"] = res["mux"] - res["mux", start_arc]
            res["muy"] = res["muy"] - res["muy", start_arc]

            return res

    def plot(self, beam=None, method="periodic", figlabel=None):
        if beam is None:
            return [self.plot(beam=1), self.plot(beam=2)]
        else:
            twiss = self.twiss(beam)
            if figlabel is None:
                figlabel = f"{self.name}b{beam}"
            return twiss.plot(figlabel=figlabel)

    def get_params_from_twiss(self, tw1, tw2):
        params = {
            f"mux{self.name}b1": tw1.mux[-1],
            f"muy{self.name}b1": tw1.muy[-1],
            f"mux{self.name}b2": tw2.mux[-1],
            f"muy{self.name}b2": tw2.muy[-1],
            f"muxcell{self.name[1:]}b1": tw1["mux", self.end_cell[1]]
            - tw1["mux", self.start_cell[1]],
            f"muxcell{self.name[1:]}b2": tw2["mux", self.end_cell[2]]
            - tw2["mux", self.start_cell[2]],
            f"muycell{self.name[1:]}b1": tw1["muy", self.end_cell[1]]
            - tw1["muy", self.start_cell[1]],
            f"muycell{self.name[1:]}b2": tw2["muy", self.end_cell[2]]
            - tw2["muy", self.start_cell[2]],
        }
        return params


    def get_params(self):
        """Get params from model"""
        tw1, tw2 = self.twiss()
        return self.get_params_from_twiss(tw1, tw2)


class ActionArcPhaseAdvance(xt.Action):
    def __init__(self, arc, beam):
        self.arc = arc
        self.beam = beam

    def run(self):
        tw_arc = self.arc.twiss(self.beam)

        return {
            "table": tw_arc,
            "mux": tw_arc["mux", -1] - tw_arc["mux", 0],
            "muy": tw_arc["muy", -1] - tw_arc["muy", 0],
        }
