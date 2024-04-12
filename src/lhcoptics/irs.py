from .section import (
    LHCSection,
    lhcprev,
    lhcsucc,
    sort_n,
)
from .model_madx import LHCMadModel


class LHCIR(LHCSection):
    """
    Model of an LHC Interaction Region

    twiss_from_init(beam) -> Make twiss using boundary conditions
    twiss_from_params(beam) -> Make twiss using parameters at the IP
    twiss_full(beam) -> Make twiss using full sequence
    """

    knobs = []
    _extra_param_names = []

    @classmethod
    def from_madx(cls, madx, name=None):
        madmodel=LHCMadModel(madx)
        if name is None:
            name = cls.name
        irn = int(name[-1])
        strength_names = []
        strength_names += madmodel.filter(f"kq[xt]?.*[lr]{irn}$")
        quads = madmodel.filter(f"ktq[x].*[lr]{irn}$")
        quads += madmodel.filter(f"kq[t]?[l]?[0-9][0-9]?\..*[lr]{irn}b[12]$")
        # quads += madmodel.filter(madx, f"kq[t]?.*[lr]{irn}$")
        strength_names += sort_n(quads)
        acb = madmodel.filter(f"acbx.*[lr]{irn}$")
        acb += madmodel.filter(f"acb.*[lr]{irn}b[12]$")
        strength_names += sort_n(acb)
        knobs = madmodel.make_and_set0_knobs(cls.knobs)
        strengths = {st: madx.globals[st] for st in strength_names}
        for knob in knobs:
            madx.globals[knob] = knobs[knob].value
        params = {}
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
        if name is None:
            name = self.__class__.name
        irn = int(name[-1])
        start = f"s.ds.l{irn}"
        end = f"e.ds.r{irn}"
        super().__init__(name, start, end, strengths, params, knobs)
        self.arc_left_name = f"a{lhcprev(irn)}{irn}"
        self.arc_right_name = f"a{irn}{lhcsucc(irn)}"
        self.init_left = None
        self.init_right = None
        self.irn = irn
        self.startb12 = {1: f"s.ds.l{irn}.b1", 2: f"s.ds.l{irn}.b2"}
        self.endb12 = {1: f"e.ds.r{irn}.b1", 2: f"e.ds.r{irn}.b2"}
        self.param_names=self._get_param_default_names()
        self.param_names.extend(self._extra_param_names)

    @property
    def arc_left(self):
        return getattr(self.parent, self.arc_left_name)

    @property
    def arc_right(self):
        return getattr(self.parent, self.arc_right_name)

    @property
    def quads(self):
        return {k: v for k, v in self.strengths.items() if "kq" in k}

    def set_init(self):
        arcleft = self.arc_left
        self.init_left = {
            1: arcleft.twiss_init(1)[1],
            2: arcleft.twiss_init(2)[1],
        }
        arcright = self.arc_right
        self.init_right = {
            1: arcright.twiss_init(1)[0],
            2: arcright.twiss_init(2)[0],
        }

    def twiss_from_init(self, beam=None):
        if beam is None:
            return [self.twiss_from_init(beam=1), self.twiss_from_init(beam=2)]
        if self.init_left is None:
            self.set_init()
        start = self.startb12[beam]
        end = self.endb12[beam]
        return self.model.sequence[beam].twiss(
            start=start, end=end, init=self.init_left[beam]
        )

    def twiss_from_params(self, beam):
        pass

    def twiss_full(self, beam):
        if beam is None:
            return [self.twiss_full(beam=1), self.twiss_full(beam=2)]
        sequence = self.model.sequence[beam]
        start = self.startb12[beam]
        end = self.endb12[beam]
        init = sequence.twiss().get_twiss_init(start)
        return sequence.twiss(start=start, end=end, init=init)

    def plot(self, beam=None, model="init", figlabel=None):
        if beam is None:
            return [self.plot(beam=1), self.plot(beam=2)]
        else:
            if model == "init":
                mktwiss = self.twiss_from_init
            elif model == "full":
                mktwiss = self.twiss_full
            elif model == "params":
                mktwiss = self.twiss_from_params
            if figlabel is None:
                figlabel = f"{self.name}b{beam}"
            return mktwiss(beam).plot(figlabel=figlabel)

    def _get_param_default_names(self):
        ipname = f"ip{self.irn}"
        params = []
        for param in "betx bety alfx alfy dx dy".split():
            for beam in '12':
                params.append(f"{param}{ipname}{beam}")
        for param in "mux muy".split():
            for beam in '12':
                params.append(f"{param}{ipname}b{beam}")
                params.append(f"{param}{ipname}b{beam}_l")
                params.append(f"{param}{ipname}b{beam}_r")
        return params


    def get_params_from_twiss(self, tw1, tw2):
        ipname = f"ip{self.irn}"
        params = {}
        for param in "betx bety alfx alfy dx dy".split():
            for beam, tw in zip([1, 2], [tw1, tw2]):
                params[f"{param}{ipname}{beam}"] = tw[param, ipname]
        for param in "mux muy".split():
            for beam, tw in zip([1, 2], [tw1, tw2]):
                params[f"{param}{ipname}b{beam}"] = (
                    tw[param, self.endb12[beam]]
                    - tw[param, self.startb12[beam]]
                )
                params[f"{param}{ipname}b{beam}_l"] = (
                    tw[param, ipname] - tw[param, self.startb12[beam]]
                )
                params[f"{param}{ipname}b{beam}_r"] = (
                    tw[param, self.endb12[beam]] - tw[param, ipname]
                )
        return params

    def get_params(self, mode="init"):
        if mode == "init":
            tw1 = self.twiss_from_init(1)
            tw2 = self.twiss_from_init(2)
        elif mode == "full":
            tw1 = self.twiss_full(1)
            tw2 = self.twiss_full(2)
        return self.get_params_from_twiss(tw1, tw2)
