import re

import numpy as np
import xtrack as xt
import matplotlib.pyplot as plt

from .model_madx import LHCMadxModel
from .section import LHCSection, lhcprev, lhcsucc, sort_n


class LHCIR(LHCSection):
    """
    Model of an LHC Interaction Region

    twiss_from_init(beam) -> Make twiss using boundary conditions
    twiss_from_params(beam) -> Make twiss using parameters at the IP
    twiss_full(beam) -> Make twiss using full sequence
    """

    knob_names = []
    _extra_param_names = []
    default_twiss_method = "init"

    @classmethod
    def from_madx(cls, madx, name=None):
        madmodel = LHCMadxModel(madx)
        if name is None:
            name = cls.name
        irn = int(name[-1])
        strength_names = []
        quads = madmodel.filter(f"kt?q[xt]?[0-9]?\\.[lr]{irn}$")
        quads += madmodel.filter(f"kq[0-9]\\.lr{irn}$")
        quads += madmodel.filter(f"kqt?l?[0-9][0-9]?\\..*[lr]{irn}b[12]$")
        if irn == 7:
            if "kqt5.l7" in quads:
                quads.remove("kqt5.l7")
            if "kqt5.r7" in quads:
                quads.remove("kqt5.r7")
        strength_names += sort_n(quads)
        strength_names += madmodel.filter(f"kqs\\..*[lr]{irn}b[12]$")
        acb = madmodel.filter(f"acbx.*[lr]{irn}$")
        acb += madmodel.filter(f"acb.*[lr]{irn}b[12]$")
        strength_names += sort_n(acb)
        knobs = madmodel.make_and_set0_knobs(cls.knob_names)
        strengths = {st: madx.globals[st] for st in strength_names}
        for knob in knobs:
            madx.globals[knob] = knobs[knob].value
        params = {}
        for param in "betx bety alfx alfy dx dpx".split():
            for beam in "12":
                ppname = f"{param}ip{irn}b{beam}"
                if ppname in madx.globals:
                    params[ppname] = madx.globals[ppname]
        for param in "mux muy".split():
            for beam in "12":
                for suffix in ["", "_l", "_r"]:
                    ppname = f"{param}ip{irn}b{beam}{suffix}"
                    if ppname in madx.globals:
                        params[ppname] = madx.globals[ppname]
        return cls(name, strengths, params, knobs)

    @classmethod
    def from_madxfile(cls, filename, name=None, stdout=False):
        from cpymad.madx import Madx

        madx = Madx(stdout=stdout)
        madx.call(filename)
        return cls.from_madx(madx, name)

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
        if name is None:
            name = self.__class__.name
        irn = int(name[-1])
        start = f"s.ds.l{irn}"
        end = f"e.ds.r{irn}"
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
        self.arc_left_name = f"a{lhcprev(irn)}{irn}"
        self.arc_right_name = f"a{irn}{lhcsucc(irn)}"
        self.init_left = None
        self.init_right = None
        self.irn = irn
        self.ipname = f"ip{irn}"
        self.startb1 = f"s.ds.l{irn}.b1"
        self.startb2 = f"s.ds.l{irn}.b2"
        self.endb1 = f"e.ds.r{irn}.b1"
        self.endb2 = f"e.ds.r{irn}.b2"
        self.startb12 = (self.startb1, self.startb2)
        self.endb12 = (self.endb1, self.endb2)
        self.param_names = self._get_param_default_names()
        self.param_names.extend(self._extra_param_names)

    def __getitem__(self, key):
        if re.match(r"kqx[123]\.[lr]", key):
            return self.get_kqx(int(key[3]), key[-2])
        return super().__getitem__(key)

    def __repr__(self):
        if self.parent is not None:
            return f"<LHCIR{self.irn} in {self.parent.name!r}>"
        elif self.filename is not None:
            return f"<LHCIR{self.irn} from {self.filename!r}>"
        else:
            return f"<LHCIR{self.irn}>"

    def _get_param_default_names(self):
        ipname = self.ipname
        params = []
        for param in "betx bety alfx alfy dx dy".split():
            for beam in "12":
                params.append(f"{param}{ipname}{beam}")
        for param in "mux muy".split():
            for beam in "12":
                params.append(f"{param}{ipname}b{beam}")
                params.append(f"{param}{ipname}b{beam}_l")
                params.append(f"{param}{ipname}b{beam}_r")
        return params

    @property
    def arc_left(self):
        return getattr(self.parent, self.arc_left_name)

    @property
    def arc_right(self):
        return getattr(self.parent, self.arc_right_name)

    @property
    def quads(self):
        return {
            k: v for k, v in self.strengths.items() if re.match("kt?q[^s]", k)
        }

    @property
    def kqxl(self):
        return [k for k in self.quads if "l" in k and "x" in k]

    @property
    def kqxr(self):
        return [k for k in self.quads if "r" in k and "x" in k]

    def check_quad_strengths(
        self,
        verbose=False,
        p0c=None,
        ratio=1.5,
        margin=0.1,
    ):
        if p0c is None:
            p0c = self.parent.params["p0c"]
        out={}
        if ratio is not None:
            self.get_quad_max_ratio(
                verbose=verbose, ratio=ratio
            )
        if margin is not None:
            #if verbose:
            #    print(f"Name                  Strength      Low    High")
            for k, v in self.quads.items():
                kmin, kmax = self.parent.get_quad_margin(k)
                if kmin < margin or kmax < margin:
                    out[k] = (v, kmin, kmax)
                    if verbose:
                      print(
                        f"{k:20} {v:12.8f} {kmin*100:5.1f}% {kmax*100:5.1f}%"
                    )
        return out

    def get_kqx(self, n, lr):
        side = lr + f"{self.irn}"
        kq = self.strengths[f"kqx.{side}"]
        if n == 3:
            return kq
        elif n == 1 or n == 2:
            ktq = self.strengths[f"ktqx{n}.{side}"]
            return kq + ktq
        else:
            raise ValueError(f"Invalid n={n} for kqx{n}.{side}")

    def get_params_from_twiss(self, tw1, tw2):
        ipname = self.ipname
        params = {}
        for param in "betx bety alfx alfy dx dpx".split():
            for beam, tw in zip([1, 2], [tw1, tw2]):
                params[f"{param}{ipname}b{beam}"] = tw[param, ipname]
        for param in "mux muy".split():
            for beam, tw in zip([1, 2], [tw1, tw2]):
                params[f"{param}{ipname}b{beam}"] = (
                    tw[param, self.endb12[beam - 1]]
                    - tw[param, self.startb12[beam - 1]]
                    + getattr(self.init_left[beam], param)  # ATS change
                )
        for param in "mux muy".split():
            for beam, tw in zip([1, 2], [tw1, tw2]):
                params[f"{param}{ipname}b{beam}_l"] = (
                    tw[param, ipname]
                    - tw[param, self.startb12[beam - 1]]
                    + getattr(self.init_left[beam], param)  # ATS change
                )
        for param in "mux muy".split():
            for beam, tw in zip([1, 2], [tw1, tw2]):
                params[f"{param}{ipname}b{beam}_r"] = (
                    tw[param, self.endb12[beam - 1]] - tw[param, ipname]
                )
        return params

    def get_params(self, mode="init"):
        if mode == "init":
            tw1 = self.twiss_from_init(1, strengths=False)
            tw2 = self.twiss_from_init(2, strengths=False)
        elif mode == "full":
            tw1 = self.twiss_full(1, strengths=False)
            tw2 = self.twiss_full(2, strengths=False)
        params = self.get_params_from_twiss(tw1, tw2)
        return {k: np.round(v, 8) for k, v in params.items()}

    def get_extra_match_targets(self):
        return []

    def get_match_targets(
        self,
        lrphase=False,
        left=True,
        right=True,
        b1=True,
        b2=True,
        phase=True,
    ):
        lines = []
        inits_r = []
        ends = []
        if b1:
            lines.append("b1")
            inits_r.append(self.init_right[1])
            ends.append(self.endb1)
        if b2:
            lines.append("b2")
            inits_r.append(self.init_right[2])
            ends.append(self.endb2)

        targets = []

        if right and phase:
            for tt in ["mux", "muy"]:
                for ll, end in zip(lines, ends):
                    targets.append(
                        xt.Target(
                            tt,
                            self.params[f"{tt}{self.ipname}{ll}"],
                            line=ll,
                            at=end,
                            tag=f"{tt}{self.ipname}{ll}",
                        )
                    )

        if lrphase and left:
            for tt in ["mux", "muy"]:
                for ll in lines:
                    targets.append(
                        xt.Target(
                            tt,
                            self.params[f"{tt}{self.ipname}{ll}_l"],
                            line=ll,
                            at=self.ipname,
                            tag=f"{tt}{self.ipname}{ll}_l",
                        )
                    )

        if lrphase and right:
            for tt in ["mux", "muy"]:
                for ll in lines:
                    targets.append(
                        xt.Target(
                            tt,
                            self.params[f"{tt}{self.ipname}{ll}_r"],
                            line=ll,
                            at=self.ipname,
                            tag=f"{tt}{self.ipname}{ll}_r",
                        )
                    )

        if right:
            for ll, init, end in zip(lines, inits_r, ends):
                for tt in ["alfx", "alfy", "betx", "bety", "dx", "dpx"]:
                    targets.append(
                        xt.Target(
                            tt,
                            getattr(init, tt),
                            line=ll,
                            at=end,
                            tag=f"e_{tt}{ll}",
                        )
                    )

        if left:
            for tt in ["betx", "bety", "alfx", "alfy", "dx", "dpx"]:
                for ll in lines:
                    targets.append(
                        xt.Target(
                            tt,
                            self.params[f"{tt}{self.ipname}{ll}"],
                            line=ll,
                            at=self.ipname,
                            tag=f"ip_{tt}{ll}",
                        )
                    )

        return targets

    def get_match_vary(
        self,
        b1=True,
        b2=True,
        left=True,
        right=True,
        common=True,
        dkmin=0.0,
        dkmax=0.0,
    ):
        varylst = []
        for kk in self.quads:
            add = False
            if "b1" in kk:
                if b1:
                    tag = "b1"
                    add = True
            elif "b2" in kk:
                if b2:
                    tag = "b2"
                    add = True
            elif common:
                tag = "common"
                add = True
            if not left and f"l{self.irn}" in kk:
                add = False
            if not right and f"r{self.irn}" in kk:
                add = False
            if add:
                limits = self.parent.circuits.get_klimits(
                    kk, self.parent.params["p0c"]
                )
                if abs(limits[0]) > abs(limits[1]) * 1.2:
                    limits[0] *= 1 - dkmax
                    limits[1] *= 1 + dkmin
                elif abs(limits[0]) < abs(limits[1]) * 0.8:
                    limits[0] *= 1 + dkmin
                    limits[1] *= 1 - dkmax
                else:
                    limits[0] *= 1 - dkmax
                    limits[1] *= 1 - dkmax
                varylst.append(xt.Vary(kk, limits=limits, step=1e-9, tag=tag))
        return varylst

    def get_quad_max_ratio(self, verbose=False, ratio=2):
        rmax = 1
        for k, v in self.strengths.items():
            if "b1" in k and abs(v) > 0 and "kqt" not in k:
                k2 = k.replace("b1", "b2")
                rat = abs(v / self.strengths[k2])
                rat1 = rat if rat > 1 else 1 / rat
                if rat1 > ratio or verbose:
                    print(f"Ratio {k}/{k2} = {rat:.5f}")
                if rat1 > rmax:
                    rmax = rat1
        if verbose:
            print(f"Max ratio {self}: {rmax:.5f}")
        return rmax

    def match(
        self,
        dkmin=None,
        dkmax=None,
        b1=True,
        b2=True,
        common=True,
        hold_init=False,
        sym_triplets=True,
        no_triplets=False,
        lrphase=False,
        phase=True,
        left=True,
        right=True,
        extra_targets=None,
        vary_ratio=None,
        ratio_threshold=1.5,
        verbose=True,
    ):
        if self.parent.model is None:
            raise ValueError(f"Model not set for {self}")
        if self.parent.circuits is None:
            raise ValueError(f"Circuits not set for {self}")
        if not hold_init:
            self.set_init()
        if len(self.params) == 0:
            self.set_params()
        lhc = self.parent.model.multiline
        if lhc.b1.tracker is None:
            lhc.b1.build_tracker()
        if lhc.b2.tracker is None:
            lhc.b2.build_tracker()

        if dkmin is None:
            dkmin = self.parent.params.get("match_dkmin", 0.01)
        if dkmax is None:
            dkmax = self.parent.params.get("match_dkmax", 0.01)

        targets = LHCIR.get_match_targets(
            self,
            b1=b1,
            b2=b2,
            phase=phase,
            left=left,
            right=right,
            lrphase=lrphase,
        )
        if extra_targets is not None:
            for name, attr in extra_targets.items():
                if name.endswith("b1"):
                    line = "b1"
                    value = self.twiss(beam=1)[attr, name]
                elif name.endswith("b2"):
                    line = "b2"
                    value = self.twiss(beam=2)[attr, name]
                else:
                    raise ValueError(f"Unknown beam for {name}")

                targets.append(
                    xt.Target(
                        tar=attr,
                        value=value,
                        at=name,
                        line=line,
                        tag=f"{name}_{attr}",
                    )
                )

        for target in self.get_extra_match_targets():
            targets.append(target)

        varylst = LHCIR.get_match_vary(
            self,
            b1=b1,
            b2=b2,
            common=common,
            right=right,
            left=left,
            dkmin=dkmin,
            dkmax=dkmax,
        )
        if self.name == "ir1":
            varylst = [v for v in varylst if not v.name.startswith("kq4")]

        if vary_ratio is not None:
            model = self.parent.model
            for k in vary_ratio:
                if "b1" in k:
                    kother = k.replace("b1", "b2")
                rat_name = f"{k}_ratio"
                model[rat_name] = -model[k] / model[kother]
                model.vars[k] = -model.vars[rat_name] * model.vars[kother]
                varylst.append(
                    xt.Vary(
                        rat_name,
                        limits=[1 / ratio_threshold, ratio_threshold],
                        step=1e-9,
                        tag="ratio",
                    )
                )
            # for k in vary_ratio:
            #    varylst = [v for v in varylst if not v.name == k]

        if vary_ratio is not None:
            for k in vary_ratio:
                for vv in varylst:
                    if vv.name == k:
                        vv.active = False

        match = lhc.match(
            solve=False,
            default_tol={None: 5e-8},
            solver_options=dict(max_rel_penalty_increase=2.0),
            lines=["b1", "b2"],
            start=self.startb12,
            end=self.endb12,
            init=[self.init_left[1], self.init_left[2]],
            targets=targets,
            vary=varylst,
            check_limits=False,
            strengths=False,
        )

        if lrphase is False:
            match.disable(target="mu.*_l")

        if sym_triplets:
            for kl, kr in zip(self.kqxl, self.kqxr):
                self.parent.model.vars[kl] = -self.parent.model.vars[kr]
                match.disable(vary_name=kl)

        if no_triplets:
            for kl, kr in zip(self.kqxl, self.kqxr):
                match.disable(vary_name=kl)
                match.disable(vary_name=kr)

        if self.parent.params["match_inj"]:
            if self.name == "ir2" or self.name == "ir8":
                match.disable(vary_name="kt?qx.*")
                self.parent.model[f"kqx.l{self.irn}"] = 0.950981581300e-02
                self.parent.model[f"kqx.r{self.irn}"] = -0.950981581300e-02
                self.parent.model[f"ktqx1.l{self.irn}"] = 0.0
                self.parent.model[f"ktqx1.r{self.irn}"] = 0.0
                self.parent.model[f"ktqx2.l{self.irn}"] = 0.0
                self.parent.model[f"ktqx2.r{self.irn}"] = 0.0
            if self.name == "ir2":
                match.disable(vary_name=r"kq[45]\.[lr]2b1")
                self.parent.model["kq4.l2b1"] = -0.549274522900e-02
                self.parent.model["kq4.r2b1"] = 0.471284923000e-02
                self.parent.model["kq5.l2b1"] = 0.482678438300e-02
                self.parent.model["kq5.r2b1"] = -0.461752389200e-02
            if self.name == "ir8":
                match.disable(vary_name=r"kq[45]\.[lr]8b2")
                self.parent.model["kq4.l8b2"] = 0.449559181916e-02
                self.parent.model["kq4.r8b2"] = -0.447368899600e-02
                self.parent.model["kq5.l8b2"] = -0.538821723331e-02
                self.parent.model["kq5.r8b2"] = 0.425682473400e-02
        if self.name == "ir6":
            match.disable(vary_name="kq4.l6b1")
            match.disable(vary_name="kq4.r6b2")
        self.optimizer = match
        if verbose:
            match.target_status()
            match.vary_status()
        return match

    def match_knobs(self, **kwargs):
        for knob in self.knobs.values():
            if hasattr(knob, "match"):
                knob.match(**kwargs)
        return self

    def plot(
        self,
        beam=None,
        method="init",
        figlabel=None,
        yr=None,
        yl=None,
        filename=None,
    ):
        if beam is None:
            if figlabel is None:
                figlabel1 = f"{self.name}b1".upper()
                figlabel2 = f"{self.name}b2".upper()
            return [
                self.plot(
                    beam=1,
                    method=method,
                    figlabel=figlabel1,
                    yr=yr,
                    yl=yl,
                    filename=filename,
                ),
                self.plot(
                    beam=2,
                    method=method,
                    figlabel=figlabel2,
                    yr=yr,
                    yl=yl,
                    filename=filename,
                ),
            ]
        else:
            if method == "init":
                mktwiss = self.twiss_from_init
            elif method == "full":
                mktwiss = self.twiss_full
            elif method == "params":
                mktwiss = self.twiss_from_params
            if figlabel is None:
                figlabel = f"{self.name}b{beam}".upper()
            plot = mktwiss(beam).plot(figlabel=figlabel, yr=yr, yl=yl)
            plot.ax.set_title(figlabel)
            if filename is not None:
                plot.savefig(filename.format(figlabel=figlabel))
            return plot

    def set_bumps_off(self):
        for k, knob in self.knob_names.items():
            if re.match(r"on_[xsao]", k):
                knob.value = 0
                self.parent.model[k] = 0

    def set_betastar(self, betx=None, bety=None):
        if bety is None:
            bety=betx
        if self.irn in [1, 2, 5, 8]:
            self.params[f"betx{self.ipname}b1"] = betx
            self.params[f"betx{self.ipname}b2"] = betx
            self.params[f"bety{self.ipname}b1"] = bety
            self.params[f"bety{self.ipname}b2"] = bety
        else:
            raise ValueError(f"IR{self.irn} not allowed for beta* setting")

    def set_init_left(self, beam):
        self.init_left[beam] = self.arc_left.get_init_right(beam)

    def set_init_right(self, beam):
        self.init_right[beam] = self.arc_right.get_init_left(beam)

    def set_init(self):
        self.init_left = {}
        self.set_init_left(1),
        self.set_init_left(2),
        self.init_right = {}
        self.set_init_right(1),
        self.set_init_right(2),

    def specialize_knobs(self):
        for k, knob in list(self.knobs.items()):
            ## workaround because knob.specialize is also class method
            ## of an already specialized knob
            self.knobs[k] = knob.specialize(knob)

    def to_table(self, *rows):
        from .opttable import LHCIRTable

        return LHCIRTable([self.copy()] + list(rows))

    def twiss(self, beam=None, method="init", strengths=True):
        if method == "init":
            return self.twiss_from_init(beam, strengths=strengths)
        elif method == "full":
            return self.twiss_full(beam, strengths=strengths)
        elif method == "params":
            return self.twiss_from_params(beam, strengths=strengths)

    def twiss_from_init(self, beam=None, strengths=True):
        if beam is None:
            return [
                self.twiss_from_init(beam=1, strengths=strengths),
                self.twiss_from_init(beam=2, strengths=strengths),
            ]
        if self.init_left is None:
            self.set_init()
        start = self.startb12[beam - 1]
        end = self.endb12[beam - 1]
        return self.model.sequence[beam].twiss(
            start=start,
            end=end,
            init=self.init_left[beam],
            strengths=strengths,
        )

    def twiss_from_params(self, beam):
        if beam is None:
            return [
                self.twiss_from_params(beam=1),
                self.twiss_from_params(beam=2),
            ]
        sequence = self.model.sequence[beam]
        init = xt.TwissInit(
            element_name=self.ipname,
            betx=self.params[f"betx{self.ipname}b{beam}"],
            alfx=self.params[f"alfx{self.ipname}b{beam}"],
            bety=self.params[f"bety{self.ipname}b{beam}"],
            alfy=self.params[f"alfy{self.ipname}b{beam}"],
            dx=self.params[f"dx{self.ipname}b{beam}"],
            dpx=self.params[f"dpx{self.ipname}b{beam}"],
        )
        return sequence.twiss(
            start=self.startb12[beam - 1], end=self.endb12[beam - 1], init=init
        )

    def twiss_full(self, beam, strengths=True):
        if beam is None:
            return [
                self.twiss_full(beam=1, strengths=strengths),
                self.twiss_full(beam=2, strengths=strengths),
            ]
        sequence = self.model.sequence[beam]
        start = self.startb12[beam - 1]
        end = self.endb12[beam - 1]
        init = sequence.twiss(strengths=strengths).get_twiss_init(start)
        return sequence.twiss(
            start=start, end=end, init=init, strengths=strengths
        )
