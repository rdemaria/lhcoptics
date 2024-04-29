import re

import xtrack as xt
import xdeps as xd
import numpy as np
import matplotlib.pyplot as plt

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
    default_twiss_method = "init"

    @classmethod
    def from_madx(cls, madx, name=None):
        madmodel = LHCMadModel(madx)
        if name is None:
            name = cls.name
        irn = int(name[-1])
        strength_names = []
        quads = madmodel.filter(f"kt?q[xt]?[0-9]?.*[lr]{irn}$")
        #quads += madmodel.filter(f"ktq[x][0-9].*[lr]{irn}$")
        quads += madmodel.filter(f"kqt?l?[0-9][0-9]?\..*[lr]{irn}b[12]$")
        if irn == 7:
            if "kqt5.l7" in quads:
                quads.remove("kqt5.l7")
            if "kqt5.r7" in quads:
                quads.remove("kqt5.r7")
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
        filename=None,
    ):
        if name is None:
            name = self.__class__.name
        irn = int(name[-1])
        start = f"s.ds.l{irn}"
        end = f"e.ds.r{irn}"
        super().__init__(
            name, start, end, strengths, params, knobs, filename=filename
        )
        self.arc_left_name = f"a{lhcprev(irn)}{irn}"
        self.arc_right_name = f"a{irn}{lhcsucc(irn)}"
        self.init_left = None
        self.init_right = None
        self.irn = irn
        self.ipname = f"ip{irn}"
        self.startb12 = {1: f"s.ds.l{irn}.b1", 2: f"s.ds.l{irn}.b2"}
        self.endb12 = {1: f"e.ds.r{irn}.b1", 2: f"e.ds.r{irn}.b2"}
        self.param_names = self._get_param_default_names()
        self.param_names.extend(self._extra_param_names)

    def __repr__(self):
        if self.parent is not None:
            return f"<LHCIR{self.irn} in {self.parent.name}>"
        elif self.filename is not None:
            return f"<LHCIR{self.irn} from {self.filename}>"
        else:
            return f"<LHCIR{self.irn}>"

    def to_table(self, *rows):
        return LHCIRTable(list(rows))

    @property
    def arc_left(self):
        return getattr(self.parent, self.arc_left_name)

    @property
    def arc_right(self):
        return getattr(self.parent, self.arc_right_name)

    @property
    def quads(self):
        return {
            k: v
            for k, v in self.strengths.items()
            if "kq" in k and not "kqs" in k
        }

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
        if beam is None:
            return [
                self.twiss_from_params(beam=1),
                self.twiss_from_params(beam=2),
            ]
        sequence = self.model.sequence[beam]
        init = xt.TwissInit(
            element_name=self.ipname,
            betx=self.params[f"betx{self.ipname}{beam}"],
            alfx=self.params[f"alfx{self.ipname}{beam}"],
            bety=self.params[f"bety{self.ipname}{beam}"],
            alfy=self.params[f"alfy{self.ipname}{beam}"],
            dx=self.params[f"dx{self.ipname}{beam}"],
            dy=self.params[f"dy{self.ipname}{beam}"],
        )
        return sequence.twiss(
            start=self.startb12[beam], end=self.endb12[beam], init=init
        )

    def twiss_full(self, beam):
        if beam is None:
            return [self.twiss_full(beam=1), self.twiss_full(beam=2)]
        sequence = self.model.sequence[beam]
        start = self.startb12[beam]
        end = self.endb12[beam]
        init = sequence.twiss().get_twiss_init(start)
        return sequence.twiss(start=start, end=end, init=init)

    def twiss(self, beam=None, method="init"):
        if method == "init":
            return self.twiss_from_init(beam)
        elif method == "full":
            return self.twiss_full(beam)
        elif method == "params":
            return self.twiss_from_params(beam)

    def plot(self, beam=None, method="init", figlabel=None):
        if beam is None:
            if figlabel is None:
                figlabel1 = f"{self.name}b1"
                figlabel2 = f"{self.name}b2"
            return [
                self.plot(beam=1, method=method, figlabel=figlabel1),
                self.plot(beam=2, method=method, figlabel=figlabel2),
            ]
        else:
            if method == "init":
                mktwiss = self.twiss_from_init
            elif method == "full":
                mktwiss = self.twiss_full
            elif method == "params":
                mktwiss = self.twiss_from_params
            if figlabel is None:
                figlabel = f"{self.name}b{beam}"
            return mktwiss(beam).plot(figlabel=figlabel)

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

    def get_params_from_twiss(self, tw1, tw2):
        ipname = self.ipname
        params = {}
        for param in "betx bety alfx alfy dx dpx".split():
            for beam, tw in zip([1, 2], [tw1, tw2]):
                params[f"{param}{ipname}b{beam}"] = tw[param, ipname]
        for param in "mux muy".split():
            for beam, tw in zip([1, 2], [tw1, tw2]):
                params[f"{param}{ipname}b{beam}"] = (
                    tw[param, self.endb12[beam]]
                    - tw[param, self.startb12[beam]]
                )
        for param in "mux muy".split():
            for beam, tw in zip([1, 2], [tw1, tw2]):
                params[f"{param}{ipname}b{beam}_l"] = (
                    tw[param, ipname] - tw[param, self.startb12[beam]]
                )
        for param in "mux muy".split():
            for beam, tw in zip([1, 2], [tw1, tw2]):
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
        params = self.get_params_from_twiss(tw1, tw2)
        return {k: np.round(v, 8) for k, v in params.items()}

    def get_match_targets(
        self, lrphase=False, left=True, right=True, b1=True, b2=True
    ):
        lines = []
        inits_r = []
        ends = []
        if b1:
            lines.append("b1")
            inits_r.append(self.init_right[1])
            ends.append(self.endb12[1])
        if b2:
            lines.append("b2")
            inits_r.append(self.init_right[2])
            ends.append(self.endb12[2])

        lines = [f"b{beam}" for beam in [1, 2]]

        targets = []

        if right:
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

        if lrphase:
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
                            tol=1e-1,
                            tag=f"ip_{tt}{ll}",
                        )
                    )

        return targets

    def get_match_vary(
        self, b1=True, b2=True, common=True, kmin_marg=0.0, kmax_marg=0.0
    ):
        varylst = []
        for kk in self.quads:
            limits = self.parent.circuits[kk].get_klimits()
            limits[0] *= 1 + kmin_marg
            limits[1] *= 1 - kmax_marg
            if "b1" in kk and b1:
                tag = "b1"
                add = True
            elif "b2" in kk and b2:
                tag = "b2"
                add = True
            elif common:
                tag = "common"
                add = True
            if add:
                varylst.append(xt.Vary(kk, limits=limits, step=1e-9, tag=tag))
        return varylst


class Col:
    def __init__(self, attr, rows):
        self.attr = attr
        self.rows = rows

    def __getitem__(self, k):
        attrs = [getattr(row, self.attr) for row in self.rows]
        return [attr[k] for attr in attrs]

    def __repr__(self) -> str:
        return f"<Col {self.attr!r} {len(self.rows)} rows>"


class LHCIRTable:

    def __init__(self, rows):
        self.rows = rows
        self.strengths = Col("strengths", rows)
        self.params = Col("params", rows)
        self.knobs = Col("knobs", rows)

    def __len__(self):
        return len(self.rows)

    def __repr__(self):
        return f"<LHCIRTable {len(self)} rows>"

    def get_quads(self, n=None):
        if n is None:
            dct = {}
            for n in range(1, 13):
                dct.update(self.tab_quads(n))
        else:
            ir = self.rows[0]
            quad_names = [
                k for k in ir.quads if re.match(f"kq[xtl]*{n}\\.", k)
            ]
            return {k: [ir.quads[k] for ir in self.rows] for k in quad_names}

    @property
    def tab(self):
        tab = {}
        tab["id"] = np.arange(len(self))
        ir0 = self.rows[0]
        for k in ir0.strengths:
            tab[k] = [ir.strengths[k] for ir in self.rows]
        for k in ir0.params:
            tab[k] = [ir.params[k] for ir in self.rows]
        for k in ir0.knobs:
            tab[f"{k.name}_value"] = [ir.knobs[k].value for ir in self.rows]
            for w in ir0.knobs[k].weights:
                tab[f"{k}_{w}"] = [ir.knobs[k].weights[w] for ir in self.rows]
        return xd.Table(tab, index="id")

    def __getitem__(self, k):
        if k == "id":
            return np.arange(len(self))
        else:
            return np.array([ir[k] for ir in self.rows])

    def plot_quad(
        self, n, xaxis="id", ax=None, title=None, figname=None, p0c=6.8e12
    ):
        brho = p0c / 299792458
        if title is None:
            title = f"{self.rows[0].name.upper()} Q{n}"
        if figname is None:
            figname = f"{self.rows[0].name.upper()} Q{n}"
        if ax is None:
            fig, ax = plt.subplots(num=figname)
            ax = plt.gca()
        xx = self[xaxis]
        for q in self.get_quads(n):
            ax.plot(xx, self[q] * brho, label=q)
        ax.set_title(title)
        ax.set_xlabel(xaxis)
        ax.set_ylabel(r"k [$T/m$]")
        ax.legend()

    def plot_quads(
        self, xaxis="id", fig=None, title=None, figname=None, p0c=6.8e12
    ):
        nq = []
        for n in range(1, 13):
            if len(self.get_quads(n)) > 0:
                nq.append(n)
        rows = len(nq) // 3
        cols = len(nq) // rows
        if hasattr(self, "fig") and self.fig is not None:
            fig = self.fig
        if fig is None:
            if title is None:
                title = f"{self.rows[0].name.upper()} Quads"
            if figname is None:
                figname = f"{self.rows[0].name.upper()} Quads"
            fig, axs = plt.subplots(
                cols, rows, num=figname, figsize=(2.5 * cols, 2.5 * cols)
            )
            self.fig = fig
            axs = axs.flatten()
        else:
            axs = fig.axes
        for i, (n, ax) in enumerate(zip(nq, axs)):
            ax.clear()
            self.plot_quad(n, xaxis, ax, p0c=p0c)
            ax.title.set_text(None)
            if i % rows > 0:
                ax.set_ylabel(None)
            if i < len(nq) - cols:
                ax.set_xlabel(None)
        for ax in axs[len(nq) :]:
            ax.set_visible(False)
        plt.suptitle(title)
        # plt.tight_layout()

    def interp_val(self, p, kname, order=1, pname="id"):
        pp = self[pname]
        yy = self[kname]
        return np.interp(p, pp, yy)

    def interp(self, n, order=1, xaxis="id"):
        ir0 = self.rows[0]
        strengths = {
            k: self.interp_val(n, k, order, xaxis) for k in ir0.strengths
        }
        params = {k: self.interp_val(n, k, order, xaxis) for k in ir0.params}
        return ir0.__class__(strengths=strengths, params=params)
