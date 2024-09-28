import re
import json

import matplotlib.pyplot as plt
import numpy as np
import xdeps as xd

from .lsa_util import get_lsa
from .rdmsignal import poly_fit, poly_val


class Col:
    def __init__(self, attr, rows):
        self.attr = attr
        self.rows = rows

    def __getitem__(self, k):
        attrs = [getattr(row, self.attr) for row in self.rows]
        return np.array([attr[k] for attr in attrs])

    def __setitem__(self, k, v):
        attrs = [getattr(row, self.attr) for row in self.rows]
        if np.isscalar(v):
            for attr in attrs:
                attr[k] = v
        else:
            for attr, vv in zip(attrs, v):
                attr[k] = vv

    def __repr__(self) -> str:
        return f"<Col {self.attr!r} {len(self.rows)} rows>"

    def show(self):
        attrs = getattr(self.rows[0], self.attr)
        for k in attrs:
            print(k, self[k])


class ColKnob:
    def __init__(self, attr, rows):
        self.attr = attr
        self.rows = rows

    def __getitem__(self, k):
        attrs = [getattr(row, self.attr) for row in self.rows]
        return np.array([attr[k].value for attr in attrs])

    def __repr__(self) -> str:
        return f"<ColKnob {self.attr!r} {len(self.rows)} rows>"


class LHCSectionTable:
    def to_json(self, filename):
        rows = [row.to_dict() for row in self.rows]
        clsname = self.rows[0].__class__.__name__
        dct = {"class": clsname, "rows": rows}
        json.dump(dct, open(filename, "w"), indent=2)
        return self

    @classmethod
    def from_json(cls, filename):
        dct = json.load(open(filename))
        rowcls = globals()[dct["class"]]
        rows = [rowcls.from_dict(row) for row in dct["rows"]]
        return cls(rows)

    def __init__(self, rows, parent=None):
        rows = list(rows)
        self.rows = rows
        self.strengths = Col("strengths", rows)
        self.params = Col("params", rows)
        self.knobs = Col("knobs", rows)
        self.parent = parent

    def clear(self):
        self.rows.clear()
        return self

    def append(self, row):
        self.rows.append(row)
        return self

    def extend(self, rows):
        self.rows.extend(rows)
        return self

    def insert(self, i, row):
        self.rows.insert(i, row)
        return self

    def remove(self, row):
        self.rows.remove(row)
        return self

    def pop(self, i):
        return self.rows.pop(i)

    def index(self, row):
        return self.rows.index(row)

    def count(self, row):
        return self.rows.count(row)

    def get_p0c(self):
        return [row.params["p0c"] for row in self.rows]

    def copy(self):
        return self.__class__([row.copy() for row in self.rows], self.parent)

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
            tab[f"{k}_value"] = [ir.knobs[k].value for ir in self.rows]
            for w in ir0.knobs[k].weights:
                tab[f"{k}_{w}"] = [ir.knobs[k].weights[w] for ir in self.rows]
        return xd.Table(tab, index="id")

    def interp_val(self, x, kname, order=1, xname="id", soft=False):
        xx = self[xname]
        yy = self[kname]
        if not isinstance(yy[0], (int, float)):
            return yy[0]
        if order == -1:  # Nearest
            return np.interp(x, xx, yy)
        if order == 0:  # piecewise linear
            # return scipy.interpolate.interp1d(xx, yy, kind="linear")(x)
            return np.interp(x, xx, yy)
        if order > 0:
            x0 = [xx[0], xx[-1]]
            y0 = [yy[0], yy[-1]]
            if soft:
                xp0 = [xx[0], xx[-1]]
                yp0 = [0, 0]
            else:
                xp0 = []
                yp0 = []
            poly = poly_fit(order, xx, yy, x0, y0, xp0, yp0)
            return poly_val(poly, x)
        else:
            return np.interp(x, xx, yy)

    def __getitem__(self, k):
        if isinstance(k, int):
            return self.rows[k]
        elif k == "id":
            return np.arange(len(self))
        else:
            return np.array([ss[k] for ss in self.rows])

    def __setitem__(self, k, row):
        if isinstance(k, int):
            self.rows[k] = row
            if hasattr(self, "irs") and hasattr(row, "irs"):
                print(k, row)
                for ss, ss_row in zip(
                    self.irs + self.arcs, row.irs + row.arcs
                ):
                    ss.rows[k] = ss_row
        else:
            raise ValueError(f"Cannot set item {k}")

    def __iter__(self):
        return iter(self.rows)

    def __len__(self):
        return len(self.rows)

    def __repr__(self):
        if len(self) == 0:
            return "<Table: 0 rows>"
        cls = self.rows[0].__class__.__name__
        return f"<Table {cls}: {len(self)} rows>"

    def knob_table(self, knobname, max=True):
        knob = self.knobs[knobname]
        weights = {"id": self["id"]}
        for k in knob[0].weights:
            weights[k] = np.array([kk.weights[k] for kk in knob])
        tab = xd.Table(weights, index="id")
        if max:
            tab._append_row(
                {**{k: abs(v).max() for k, v in weights.items()}, "id": "max"}
            )
        return xd.Table(weights, index="id")

    def knob_plot(self, knobname):
        tab = self.knob_table(knobname, max=False)
        fig, ax = plt.subplots(num=knobname)
        for k in tab:
            if k == "id":
                continue
            ax.plot(tab["id"], tab[k], label=k)
        ax.legend()
        ax.set_xlabel("id")
        ax.set_ylabel("factors")
        ax.set_title(knobname)


class LHCIRTable(LHCSectionTable):

    def interp(self, n, order=1, xaxis="id"):
        ir0 = self.rows[0]
        strengths = {
            k: self.interp_val(n, k, order, xaxis) for k in ir0.strengths
        }
        params = {k: self.interp_val(n, k, order, xaxis) for k in ir0.params}
        ##TODO add knobs
        return ir0.__class__(
            strengths=strengths, params=params, parent=self.parent
        )

    def get_quads(self, n=None):
        if n is None:
            dct = {}
            for n in range(1, 13):
                dct.update(self.tab_quads(n))
        else:
            ir = self.rows[0]
            if n > 3:
                quad_names = [
                    k for k in ir.quads if re.match(f"kt?q[xtl]*{n}\\.", k)
                ]
            elif n == 1:
                quad_names = [k for k in ir.quads if re.match("kt?qx", k)]
            else:
                quad_names = []
            return {k: [ir.quads[k] for ir in self.rows] for k in quad_names}

    def get_p0c(self):
        if self.rows[0].parent is None or self.rows[-1].parent is None:
            return np.array([7e12 for row in self.rows])
        else:
            return np.array([row.parent.params["p0c"] for row in self.rows])

    def get(self, k):
        return np.array([ir[k] for ir in self.rows])

    def get_gradient(self, kname, p0c=None):
        if p0c is None:
            p0c = self.get_p0c()
        brho = p0c / 299792458
        return self[kname] * brho

    def plot_params(self, xaxis="id", ax=None, figname=None):
        xx = self[xaxis]
        if figname is None:
            figname = f"{self} Params"
        rows = 2
        cols = 3
        irn = self.rows[0].irn
        plt.figure(num=figname, figsize=(6 * cols, 4 * rows)).clear()
        fig, axs = plt.subplots(rows, cols, num=figname)
        # axs = fig.axes
        axs = axs.flatten()
        atn = "bet alf mu".split()
        for atn, ax in zip(atn, axs):
            ax.clear()
            for xy in "xy":
                for b12 in "12":
                    attr = f"{atn}{xy}ip{irn}b{b12}"
                    ax.plot(xx, self.params[attr], label=attr)
                    ax.set_ylabel(atn)
                    ax.set_xlabel(xaxis)
            ax.legend()
        atn = "dx dpx".split()
        for atn, ax in zip(atn, axs[3:]):
            ax.clear()
            for b12 in "12":
                attr = f"{atn}ip{irn}b{b12}"
                ax.plot(xx, self.params[attr], label=attr)
                ax.set_ylabel(atn)
                ax.set_xlabel(xaxis)
            ax.legend()
        ax = axs[-1]
        ax.clear()
        for xy in "xy":
            for b12 in "12":
                attr = f"mu{xy}ip{irn}b{b12}_l"
                ax.plot(xx, self.params[attr], label=attr)
                ax.set_ylabel("mu")
                ax.set_xlabel(xaxis)
        ax.legend()
        plt.suptitle(figname)
        # plt.tight_layout()

    def plot_quad(
        self, n, xaxis="id", ax=None, title=None, figname=None, p0c=None
    ):
        if p0c is None:
            p0c = self.get_p0c()
        if title is None:
            title = f"{self.rows[0].name.upper()} Q{n}"
        if figname is None:
            figname = f"{self.rows[0].name.upper()} Q{n}"
        if ax is None:
            fig, ax = plt.subplots(num=figname)
        xx = self[xaxis]
        if n == 1:
            kqx1l = self.get_gradient(f"kqx1.l{self.rows[0].irn}", p0c=p0c)
            kqx1r = self.get_gradient(f"kqx1.r{self.rows[0].irn}", p0c=p0c)
            kqx2l = self.get_gradient(f"kqx2.l{self.rows[1].irn}", p0c=p0c)
            kqx2r = self.get_gradient(f"kqx2.r{self.rows[1].irn}", p0c=p0c)
            kqx3l = self.get_gradient(f"kqx3.l{self.rows[0].irn}", p0c=p0c)
            kqx3r = self.get_gradient(f"kqx3.r{self.rows[0].irn}", p0c=p0c)
            ax.plot(xx, abs(kqx1l), label=f"kqx1.l{self.rows[0].irn}")
            ax.plot(xx, abs(kqx1r), label=f"kqx1.r{self.rows[0].irn}")
            ax.plot(xx, abs(kqx2l), label=f"kqx2.l{self.rows[1].irn}")
            ax.plot(xx, abs(kqx2r), label=f"kqx2.r{self.rows[1].irn}")
            ax.plot(xx, abs(kqx3l), label=f"kqx3.l{self.rows[0].irn}")
            ax.plot(xx, abs(kqx3r), label=f"kqx3.r{self.rows[0].irn}")
        else:
            for q in self.get_quads(n):
                grad = self.get_gradient(q, p0c=p0c)
                if "t" not in q and np.all(grad < 0):
                    ax.plot(xx, -grad, label=f"-{q}")
                else:
                    ax.plot(xx, grad, label=q)
        ax.set_title(title)
        ax.set_xlabel(xaxis)
        ax.set_ylabel(r"k [$T/m$]")
        ax.legend()

    def plot_quad_fit(
        self,
        n,
        order=1,
        soft=False,
        xaxis="id",
        ax=None,
        title=None,
        figname=None,
        p0c=6.8e12,
    ):
        brho = p0c / 299792458
        if title is None:
            title = f"{self.rows[0].name.upper()} Q{n}"
        if figname is None:
            figname = f"{self.rows[0].name.upper()} Q{n}"
        if ax is None:
            fig, ax = plt.subplots(num=figname, clear=True)
        xx = self[xaxis]
        for nkk, kname in enumerate(self.get_quads(n)):
            yy = self[kname]
            xx_fit = np.linspace(xx[0], xx[-1], 1001)
            yy_fit = self.interp_val(
                xx_fit, kname, order=order, xname=xaxis, soft=soft
            )
            ax.plot(xx, yy * brho, "o", label=f"{kname}", color=f"C{nkk}")
            ax.plot(xx_fit, yy_fit * brho, label=f"fit {kname}", color="k")
            yy_res = self.interp_val(
                xx, kname, order=order, xname=xaxis, soft=soft
            )
            print("residuals", kname, np.abs(yy - yy_res).max())
        ax.set_title(title)
        ax.set_xlabel(xaxis)
        ax.set_ylabel(r"k [$T/m$]")
        ax.legend()

    def plot_param_fit(
        self, attr, order=1, xaxis="id", soft=False, ax=None, title=None
    ):
        if title is None:
            title = f"{self.rows[0].name.upper()} {attr}"
        if ax is None:
            fig, ax = plt.subplots(num=title)
        xx = self[xaxis]
        yy = self[attr]
        xx_fit = np.linspace(xx[0], xx[-1], 1001)
        yy_fit = self.interp_val(
            xx_fit, attr, order=order, xname=xaxis, soft=soft
        )
        ax.plot(xx, yy, "o", label=attr)
        ax.plot(xx_fit, yy_fit, label=f"fit {attr}")
        ax.set_title(title)
        ax.set_xlabel(xaxis)
        ax.set_ylabel(attr)
        ax.legend()

    def plot_quads(
        self,
        xaxis="id",
        fig=None,
        title=None,
        figname=None,
        p0c=None,
        current=False,
    ):
        nq = []
        for n in range(1, 14):
            if len(self.get_quads(n)) > 0:
                nq.append(n)
        rows = int(np.ceil(len(nq) / 3))
        cols = int(np.ceil(len(nq) / rows))
        assert cols * rows >= len(nq)
        # print(f"cols={cols}, rows={rows}")
        # if hasattr(self, "fig") and self.fig is not None:
        #    fig = self.fig
        if fig is None:
            if title is None:
                title = f"{self.rows[0].name.upper()} Quads"
            if figname is None:
                figname = f"{self.rows[0].name.upper()} Quads"
            plt.figure(num=figname, figsize=(4 * rows, 4 * cols)).clear()
            fig, axs = plt.subplots(cols, rows, num=figname)
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


class LHCArcTable(LHCSectionTable):

    def interp(self, n, order=1, xaxis="id", soft=True):
        arc0 = self.rows[0]
        strengths = {
            k: self.interp_val(n, k, order, xaxis, soft=soft)
            for k in arc0.strengths
        }
        params = {
            k: self.interp_val(
                n,
                k,
                order,
                xaxis,
                soft=soft,
            )
            for k in arc0.params
        }
        ##TODO add knobs
        return arc0.__class__(
            arc0.name, strengths=strengths, params=params, parent=self.parent
        )


class LHCOpticsTable(LHCSectionTable):
    def tab_from_lsa(self, beamprocess, parameters=None):
        if parameters is None:
            parameters = ["LHCBEAM/MOMENTUM"]
        lsa = get_lsa()
        opttable = lsa.getOpticTable(beamprocess)
        out = {
            "name": np.array([oo.name for oo in opttable]),
            "time": np.array([oo.time for oo in opttable]),
        }
        for pp in parameters:
            trim = lsa.getLastTrim(pp, beamprocess)
            ptime = trim.data[0]
            pval = trim.data[1]
            out[pp] = np.interp(out["time"], ptime, pval)
        return xd.Table(out)

    def __init__(self, rows, time=None ,beamprocess=None):
        self.rows = rows
        self.params = Col("params", rows)
        self.knobs = Col("knobs", rows)
        self.irs = [
            LHCIRTable(irlst, self)
            for irlst in zip(*[row.irs for row in self.rows])
        ]
        self.arcs = [
            LHCArcTable(arclst, self)
            for arclst in zip(*[row.arcs for row in self.rows])
        ]
        for ir in self.irs:
            setattr(self, ir.rows[0].name, ir)
        for arc in self.arcs:
            setattr(self, arc.rows[0].name, arc)

        self.time = time
        self.beamprocess = beamprocess

    def interp(self, n, order=1, xaxis="id"):
        opt0 = self.rows[0]
        params = {k: self.interp_val(n, k, order, xaxis) for k in opt0.params}
        irs = [ir.interp(n, order, xaxis) for ir in self.irs]
        arcs = [arc.interp(n, order, xaxis) for arc in self.arcs]
        return opt0.__class__(
            name=f"{xaxis}={n}", params=params, irs=irs, arcs=arcs
        )

    def clear(self):
        self.rows.clear()
        for ss in self.irs + self.arcs:
            ss.clear()
        return self

    def append(self, row):
        self.rows.append(row)
        for ss, ss_row in zip(self.irs + self.arcs, row.irs + row.arcs):
            ss.rows.append(ss_row)
        return self

    def extend(self, rows):
        self.rows.extend(rows)
        for ss, ss_rows in zip(
            self.irs + self.arcs, zip(*[row.irs for row in rows])
        ):
            ss.extend(ss_rows)
        return self

    def insert(self, i, row):
        self.rows.insert(i, row)
        for ss, ss_row in zip(self.irs + self.arcs, row.irs + row.arcs):
            ss.rows.insert(i, ss_row)
        return self

    def remove(self, row):
        self.rows.remove(row)
        for ss, ss_row in zip(self.irs + self.arcs, row.irs + row.arcs):
            ss.rows.remove(ss_row)
        return self

    def plot_quads(self, current=False, p0c=None):
        for ir in self.irs:
            ir.plot_quads(current=current, p0c=p0c)


class BeamProcess:

    def __init__(self, name, opticstable, parameters=None):
        self.name = name
        self.opticstable = opticstable
        self.parameters = parameters

    @classmethod
    def from_lsa(cls, name, beamprocess, parameters=None):
        if parameters is None:
            parameters = ["LHCBEAM/MOMENTUM"]
        lsa = get_lsa()
        opttable = lsa.getOpticTable(beamprocess)
        out = {
            "name": np.array([oo.name for oo in opttable]),
            "time": np.array([oo.time for oo in opttable]),
        }
        for pp in parameters:
            trim = lsa.getLastTrim(pp, beamprocess)
            ptime = trim.data[0]
            pval = trim.data[1]
            out[pp] = np.interp(out["time"], ptime, pval)
        return xd.Table(out)
