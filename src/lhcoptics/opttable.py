import re

import matplotlib.pyplot as plt
import numpy as np
import xdeps as xd

from .rdmsignal import poly_fit, poly_val


class Col:
    def __init__(self, attr, rows):
        self.attr = attr
        self.rows = rows

    def __getitem__(self, k):
        attrs = [getattr(row, self.attr) for row in self.rows]
        return np.array([attr[k] for attr in attrs])

    def __repr__(self) -> str:
        return f"<Col {self.attr!r} {len(self.rows)} rows>"


class LHCSectionTable:
    def __init__(self, rows):
        self.rows = list(rows)
        self.strengths = Col("strengths", rows)
        self.params = Col("params", rows)
        self.knobs = Col("knobs", rows)

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
        if order == 0:  # Nearest
            return np.interp(x, xx, yy)
        if order == 1:
            # return scipy.interpolate.interp1d(xx, yy, kind="linear")(x)
            return np.interp(x, xx, yy)
        if order > 1:
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
            return np.array([ir[k] for ir in self.rows])

    def __iter__(self):
        return iter(self.rows)

    def __len__(self):
        return len(self.rows)

    def __repr__(self):
        if len(self) == 0:
            return "<Table: 0 rows>"
        cls = self.rows[0].__class__.__name__
        return f"<Table {cls}: {len(self)} rows>"


class LHCIRTable(LHCSectionTable):

    def interp(self, n, order=1, xaxis="id"):
        ir0 = self.rows[0]
        strengths = {
            k: self.interp_val(n, k, order, xaxis) for k in ir0.strengths
        }
        params = {k: self.interp_val(n, k, order, xaxis) for k in ir0.params}
        return ir0.__class__(strengths=strengths, params=params)

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
        xx = self[xaxis]
        if n == 1:
            kqxl = self[f"kqx.l{self.rows[0].irn}"] * brho
            kqxr = self[f"kqx.r{self.rows[0].irn}"] * brho
            ktqx1l = self[f"ktqx1.l{self.rows[0].irn}"] * brho
            ktqx1r = self[f"ktqx1.r{self.rows[0].irn}"] * brho
            ktqx2l = self[f"ktqx2.l{self.rows[1].irn}"] * brho
            ktqx2r = self[f"ktqx2.r{self.rows[1].irn}"] * brho
            kqx1l = kqxl + ktqx1l
            kqx1r = kqxr + ktqx1r
            kqx2l = kqxl + ktqx2l
            kqx2r = kqxr + ktqx2r
            ax.plot(xx, abs(kqx1l), label=f"kqx1.l{self.rows[0].irn}")
            ax.plot(xx, abs(kqx1r), label=f"kqx1.r{self.rows[0].irn}")
            ax.plot(xx, abs(kqx2l), label=f"kqx2.l{self.rows[1].irn}")
            ax.plot(xx, abs(kqx2r), label=f"kqx2.r{self.rows[1].irn}")
            ax.plot(xx, abs(kqxl), label=f"kqx3.l{self.rows[0].irn}")
            ax.plot(xx, abs(kqxr), label=f"kqx3.r{self.rows[0].irn}")
        else:
            for q in self.get_quads(n):
                if "t" not in q and np.all(self[q] < 0):
                    ax.plot(xx, -self[q] * brho, label=f"-{q}")
                else:
                    ax.plot(xx, self[q] * brho, label=q)
        ax.set_title(title)
        ax.set_xlabel(xaxis)
        ax.set_ylabel(r"k [$T/m$]")
        ax.legend()

    def plot_quad_fit(
        self,
        n,
        order=1,
        soft=True,
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
            fig, ax = plt.subplots(num=figname)
        xx = self[xaxis]
        for kname in self.get_quads(n):
            yy = self[kname]
            xx_fit = np.linspace(xx[0], xx[-1], 1000)
            yy_fit = self.interp_val(
                xx_fit, kname, order=order, xname=xaxis, soft=soft
            )
            ax.plot(xx, yy * brho, label=f"{kname}")
            ax.plot(xx_fit, yy_fit * brho, label=f"fit {kname}")
        ax.set_title(title)
        ax.set_xlabel(xaxis)
        ax.set_ylabel(r"k [$T/m$]")
        ax.legend()

    def plot_quads(
        self, xaxis="id", fig=None, title=None, figname=None, p0c=6.8e12
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
        return arc0.__class__(arc0.name, strengths=strengths, params=params)


class LHCOpticsTable(LHCSectionTable):
    def __init__(self, rows):
        self.rows = rows
        self.params = Col("params", rows)
        self.knobs = Col("knobs", rows)
        self.irs = [
            LHCIRTable(irlst) for irlst in zip(*[row.irs for row in self.rows])
        ]
        self.arcs = [
            LHCArcTable(arclst)
            for arclst in zip(*[row.arcs for row in self.rows])
        ]
        for ir in self.irs:
            setattr(self, ir.rows[0].name, ir)
        for arc in self.arcs:
            setattr(self, arc.rows[0].name, arc)

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
        return self

    def insert(self, i, row):
        self.rows.insert(i, row)
        return self

    def remove(self, row):
        self.rows.remove(row)
        return self

    def plot_quads(self):
        for ir in self.irs:
            ir.plot_quads()
