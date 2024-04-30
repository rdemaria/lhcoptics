import numpy as np
import matplotlib.pyplot as plt
import re
import xdeps as xd


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
