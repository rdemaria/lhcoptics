import numpy as np
import xdeps
import xtrack as xt

from .knob import Knob
from .model_madx import LHCMadxModel

import matplotlib.pyplot as plt


class SinglePassDispersion(xdeps.Action):
    def __init__(self, line, ele_start, ele_stop, backtrack=False, delta=1e-3):
        self.line = line
        self.ele_start = ele_start
        self.ele_stop = ele_stop
        self.delta = delta
        self.backtrack = backtrack
        self._pp = line.build_particles(delta=delta)

    def run(self):
        for nn in ["x", "px", "y", "py", "zeta", "delta", "at_element"]:
            setattr(self._pp, nn, 0)
        self._pp.delta = self.delta
        self.line.track(
            self._pp,
            ele_start=self.ele_start,
            ele_stop=self.ele_stop,
            backtrack=self.backtrack,
        )
        return {
            "d" + nn: getattr(self._pp, nn)[0] / self.delta
            for nn in ["x", "px", "y", "py"]
        }


def termlist(ex, lst=[]):
    if isinstance(ex, xdeps.refs.AddExpr):
        return lst + termlist(ex._lhs) + termlist(ex._rhs)
    if isinstance(ex, xdeps.refs.SubExpr):
        if isinstance(ex._rhs, xdeps.refs.MulExpr):
            ex = ex._lhs + (-1 * ex._rhs._lhs) * ex._rhs._rhs
        else:
            ex = ex._lhs + (-1) * ex._rhs
        return lst + termlist(ex._lhs) + termlist(ex._rhs)
    else:
        return [ex]


def pprint(expr):
    return "\n + ".join([str(t) for t in termlist(expr)])


class LHCXsuiteModel:
    _xt = xt

    def __init__(
        self,
        multiline=None,
        optics=None,
        settings=None,
        jsonfile=None,
        madxfile=None,
    ):
        self.multiline = multiline
        self.optics = optics
        self.settings = settings
        self.jsonfile = jsonfile
        self._var_values = multiline._xdeps_vref._owner
        self.vars = multiline._xdeps_vref
        self.mgr = multiline._xdeps_manager
        self.madxfile = madxfile
        self.sequence = {1: multiline.b1, 2: multiline.b2}
        self.b1.build_tracker()
        self.b2.build_tracker()
        self._aperture = None

    @classmethod
    def from_madxfile(cls, madxfile, sliced=False):
        """
        Create a LHCXsuiteModel from a MAD-X input file
        """
        madmodel = LHCMadxModel.from_madxfile(madxfile)

        self = cls.from_madx(madmodel.madx, sliced=sliced, madxfile=madxfile)
        self.madxfile = madxfile
        return self

    @classmethod
    def from_madx(cls, madx, sliced=False, madxfile=None):

        lines = xt.Multiline.from_madx(
            madx=madx, enable_layout_data=True, return_lines=True
        )
        lines["b1"] = lines.pop("lhcb1")
        lines["b2"] = lines.pop("lhcb2")
        lines["b1"].particle_ref = xt.Particles(
            mass0=xt.PROTON_MASS_EV, p0c=450e9
        )
        lines["b2"].particle_ref = xt.Particles(
            mass0=xt.PROTON_MASS_EV, p0c=450e9
        )
        lines["b1"].twiss_default["method"] = "4d"
        lines["b2"].twiss_default["method"] = "4d"
        lines["b1"].twiss_default["co_search_at"] = "ip7"
        lines["b2"].twiss_default["co_search_at"] = "ip7"
        lines["b1"].twiss_default["strengths"] = True
        lines["b2"].twiss_default["strengths"] = True
        lines["b1"].twiss_default["compute_chromatic_properties"] = False
        lines["b2"].twiss_default["compute_chromatic_properties"] = False
        if sliced:
            for ln, ll in list(lines.items()):
                ls = ll.copy()
                ls.slice_thick_elements(
                    slicing_strategies=[
                        xt.Strategy(slicing=None),
                        xt.Strategy(
                            slicing=xt.Uniform(8, mode="thick"), name="mb.*"
                        ),
                        xt.Strategy(
                            slicing=xt.Uniform(8, mode="thick"), name="mq.*"
                        ),
                    ]
                )
                lines[f"{ln}s"] = ls

        lhc = xt.Multiline(lines)
        out = cls(multiline=lhc, madxfile=madxfile)
        return out

    @classmethod
    def from_json(cls, jsonfile):
        import xtrack as xt

        lhc = xt.Multiline.from_json(jsonfile)
        return cls(lhc, jsonfile=jsonfile)

    def to_json(self, jsonfile="lhc.json"):
        self.multiline.to_json(jsonfile)

    @property
    def b1(self):
        return self.multiline.b1

    @property
    def b2(self):
        return self.multiline.b2

    @property
    def b1s(self):
        return self.multiline.b1s

    @property
    def b2s(self):
        return self.multiline.b2s

    def info(self, k):
        return self.multiline.vars[k]._info(limit=None)

    @property
    def p0c(self):
        return self.multiline.b1.particle_ref.p0c[0]

    @p0c.setter
    def p0c(self, value):
        self.multiline.b1.particle_ref.p0c = value
        self.multiline.b2.particle_ref.p0c = value

    def update_vars(self, strengths, verbose=False):
        for k, v in strengths.items():
            if verbose:
                if k in self and self[k] != v:
                    print(f"Update {k} from {self[k]:15.6g} to {v:15.6g}")
            self[k] = v

    def knob_delete_all(self):
        for vv in self._var_values:
            self[vv] = self[vv]

    def knob_check(self, knob, verbose=False):
        """
        Return True has the expeceted structure
        Return False has a different structure
        Return None if it does not exist
        """
        knobname = knob.name
        deps = self.mgr.rdeps.get(self.vars[knobname], {})
        if len(deps) == 0:
            if verbose:
                print(f"Missing knob {knobname}")
            return None
        depnames = {dep._key for dep in deps}
        if verbose:
            print(f"Check knob {knobname}")
            print(f"Dependencies {depnames!r}")
        if depnames != set(knob.weights.keys()):
            if verbose:
                print(
                    f"Dependencies do not match with {set(knob.weights.keys())}"
                )
            return False
        else:
            for depname in depnames:
                wname = f"{depname}_from_{knobname}"
                if wname not in self._var_values:
                    if verbose:
                        print(f"Missing weight {wname}")
                    return False
            return True

    def update_knobs(self, knobs, verbose=False, knobs_off=False):
        for k, knob in knobs.items():
            self.update_knob(knob, verbose=verbose, knobs_off=knobs_off)

    def update_knob(self, knob, verbose=False, knobs_off=False):
        """
        Update the model with the knob

        Check that the knob exists, that is k has dependent targets.
        If it exists, check that has the same structure
        else raise an error.
        If it does not exist, create it.

        """
        knobname = knob.name
        check = self.knob_check(knob)
        if check is False:
            self.knob_check(knob, verbose=verbose)
            raise ValueError(
                f"Knob {knobname} has different structure in {self}"
            )
        else:
            if not knobs_off:
                if verbose and knob.value != self[knobname]:
                    print(
                        f"Update {knobname} from {self[knobname]:15.6g} to {knob.value:15.6g}"
                    )
                self[knobname] = knob.value
            for wtarget, value in knob.weights.items():
                wname = f"{wtarget}_from_{knobname}"
                if verbose and wname in self and self[wname] != value:
                    print(
                        f"Update {wname} from {self[wname]:15.6g} to {value:15.6g}"
                    )
                self[wname] = value
                if check is None:
                    # if verbose:
                    #    print(f"Add expression {wtarget} += {wname}*{knobname}")
                    self.vars[wtarget] += (
                        self.vars[wname] * self.vars[knobname]
                    )

    def _erase_knob(self, knob):
        """
        Delete knobs and weights from the model, very unsafe, because it can break other knobs. To be used only in tests.
        """
        knobname = knob.name
        deps = list(self.mgr.rdeps.get(self.vars[knobname], {}))
        for dep in deps:
            wname = f"{dep._key}_from_{knobname}"
            self.mgr.set_value(dep, dep._get_value())
            if wname in self._var_values:
                del self._var_values[wname]

    def get_knob(self, knob):
        value = self._var_values[knob.name]
        weights = {}
        for wname in knob.weights:
            weights[wname] = self._var_values[f"{wname}_from_{knob.name}"]
        return Knob(knob.name, value, weights)

    def show_knob(self, knobname):
        print(f"Knob: {knobname} = {self[knobname]:15.6g}")
        for deps in self.mgr.rdeps.get(self.vars[knobname], {}):
            print("Target:", deps._key)
            print("     Expr:", pprint(deps._expr))
            wname = f"{deps._key}_from_{knobname}"
            if wname in self:
                print(f"    Weight {wname} = {self[wname]:15.6g}")

    def get_knob_by_weight_names(self, name):
        weights = {}
        value = self._var_values[name]
        wname = f"_from_{name}"
        for k in self._var_values:
            if k.endswith(wname):
                weights[k] = self._var_values[k]
        return Knob(name, value, weights)

    def get_knob_by_probing(self, name):
        weights = {}
        oldvars = self._var_values.copy()
        oldvalue = self._var_values[name]
        self._var_values[name] = oldvalue + 1
        for k in self._var_values:
            vnew = self._var_values[k]
            if hasattr(vnew, "__sub__"):
                dvar = self._var_values[k] - oldvars[k]
                if dvar != 0:
                    weights[k] = dvar
        self._var_values[name] = oldvalue
        return Knob(name, oldvalue, weights)

    def update(self, src):
        if hasattr(src, "strengths"):
            self.update_vars(src.strengths)
        else:
            self.update_vars(src)
        if hasattr(src, "knobs"):
            self.update_knobs(src.knobs)

    def twiss_open(self, start, end, init, beam, strengths=True):
        # line=self.sequence[beam]
        # aux=line.element_names[0]
        # line.cycle(init.element_name)
        tw = self.sequence[beam].twiss(start=start, end=end, init=init, strengths=strengths)
        # line.cycle(aux)
        return tw

    def twiss_init(self, start, end, init_at, beam):
        # line=self.sequence[beam]
        # aux=line.element_names[0]
        # self.sequence[beam].cycle(init_at)
        init = (
            self.sequence[beam]
            .twiss(start=start, end=end, init="periodic")
            .get_twiss_init(init_at)
        )
        # line.cycle(aux)
        return init

    def twiss(
        self,
        start=None,
        end=None,
        init=None,
        init_at=None,
        beam=None,
        full=True,
        chrom=False,
        strengths=True,
    ):
        """
        Examples
        - twiss(): periodic solution, full machine, start/end of the line
        - twiss(start="ip8", end="ip2"): as before by data at the start/end of the line
        - twiss(start="ip8", end="ip2", full=True): full machine, start/end of the line
        - twiss(start="ip8", end="ip2", init_at="ip1"): periodic solution, full machine, start/end of the line, s,mux,muy=0 at ip1
        - twiss(start="ip8", end="ip2", init="init"):


        NB: Still fails when full=False and boundaries are reversed w.r.t the line orde
        """
        if beam is None:
            return [
                self.twiss(
                    start,
                    end,
                    init,
                    init_at=init_at,
                    full=full,
                    chrom=chrom,
                    beam=1,
                    strengths=strengths,
                ),
                self.twiss(
                    start,
                    end,
                    init,
                    init_at=init_at,
                    full=full,
                    chrom=chrom,
                    beam=2,
                    strengths=strengths,
                ),
            ]
        if beam == 1:
            line_start = self.sequence[beam].element_names[0]
            line_end = self.sequence[beam].element_names[-1]
        else:
            line_start = self.sequence[beam].element_names[-1]
            line_end = self.sequence[beam].element_names[0]
        if start is None:
            start = line_start
        if end is None:
            end = line_end
        if full:
            boundary_start = line_start
            boundary_end = line_end
        else:
            boundary_start = start
            boundary_end = end
        if init is None:
            if init_at is None:
                init_at = start
            init = self.twiss_init(boundary_start, boundary_end, init_at, beam)
            init.s = 0
            init.mux = 0
            init.muy = 0
        return self.twiss_open(start, end, init, beam, strengths=strengths)

    def copy(self):
        return self.__class__(
            multiline=self.multiline.copy(),
            settings=self.settings,
            jsonfile=self.jsonfile,
            madxfile=self.madxfile,
        )

    def cycle(self, element):
        self.b1.cycle(element,inplace=True)
        self.b2.cycle(element,inplace=True)

    def diff(self, other):
        allk = set(self._var_values.keys()) | set(other._var_values.keys())
        for k in allk:
            if k not in self._var_values:
                print(f"{k:20} {other._var_values[k]:15.6g} only in other")
            elif k not in other._var_values:
                print(f"{k:20} {self._var_values[k]:15.6g} only in self")
            elif self._var_values[k] != other._var_values[k]:
                print(
                    f"{k:20} {self._var_values[k]:15.6g} {other._var_values[k]:15.6g}"
                )

    def match(self, *args, **kwargs):
        return self.multiline.match(*args, **kwargs)

    def match_knob(self, *args, **kwargs):
        return self.multiline.match_knob(*args, **kwargs)

    def __getitem__(self, key):
        return self._var_values[key]

    def __setitem__(self, key, value):
        self.vars[key] = value

    def __contains__(self, key):
        return key in self._var_values

    def __repr__(self):
        if self.madxfile is not None:
            return f"<LHCXsuiteModel {self.madxfile!r}>"
        elif self.jsonfile is not None:
            return f"<LHCXsuiteModel {self.jsonfile!r}>"
        else:
            return f"<LHCXsuiteModel {id(self)}>"

    def plot_beamsize(
        self,
        beam=None,
        emit_n=2.5e-6,
        p0c=None,
        bbeat=1.1,
        co_error=2e-3,
        ndisp_err=0.1,
        delta_err=2e-4,
        nsigma=12,
        survey=False,
    ):
        if beam is None:
            return [
                self.plot_beamsize(
                    beam=1,
                    emit_n=emit_n,
                    p0c=p0c,
                    bbeat=bbeat,
                    co_error=co_error,
                    ndisp_err=ndisp_err,
                    delta_err=delta_err,
                    nsigma=nsigma,
                    survey=survey,
                ),
                self.plot_beamsize(
                    beam=2,
                    emit_n=emit_n,
                    p0c=p0c,
                    bbeat=bbeat,
                    co_error=co_error,
                    ndisp_err=ndisp_err,
                    delta_err=delta_err,
                    nsigma=nsigma,
                    survey=survey,
                ),
            ]
        line = self.sequence[beam]
        tw = line.twiss(strengths=False)
        if p0c is None:
            p0c = self.p0c
        ex = emit_n / p0c * 0.9382720813e9
        ey = ex
        dx_err = 2.0 * np.sqrt(tw.betx / 170) * ndisp_err
        dy_err = 2.0 * np.sqrt(tw.bety / 170) * ndisp_err
        dx = tw.dx + dx_err
        dy = tw.dy + dy_err
        #print(f"dx_err={dx_err.max()} dy_err={dy_err.max()}")
        #print(f"dx={dx_err.max()} dy={dy.max()}")
        #print(f"ex={ex} ey={ey}")
        sx = (
            nsigma * bbeat * np.sqrt(tw.betx * ex)
            + abs(dx) * delta_err
            + co_error
        )
        sy = (
            nsigma * bbeat * np.sqrt(tw.bety * ey)
            + abs(dy) * delta_err
            + co_error
        )
        x=tw.x
        y=tw.y
        if survey:
            su=self.get_survey_flat(beam)
            x+=su.X
            y+=su.Y
        xp = x + sx
        xm = x - sx
        yp = y + sy
        ym = y - sy
        if not survey:
            fig, (ax1, ax2) = plt.subplots(2, 1, num=f"aperture{beam}", clear=True)
        else:
            if beam==1:
                fig, (ax1, ax2) = plt.subplots(2, 1, num=f"aperture{beam}", figsize=(12, 6))
            else:
                ax1, ax2 = plt.gcf().get_axes()
        color = "b" if beam == 1 else "r"
        ax1.plot(tw.s, x, label="x", color=color)
        ax2.plot(tw.s, y, label="y", color=color)
        ax1.set_ylabel("x [m]")
        ax2.set_ylabel("y [m]")
        ax1.fill_between(
            tw.s, xp, xm, alpha=0.5, color=color, label=f"{nsigma} sigma"
        )
        ax2.fill_between(
            tw.s, yp, ym, alpha=0.5, color=color, label=f"{nsigma} sigma"
        )
        return ax1, ax2

    def plot_aperture(self, beam=None):
        if beam is None:
            return [self.plot_aperture(beam=1), self.plot_aperture(beam=2)]
        su = self.get_survey_flat(beam)
        line = self.sequence[beam]
        fig, (ax1, ax2) = plt.subplots(2, 1, num=f"aperture{beam}", clear=True)
        print(su,line, fig, ax1, ax2)
        raise NotImplementedError("Not implemented")

    def get_survey_flat(self, beam=None):
        if beam is None:
            return [self.get_survey_flat(beam=1), self.get_survey_flat(beam=2)]
        line = self.sequence[beam].copy()
        line.build_tracker()
        for name, elem in line.element_dict.items():
            if name.startswith("mb."):
                elem.h = 0
        su = line.survey(reverse=False) #needs to force it because not supported
        if beam == 2:
            su = su.reverse()
        return su

    def get_survey(self, beam=None):
        if beam is None:
            return [self.get_survey(beam=1), self.get_survey(beam=2)]
        if beam == 1:
            return self.sequence[beam].survey()
        else:
            return self.sequence[beam].survey(reverse=False).reverse()

    def plot_survey(self, beam=None):
        if beam is None:
            return [self.plot_survey(beam=1), self.plot_survey(beam=2)]
        su = self.get_survey(beam)

        plt.figure("LHC Survey",figsize=(6,6))
        ax=plt.subplot(111)
        color="b" if beam==1 else "r"
        ax.plot(su.Z, su.X, label=f"Beam {beam}",color=color)
        ax.set_xlabel("Z [m]")
        ax.set_ylabel("X [m]")
        suips=su.rows["ip[1-8]"].cols["name Z X"]
        for name,x,y in suips.rows:
            plt.text(x,y,name.upper(),color="black")
        return self

    def plot_survey_flat(self, figsize=(12, 3)):
        su1 = self.get_survey_flat(beam=1)
        su2 = self.get_survey_flat(beam=2)
        plt.figure("LHC Survey Flat",figsize=figsize)
        plt.plot(su1.s, su1.X,label="Beam 1", color="blue")
        plt.plot(su2.s, su2.X,label="Beam 2", color="red")
        plt.xlabel("S [m]")
        plt.ylabel("X [m]")
        suips=su1.rows["ip[1-8]"].cols["name s X"]
        for name,x,y in suips.rows:
            plt.text(x,0,name.upper(),color="black")
        plt.tight_layout()
        return self

    def slice(self, slices=8):
        for beam in [1, 2]:
            line = self.sequence[beam]
            line.slice_thick_elements(
                slicing_strategies=[
                    xt.Strategy(slicing=None),
                    xt.Strategy(
                        slicing=xt.Uniform(slices, mode="thick"), name="mb.*"
                    ),
                    xt.Strategy(
                        slicing=xt.Uniform(slices, mode="thick"), name="mq.*"
                    ),
                ]
            )

    def make_aperture(self):
        from .aperture import LHCAperture

        return LHCAperture.from_xsuite_model(self)

    def aperture(
        self,
        beam,
        emit=2.5e-6,
        p0c=None,
        bbeat=1.1,
        delta_err=2e-4,
        ndisp_err=0.1,
        co_error=2e-3,
        nsigma=12,
    ):
        if self._aperture is None:
            self._aperture = self.make_aperture()
        tw = self.sequence[beam].twiss(strengths=False)
        ap = self._aperture.apertures[beam - 1]
        if p0c is None:
            p0c = tw.particle_on_co.p0c
        bsx = np.sqrt(tw.betx * emit / p0c * 0.938e9 + (tw.dx * delta_err) ** 2)
        xap = (
            ap.offset[:, 0] + tw.x
        )  # position of the beam with respect to the aperture
        ap_xmarg = ap.bbox[:, 0] - abs(xap) - co_error
        ap_x = ap_xmarg / bsx
        tw["ap_bsx"] = bsx
        tw["ap_x"] = ap_x
        tw["ap_xmarg"] = ap_xmarg
        return tw.rows[ap.profile != -1]


def test_coupling_knobs(collider):
    line = collider.b1
    # Check orthogonality
    line.vars["cmrs.b1_op"] = 1e-3
    line.vars["cmis.b1_op"] = 1e-3
    assert np.isclose(
        collider.b1.twiss().c_minus / np.sqrt(2), 1e-3, rtol=0, atol=1.5e-5
    )

    line.vars["cmrs.b2_op"] = 1e-3
    line.vars["cmis.b2_op"] = 1e-3
    assert np.isclose(
        collider.b2.twiss().c_minus / np.sqrt(2), 1e-3, rtol=0, atol=1.5e-5
    )


def test_chroma_knobs(collider):
    # Test the knobs

    # Correct to zero
    collider.b1.match(
        vary=xt.VaryList(["dqpx.b1_op", "dqpy.b1_op"], step=1e-4),
        targets=xt.TargetSet(dqx=0.0, dqy=0.0, tol=1e-4),
    )
    collider.b2.match(
        vary=xt.VaryList(["dqpx.b2_op", "dqpy.b2_op"], step=1e-4),
        targets=xt.TargetSet(dqx=0.0, dqy=0.0, tol=1e-4),
    )

    # Apply deltas
    collider.vars["dqpx.b1_op"] += 2
    collider.vars["dqpy.b1_op"] += 4
    collider.vars["dqpx.b2_op"] += 3
    collider.vars["dqpy.b2_op"] += 5

    twtest = collider.twiss()

    return twtest.dqx, twtest.dqy
