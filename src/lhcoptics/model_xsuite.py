import numpy as np
import xdeps as xd
import xtrack as xt

from .knob import Knob
from .model_madx import LHCMadxModel


class SinglePassDispersion(xd.Action):
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

    @property
    def p0c(self):
        return self.multiline.b1.p0c

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

    def twiss_open(self, start, end, init, beam):
        # line=self.sequence[beam]
        # aux=line.element_names[0]
        # line.cycle(init.element_name)
        tw = self.sequence[beam].twiss(start=start, end=end, init=init)
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
                ),
                self.twiss(
                    start,
                    end,
                    init,
                    init_at=init_at,
                    full=full,
                    chrom=chrom,
                    beam=2,
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
        return self.twiss_open(start, end, init, beam)

    def copy(self):
        return self.__class__(
            multiline=self.multiline.copy(),
            settings=self.settings,
            jsonfile=self.jsonfile,
            madxfile=self.madxfile,
        )

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


def make_coupling_knobs(collider):
    mqs_circuits_4_quads = {}
    mqs_circuits_2_quads = {}
    mqs_circuits_4_quads["b1"] = [
        "kqs.a23b1",
        "kqs.a45b1",
        "kqs.a67b1",
        "kqs.a81b1",
    ]
    mqs_circuits_2_quads["b1"] = [
        "kqs.l2b1",
        "kqs.l4b1",
        "kqs.l6b1",
        "kqs.l8b1",
        "kqs.r1b1",
        "kqs.r3b1",
        "kqs.r5b1",
        "kqs.r7b1",
    ]

    mqs_circuits_4_quads["b2"] = [
        "kqs.a12b2",
        "kqs.a34b2",
        "kqs.a56b2",
        "kqs.a78b2",
    ]

    mqs_circuits_2_quads["b2"] = [
        "kqs.l1b2",
        "kqs.l3b2",
        "kqs.l5b2",
        "kqs.l7b2",
        "kqs.r2b2",
        "kqs.r4b2",
        "kqs.r6b2",
        "kqs.r8b2",
    ]

    # see Eq. 47 in https://cds.cern.ch/record/522049/files/lhc-project-report-501.pdf
    class ActionCmin(xt.Action):
        def __init__(self, line):
            self.line = line

        def run(self):
            tw = self.line.twiss(strengths=True)
            k1sl = tw["k1sl"]
            c_min = (
                1
                / (2 * np.pi)
                * np.sum(
                    k1sl
                    * np.sqrt(tw.betx * tw.bety)
                    * np.exp(1j * 2 * np.pi * (tw.mux - tw.muy))
                )
            )
            return {"c_min_re": c_min.real, "c_min_im": c_min.imag}

    act_cmin_b1 = ActionCmin(collider["b1"])
    act_cmin_b2 = ActionCmin(collider["b2"])

    for nn in (
        mqs_circuits_4_quads["b1"]
        + mqs_circuits_2_quads["b1"]
        + mqs_circuits_4_quads["b2"]
        + mqs_circuits_2_quads["b2"]
    ):
        collider.vars["old_" + nn] = collider.vars[nn]._expr
        collider.vars[nn] = collider.vars[nn]._value

    optimizers = {"b1": {}, "b2": {}}

    c_min_match = 1e-4
    for bname in ["b1", "b2"]:
        line = collider[f"{bname}"]
        act_cmin = ActionCmin(line)

        assert np.abs(act_cmin.run()["c_min_re"]) < 1e-6
        assert np.abs(act_cmin.run()["c_min_im"]) < 1e-6

        opt_re = line.match_knob(
            run=False,
            knob_name=f"cmrs.{bname}_op",
            knob_value_start=0,
            knob_value_end=c_min_match,
            vary=[
                xt.VaryList(mqs_circuits_2_quads[bname], step=5e-5),
                xt.VaryList(mqs_circuits_4_quads[bname], step=5e-5, weight=2),
            ],
            targets=[
                act_cmin.target("c_min_re", value=c_min_match, tol=1e-8),
                act_cmin.target("c_min_im", value=0, tol=1e-8),
            ],
        )
        opt_re.solve()
        opt_re.generate_knob()
        optimizers[bname]["re"] = opt_re

        opt_im = line.match_knob(
            run=False,
            knob_name=f"cmis.{bname}_op",
            knob_value_start=0,
            knob_value_end=c_min_match,
            vary=[
                xt.VaryList(mqs_circuits_2_quads[bname], step=5e-5),
                xt.VaryList(mqs_circuits_4_quads[bname], step=5e-5, weight=2),
            ],
            targets=[
                act_cmin.target("c_min_re", value=0, tol=1e-8),
                act_cmin.target("c_min_im", value=c_min_match, tol=1e-8),
            ],
        )
        opt_im.solve()
        opt_im.generate_knob()
        optimizers[bname]["im"] = opt_im


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


def make_chroma_knobs(collider):
    ms_circuits = {}
    ms_circuits["b1"] = [
        "ksf1.a12b1",
        "ksf1.a23b1",
        "ksf1.a34b1",
        "ksf1.a45b1",
        "ksf1.a56b1",
        "ksf1.a67b1",
        "ksf1.a78b1",
        "ksf1.a81b1",
        "ksf2.a12b1",
        "ksf2.a23b1",
        "ksf2.a34b1",
        "ksf2.a45b1",
        "ksf2.a56b1",
        "ksf2.a67b1",
        "ksf2.a78b1",
        "ksf2.a81b1",
        "ksd1.a12b1",
        "ksd1.a23b1",
        "ksd1.a34b1",
        "ksd1.a45b1",
        "ksd1.a56b1",
        "ksd1.a67b1",
        "ksd1.a78b1",
        "ksd1.a81b1",
        "ksd2.a12b1",
        "ksd2.a23b1",
        "ksd2.a34b1",
        "ksd2.a45b1",
        "ksd2.a56b1",
        "ksd2.a67b1",
        "ksd2.a78b1",
        "ksd2.a81b1",
    ]

    ms_circuits["b2"] = [
        "ksf1.a12b2",
        "ksf1.a23b2",
        "ksf1.a34b2",
        "ksf1.a45b2",
        "ksf1.a56b2",
        "ksf1.a67b2",
        "ksf1.a78b2",
        "ksf1.a81b2",
        "ksf2.a12b2",
        "ksf2.a23b2",
        "ksf2.a34b2",
        "ksf2.a45b2",
        "ksf2.a56b2",
        "ksf2.a67b2",
        "ksf2.a78b2",
        "ksf2.a81b2",
        "ksd1.a12b2",
        "ksd1.a23b2",
        "ksd1.a34b2",
        "ksd1.a45b2",
        "ksd1.a56b2",
        "ksd1.a67b2",
        "ksd1.a78b2",
        "ksd1.a81b2",
        "ksd2.a12b2",
        "ksd2.a23b2",
        "ksd2.a34b2",
        "ksd2.a45b2",
        "ksd2.a56b2",
        "ksd2.a67b2",
        "ksd2.a78b2",
        "ksd2.a81b2",
    ]

    for nn in ms_circuits["b1"] + ms_circuits["b2"]:
        collider.vars["old_" + nn] = collider.vars[nn]._expr
        collider.vars[nn] = collider.vars[nn]._value

    optimizers = {"b1": {}, "b2": {}}
    d_chrom_match = 0.5
    for bname in ["b1", "b2"]:
        tw = collider[f"{bname}"].twiss(compute_chromatic_properties=True)
        opt_qpx = collider[f"{bname}"].match_knob(
            compute_chromatic_properties=True,
            knob_name=f"dqpx.{bname}_op",
            knob_value_start=0.0,
            knob_value_end=d_chrom_match,
            run=False,
            vary=xt.VaryList(ms_circuits[bname], step=1e-5),
            targets=[
                xt.Target("dqx", tw.dqx + d_chrom_match, tol=1e-4),
                xt.Target("dqy", tw.dqy, tol=1e-4),
            ],
        )

        opt_qpx.solve()
        opt_qpx.generate_knob()
        optimizers[bname]["qpy"] = opt_qpx

        opt_qpy = collider[f"{bname}"].match_knob(
            compute_chromatic_properties=True,
            knob_name=f"dqpy.{bname}_op",
            knob_value_start=0.0,
            knob_value_end=d_chrom_match,
            run=False,
            vary=xt.VaryList(ms_circuits[bname], step=1e-5),
            targets=[
                xt.Target("dqx", tw.dqx, tol=1e-4),
                xt.Target("dqy", tw.dqy + d_chrom_match, tol=1e-4),
            ],
        )

        opt_qpy.solve()
        opt_qpy.generate_knob()
        optimizers[bname]["qpy"] = opt_qpy


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
