from .model_madx import LHCMadModel
from .knob import Knob

import xdeps

import xtrack as xt
import numpy as np


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
    @classmethod
    def from_madxfile(cls, madxfile, sliced=False):
        """
        Create a LHCXsuiteModel from a MAD-X input file
        """
        madmodel = LHCMadModel.from_madxfile(madxfile)

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
        lines["b1"].build_tracker()
        lines["b2"].build_tracker()
        out = cls(multiline=lhc, madxfile=madxfile)
        return out

    @classmethod
    def from_json(cls, jsonfile):
        import xtrack as xt

        lhc = xt.Multiline.from_json(jsonfile)
        return cls(lhc, jsonfile=jsonfile)

    def to_json(self, jsonfile="lhc.json"):
        self.multiline.to_json(jsonfile)

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

    def __repr__(self):
        if self.madxfile is not None:
            return f"<LHCXsuiteModel {self.madxfile!r}>"
        elif self.jsonfile is not None:
            return f"<LHCXsuiteModel {self.jsonfile!r}>"
        else:
            return f"<LHCXsuiteModel {id(self)}>"

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

    def __getitem__(self, key):
        return self._var_values[key]

    def __contains__(self, key):
        return key in self._var_values

    def __setitem__(self, key, value):
        self.vars[key] = value

    def update_vars(self, strengths, verbose=False):
        for k, v in strengths.items():
            if verbose:
                print(f"{k:20} {v:15.6g}")
            self[k] = v

    def update_knobs(self, knobs, verbose=False):
        for k, knob in knobs.items():
            if verbose:
                print(f"Update knob {k:20} {knob.value:15.6g}")
            self.update_knob(knob)

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

    def update_knob(self, knob, verbose=False):
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
            raise ValueError(f"Knob {knobname} has different structure")
        else:
            self[knobname] = knob.value
            for wtarget, value in knob.weights.items():
                wname = f"{wtarget}_from_{knobname}"
                self[wname] = value
                if check is None:
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

    def get_knob_by_naming(self, name):
        weights = {}
        wname = f"_from_{name}"
        for k in self._var_values:
            if k.endswith(wname):
                weights[k] = self._var_values[k]
        value = self._var_values[name]
        return Knob(name, value, weights)

    def update(self, src):
        if hasattr(src, "strengths"):
            self.update_vars(src.strengths)
        else:
            self.update_vars(src)
        if hasattr(src, "knobs"):
            self.update_knobs(src.knobs)

    def twiss(
        self,
        start=None,
        end=None,
        init=None,
        beam=None,
        full=False,
        chrom=False,
    ):
        if beam is None:
            return self.twiss(
                start=start, end=end, init=init, beam=1, full=full, chrom=chrom
            ), self.twiss(
                start=start, end=end, init=init, beam=2, full=full, chrom=chrom
            )
        line = self.sequence[beam]
        startout = line.element_names[0] if start is None else start
        endout = line.element_names[-1] if end is None else end
        startmac = line.element_names[0] if full is None else start
        endmac = line.element_names[-1] if full is None else end
        if init is None:
            tw = line.twiss(start=startmac, end=endmac, init="periodic")
            if full:
                tw = tw.rows[startout:endout]  # can fail because of cycle
        else:
            tw = line.twiss(start=startout, end=endout, init=init)

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


def mk_tune_knobs(collider):
    mqt_circuits = {}
    mqt_circuits["b1"] = [
        "kqtf.a12b1",
        "kqtf.a23b1",
        "kqtf.a34b1",
        "kqtf.a45b1",
        "kqtf.a56b1",
        "kqtf.a67b1",
        "kqtf.a78b1",
        "kqtf.a81b1",
        "kqtd.a12b1",
        "kqtd.a23b1",
        "kqtd.a34b1",
        "kqtd.a45b1",
        "kqtd.a56b1",
        "kqtd.a67b1",
        "kqtd.a78b1",
        "kqtd.a81b1",
    ]

    mqt_circuits["b2"] = [
        "kqtf.a12b2",
        "kqtf.a23b2",
        "kqtf.a34b2",
        "kqtf.a45b2",
        "kqtf.a56b2",
        "kqtf.a67b2",
        "kqtf.a78b2",
        "kqtf.a81b2",
        "kqtd.a12b2",
        "kqtd.a23b2",
        "kqtd.a34b2",
        "kqtd.a45b2",
        "kqtd.a56b2",
        "kqtd.a67b2",
        "kqtd.a78b2",
        "kqtd.a81b2",
    ]

    for nn in mqt_circuits["b1"] + mqt_circuits["b2"]:
        collider.vars["old_" + nn] = collider.vars[nn]._expr
        collider.vars[nn] = collider.vars[nn]._value

    optimizers = {"b1": {}, "b2": {}}
    dq_match = 1e-3
    for bname in ["b1", "b2"]:
        tw = collider[f"lhc{bname}"].twiss()
        opt_qx = collider[f"lhc{bname}"].match_knob(
            knob_name=f"dqx.{bname}",
            knob_value_start=0.0,
            knob_value_end=dq_match,
            run=False,
            vary=xt.VaryList(mqt_circuits[bname], step=1e-8),
            targets=[
                tw.target("qx", tw.qx + dq_match, tol=1e-7),
                tw.target("qy", tw.qy, tol=1e-7),
            ],
        )

        opt_qx.solve()
        opt_qx.generate_knob()
        optimizers[bname]["qx"] = opt_qx

        opt_qy = collider[f"lhc{bname}"].match_knob(
            knob_name=f"dqy.{bname}",
            knob_value_start=0.0,
            knob_value_end=dq_match,
            run=False,
            vary=xt.VaryList(mqt_circuits[bname], step=1e-8),
            targets=[
                tw.target("qx", tw.qx, tolerance=1e-7),
                tw.target("qy", tw.qy + dq_match, tolerance=1e-7),
            ],
        )

        opt_qy.solve()
        opt_qy.generate_knob()
        optimizers[bname]["qy"] = opt_qy

    return optimizers


def make_coupling_knob(collider):
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

    act_cmin_b1 = ActionCmin(collider["lhcb1"])
    act_cmin_b2 = ActionCmin(collider["lhcb2"])

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
        line = collider[f"lhc{bname}"]
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

    # Check orthogonality
    line.vars["cmrs.b1_op"] = 1e-3
    line.vars["cmis.b1_op"] = 1e-3
    assert np.isclose(
        collider.lhcb1.twiss().c_minus / np.sqrt(2), 1e-3, rtol=0, atol=1.5e-5
    )

    line.vars["cmrs.b2_op"] = 1e-3
    line.vars["cmis.b2_op"] = 1e-3
    assert np.isclose(
        collider.lhcb2.twiss().c_minus / np.sqrt(2), 1e-3, rtol=0, atol=1.5e-5
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
        tw = collider[f"lhc{bname}"].twiss()
        opt_qpx = collider[f"lhc{bname}"].match_knob(
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

        opt_qpy = collider[f"lhc{bname}"].match_knob(
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

    if False:
        # Test the knobs

        # Correct to zero
        collider.lhcb1.match(
            vary=xt.VaryList(["dqpx.b1_op", "dqpy.b1_op"], step=1e-4),
            targets=xt.TargetSet(dqx=0.0, dqy=0.0, tol=1e-4),
        )
        collider.lhcb2.match(
            vary=xt.VaryList(["dqpx.b2_op", "dqpy.b2_op"], step=1e-4),
            targets=xt.TargetSet(dqx=0.0, dqy=0.0, tol=1e-4),
        )

        # Apply deltas
        collider.vars["dqpx.b1_op"] += 2
        collider.vars["dqpy.b1_op"] += 4
        collider.vars["dqpx.b2_op"] += 3
        collider.vars["dqpy.b2_op"] += 5

        twtest = collider.twiss()


def make_orbit_knobs(collider):

    dct = {'on_sep1_h': {'acbch5.l1b2': -1.41918166577e-05,
    'acbch5.r1b1': 1.64239081776e-05,
    'acbch6.l1b1': -3.4974463131e-06,
    'acbch6.r1b2': 3.58269623014e-06,
    'acbxh1.l1': 8e-06,
    'acbxh1.r1': 8e-06,
    'acbxh2.l1': 8e-06,
    'acbxh2.r1': 8e-06,
    'acbxh3.l1': 8e-06,
    'acbxh3.r1': 8e-06,
    'acbyhs4.l1b1': 1.36297113998e-05,
    'acbyhs4.l1b2': 9.3708594619e-06,
    'acbyhs4.r1b1': -1.24156012491e-05,
    'acbyhs4.r1b2': -1.39157673175e-05},
    'on_sep2h': {'acbch5.r2b2': 0.0,
    'acbchs5.r2b1': 7.205639217199999e-06,
    'acbchs5.r2b2': 5.39723817366e-06,
    'acbxh1.l2': 0.0,
    'acbxh1.r2': 9e-06,
    'acbxh2.l2': 1.35e-05,
    'acbxh2.r2': 9e-06,
    'acbxh3.l2': 1.35e-05,
    'acbxh3.r2': 9e-06,
    'acbyh4.l2b2': 0.0,
    'acbyh4.r2b1': 0.0,
    'acbyh5.l2b1': 0.0,
    'acbyhs4.l2b1': 8.03040506369e-06,
    'acbyhs4.l2b2': -6.55793522704e-06,
    'acbyhs4.r2b1': 6.626871073399999e-06,
    'acbyhs4.r2b2': -1.8905667239800002e-05,
    'acbyhs5.l2b1': -1.5501793291099998e-06,
    'acbyhs5.l2b2': -6.9773770495e-06},
    'on_sep5_v': {'acbcv5.l5b1': -1.02875467053e-05,
    'acbcv5.r5b2': 9.06319153047e-06,
    'acbcv6.l5b2': -1.05495498088e-05,
    'acbcv6.r5b1': 9.86349425437e-06,
    'acbxv1.l5': 8e-06,
    'acbxv1.r5': 8e-06,
    'acbxv2.l5': 8e-06,
    'acbxv2.r5': 8e-06,
    'acbxv3.l5': 8e-06,
    'acbxv3.r5': 8e-06,
    'acbyvs4.l5b1': 2.32108182498e-05,
    'acbyvs4.l5b2': 4.25628214283e-06,
    'acbyvs4.r5b1': -2.39208859279e-06,
    'acbyvs4.r5b2': -2.10127379873e-05},
    'on_sep8v': {'acbcv5.l8b2': 0.0,
    'acbcvs5.l8b1': 4.704301035009999e-06,
    'acbcvs5.l8b2': 4.59473378142e-06,
    'acbxv1.l8': 9e-06,
    'acbxv1.r8': 9e-06,
    'acbxv2.l8': 9e-06,
    'acbxv2.r8': 9e-06,
    'acbxv3.l8': 9e-06,
    'acbxv3.r8': 9e-06,
    'acbyv4.l8b1': 0.0,
    'acbyv4.r8b2': 0.0,
    'acbyv5.r8b1': 0.0,
    'acbyvs4.l8b1': 8.407397301870002e-06,
    'acbyvs4.l8b2': -1.69056856716e-05,
    'acbyvs4.r8b1': 1.745385752e-05,
    'acbyvs4.r8b2': -8.443972688370001e-06,
    'acbyvs5.r8b1': -4.33151486387e-06,
    'acbyvs5.r8b2': -4.72205870866e-06},
    'on_sep1_v': {'acbcv5.l1b1': -1.41681624872e-05,
    'acbcv5.r1b2': 1.64237024066e-05,
    'acbcv6.l1b2': -3.47632625045e-06,
    'acbcv6.r1b1': 3.6084306567e-06,
    'acbxv1.l1': -8e-06,
    'acbxv1.r1': -8e-06,
    'acbxv2.l1': -8e-06,
    'acbxv2.r1': -8e-06,
    'acbxv3.l1': -8e-06,
    'acbxv3.r1': -8e-06,
    'acbyvs4.l1b1': 9.37344313262e-06,
    'acbyvs4.l1b2': 1.36566804063e-05,
    'acbyvs4.r1b1': -1.38925278577e-05,
    'acbyvs4.r1b2': -1.24152779217e-05},
    'on_sep2v': {'acbcv5.r2b1': 0.0,
    'acbcvs5.r2b1': -5.43938795831e-06,
    'acbcvs5.r2b2': -7.28659552868e-06,
    'acbxv1.l2': 9e-06,
    'acbxv1.r2': 9e-06,
    'acbxv2.l2': 9e-06,
    'acbxv2.r2': 9e-06,
    'acbxv3.l2': 9e-06,
    'acbxv3.r2': 9e-06,
    'acbyv4.l2b1': 0.0,
    'acbyv4.r2b2': 0.0,
    'acbyv5.l2b2': 0.0,
    'acbyvs4.l2b1': 9.67871365304e-06,
    'acbyvs4.l2b2': -1.69729431706e-05,
    'acbyvs4.r2b1': 1.87114720092e-05,
    'acbyvs4.r2b2': -6.93506266163e-06,
    'acbyvs5.l2b1': 4.8484845704e-06,
    'acbyvs5.l2b2': 4.34181842841e-06},
    'on_sep5_h': {'acbch5.l5b2': -1.03447685685e-05,
    'acbch5.r5b1': 9.05992577644e-06,
    'acbch6.l5b1': -1.04146390652e-05,
    'acbch6.r5b2': 1.00099005116e-05,
    'acbxh1.l5': -8e-06,
    'acbxh1.r5': -8e-06,
    'acbxh2.l5': -8e-06,
    'acbxh2.r5': -8e-06,
    'acbxh3.l5': -8e-06,
    'acbxh3.r5': -8e-06,
    'acbyhs4.l5b1': 3.13282618249e-06,
    'acbyhs4.l5b2': 2.28458183833e-05,
    'acbyhs4.r5b1': -2.13925054414e-05,
    'acbyhs4.r5b2': -3.49531196308e-06},
    'on_sep8h': {'acbch5.l8b1': 0.0,
    'acbchs5.l8b1': -4.40800786381e-06,
    'acbchs5.l8b2': -4.57801904082e-06,
    'acbxh1.l8': 9e-06,
    'acbxh1.r8': 9e-06,
    'acbxh2.l8': 9e-06,
    'acbxh2.r8': 9e-06,
    'acbxh3.l8': 9e-06,
    'acbxh3.r8': 9e-06,
    'acbyh4.l8b2': 0.0,
    'acbyh4.r8b1': 0.0,
    'acbyh5.r8b2': 0.0,
    'acbyhs4.l8b1': 1.69035964274e-05,
    'acbyhs4.l8b2': -8.43572526061e-06,
    'acbyhs4.r8b1': 9.30146960875e-06,
    'acbyhs4.r8b2': -1.69260077017e-05,
    'acbyhs5.r8b1': 4.83652230929e-06,
    'acbyhs5.r8b2': 4.38410426426e-06},
    'on_x1_v': {'acbcv5.l1b1': -4.49892171021e-08,
    'acbcv5.r1b2': -5.2151408641e-08,
    'acbcv6.l1b2': 4.75980448766e-08,
    'acbcv6.r1b1': 4.94068274744e-08,
    'acbxv1.l1': 5.49019607843e-08,
    'acbxv1.r1': -5.49019607843e-08,
    'acbxv2.l1': 5.49019607843e-08,
    'acbxv2.r1': -5.49019607843e-08,
    'acbxv3.l1': 5.49019607843e-08,
    'acbxv3.r1': -5.49019607843e-08,
    'acbyvs4.l1b1': -2.17704181511e-07,
    'acbyvs4.l1b2': 2.93325423015e-07,
    'acbyvs4.r1b1': 2.90096184016e-07,
    'acbyvs4.r1b2': -2.08045209558e-07},
    'on_x2v': {'acbcv5.r2b1': 0.0,
    'acbcvs5.r2b1': 2.63924957448e-07,
    'acbcvs5.r2b2': -5.5611394134e-08,
    'acbxv1.l2': 5.88235294118e-09,
    'acbxv1.r2': -5.88235294118e-09,
    'acbxv2.l2': 5.88235294118e-09,
    'acbxv2.r2': -5.88235294118e-09,
    'acbxv3.l2': 5.88235294118e-09,
    'acbxv3.r2': -5.88235294118e-09,
    'acbyv4.l2b1': 0.0,
    'acbyv4.r2b2': 0.0,
    'acbyv5.l2b2': 0.0,
    'acbyvs4.l2b1': -3.59306469082e-07,
    'acbyvs4.l2b2': 5.77672138807e-08,
    'acbyvs4.r2b1': -2.65881050814e-08,
    'acbyvs4.r2b2': -3.38366891365e-07,
    'acbyvs5.l2b1': -3.70036866051e-08,
    'acbyvs5.l2b2': 2.10669686866e-07},
    'on_x5_h': {'acbch5.l5b2': 1.02486278372e-07,
    'acbch5.r5b1': 8.97572884285e-08,
    'acbch6.l5b1': -1.6842005584e-07,
    'acbch6.r5b2': -1.61874731447e-07,
    'acbxh1.l5': 5.49019607843e-08,
    'acbxh1.r5': -5.49019607843e-08,
    'acbxh2.l5': 5.49019607843e-08,
    'acbxh2.r5': -5.49019607843e-08,
    'acbxh3.l5': 5.49019607843e-08,
    'acbxh3.r5': -5.49019607843e-08,
    'acbyhs4.l5b1': -9.83004449266e-08,
    'acbyhs4.l5b2': 2.24549528623e-07,
    'acbyhs4.r5b1': 2.38947550477e-07,
    'acbyhs4.r5b2': -9.24385638824e-08},
    'on_x1_h': {'acbch5.l1b2': 4.506876961269906e-08,
    'acbch5.r1b1': 5.2157133283597154e-08,
    'acbch6.l1b1': -4.7890510938800164e-08,
    'acbch6.r1b2': -4.9057882341900174e-08,
    'acbxh1.l1': 5.490196078429958e-08,
    'acbxh1.r1': -5.490196078429958e-08,
    'acbxh2.l1': 5.490196078429958e-08,
    'acbxh2.r1': -5.490196078429958e-08,
    'acbxh3.l1': 5.490196078429958e-08,
    'acbxh3.r1': -5.490196078429958e-08,
    'acbyhs4.l1b1': -2.936865112280011e-07,
    'acbyhs4.l1b2': 2.177064870590009e-07,
    'acbyhs4.r1b1': 2.0803740038799836e-07,
    'acbyhs4.r1b2': -2.89769479559999e-07},
    'on_x2h': {'acbch5.r2b2': 0.0,
    'acbchs5.r2b1': 5.500712482970298e-08,
    'acbchs5.r2b2': -2.619149321510001e-07,
    'acbxh1.l2': 0.0,
    'acbxh1.r2': -5.882352941177012e-09,
    'acbxh2.l2': 8.82352941176213e-09,
    'acbxh2.r2': -5.882352941177012e-09,
    'acbxh3.l2': 8.82352941176213e-09,
    'acbxh3.r2': -5.882352941177012e-09,
    'acbyh4.l2b2': 0.0,
    'acbyh4.r2b1': 0.0,
    'acbyh5.l2b1': 0.0,
    'acbyhs4.l2b1': -3.016846255819841e-08,
    'acbyhs4.l2b2': 3.516576042850015e-07,
    'acbyhs4.r2b1': 3.360059857369998e-07,
    'acbyhs4.r2b2': 3.609186240289533e-08,
    'acbyhs5.l2b1': -2.0978544129400017e-07,
    'acbyhs5.l2b2': 3.489932478159961e-08},
    'on_x5_v': {'acbcv5.l5b1': -1.0191059306399972e-07,
    'acbcv5.r5b2': -8.978187541590162e-08,
    'acbcv6.l5b2': 1.7060064662799925e-07,
    'acbcv6.r5b1': 1.5950619194600115e-07,
    'acbxv1.l5': 5.490196078429958e-08,
    'acbxv1.r5': -5.490196078429958e-08,
    'acbxv2.l5': 5.490196078429958e-08,
    'acbxv2.r5': -5.490196078429958e-08,
    'acbxv3.l5': 5.490196078429958e-08,
    'acbxv3.r5': -5.490196078429958e-08,
    'acbyvs4.l5b1': -2.209491956449975e-07,
    'acbyvs4.l5b2': 8.013405968279948e-08,
    'acbyvs4.r5b1': 1.1028061799399985e-07,
    'acbyvs4.r5b2': -2.427238375419971e-07},
    'on_x8v': {'acbcv5.l8b2': 0.0,
    'acbcvs5.l8b1': 1.0369522268200402e-08,
    'acbcvs5.l8b2': 1.956592257500014e-07,
    'acbxv1.l8': 1.1764705882401458e-08,
    'acbxv1.r8': -1.1764705882401458e-08,
    'acbxv2.l8': 1.1764705882401458e-08,
    'acbxv2.r8': -1.1764705882401458e-08,
    'acbxv3.l8': 1.1764705882401458e-08,
    'acbxv3.r8': -1.1764705882401458e-08,
    'acbyv4.l8b1': 0.0,
    'acbyv4.r8b2': 0.0,
    'acbyv5.r8b1': 0.0,
    'acbyvs4.l8b1': -3.705986005320014e-07,
    'acbyvs4.l8b2': 1.1718075751199826e-07,
    'acbyvs4.r8b1': 9.383772082330024e-08,
    'acbyvs4.r8b2': -3.705179788330009e-07,
    'acbyvs5.r8b1': 1.8445049648999976e-07,
    'acbyvs5.r8b2': 1.040864846200138e-08},
    'on_a2': {'acbch5.r2b2': 0.0,
    'acbchs5.r2b1': 1.2587850689e-07,
    'acbchs5.r2b2': 2.93963812481e-07,
    'acbyh4.l2b2': 0.0,
    'acbyh4.r2b1': 0.0,
    'acbyh5.l2b1': 0.0,
    'acbyhs4.l2b1': 3.68242425339e-08,
    'acbyhs4.l2b2': -3.30398196602e-07,
    'acbyhs4.r2b1': 2.97500680936e-07,
    'acbyhs4.r2b2': -1.04121190725e-07,
    'acbyhs5.l2b1': -2.37517327511e-07,
    'acbyhs5.l2b2': -8.31782737488e-08},
    'on_a8': {'acbcv5.l8b2': 0.0,
    'acbcvs5.l8b1': -8.21760660637e-08,
    'acbcvs5.l8b2': -2.50223614299e-07,
    'acbyv4.l8b1': 0.0,
    'acbyv4.r8b2': 0.0,
    'acbyv5.r8b1': 0.0,
    'acbyvs4.l8b1': -3.28608937393e-07,
    'acbyvs4.l8b2': -4.88045166324e-09,
    'acbyvs4.r8b1': -2.49723205992e-08,
    'acbyvs4.r8b2': 3.29247846628e-07,
    'acbyvs5.r8b1': 2.35889032138e-07,
    'acbyvs5.r8b2': 8.24862621483e-08},
    'on_ov1': {'acbcv5.l1b1': 2.04334717035e-05,
    'acbcv5.r1b2': 1.77021721706e-05,
    'acbcv6.l1b2': 1.38877440621e-05,
    'acbcv6.r1b1': 1.38099768884e-05,
    'acbcv7.l1b1': -3.57270190667e-05,
    'acbcv7.r1b2': -3.43980837178e-05,
    'acbcv8.l1b2': -3.67951166259e-05,
    'acbcv8.r1b1': -3.54017584491e-05,
    'acbyvs4.l1b1': 2.18954414715e-05,
    'acbyvs4.l1b2': 1.48813814753e-05,
    'acbyvs4.r1b1': 1.47980502321e-05,
    'acbyvs4.r1b2': 1.89687234898e-05},
    'on_ov2': {'acbcv5.r2b1': -8.09924595773e-06,
    'acbcv6.l2b1': -6.04651281594e-06,
    'acbcv6.r2b2': -1.13045475282e-05,
    'acbcv7.l2b2': -1.99307633032e-05,
    'acbcv7.r2b1': -2.02127351779e-05,
    'acbcvs5.r2b1': -8.09924595773e-06,
    'acbcvs5.r2b2': -3.77186132907e-05,
    'acbyv4.l2b1': 6.47912745271e-06,
    'acbyv4.r2b2': 1.2113362935e-05,
    'acbyv5.l2b2': -2.35466128048e-06,
    'acbyvs4.l2b1': 6.47912745271e-06,
    'acbyvs4.l2b2': 4.16028721622e-05,
    'acbyvs4.r2b1': 4.21914517202e-05,
    'acbyvs4.r2b2': 1.2113362935e-05,
    'acbyvs5.l2b1': -4.64522709319e-05,
    'acbyvs5.l2b2': -2.35466128048e-06},
    'on_oh5': {'acbch5.l5b2': 2.34391469357e-05,
    'acbch5.r5b1': 2.09738881576e-05,
    'acbch6.l5b1': 1.00210256974e-05,
    'acbch6.r5b2': 1.12146824409e-05,
    'acbch7.l5b2': -3.61933012237e-05,
    'acbch7.r5b1': -3.89631993797e-05,
    'acbch8.l5b1': -2.19370052312e-05,
    'acbch8.r5b2': -2.35263402012e-05,
    'acbyhs4.l5b1': 1.07380079522e-05,
    'acbyhs4.l5b2': 2.51161661279e-05,
    'acbyhs4.r5b1': 2.24745235292e-05,
    'acbyhs4.r5b2': 1.20170681991e-05},
    'on_oh8': {'acbch5.l8b1': -3.75977649657e-06,
    'acbch6.l8b2': -1.18221036945e-05,
    'acbch6.r8b1': -7.34487460812e-06,
    'acbch7.l8b1': -2.14223976213e-05,
    'acbch7.r8b2': -1.76714732648e-05,
    'acbchs5.l8b1': -3.75977649657e-06,
    'acbchs5.l8b2': -2.59949391155e-05,
    'acbyh4.l8b2': 1.26679491018e-05,
    'acbyh4.r8b1': 7.87038416337e-06,
    'acbyh5.r8b2': -4.73283800199e-06,
    'acbyhs4.l8b1': 4.47164644971e-05,
    'acbyhs4.l8b2': 1.26679491018e-05,
    'acbyhs4.r8b1': 7.87038416337e-06,
    'acbyhs4.r8b2': 3.68868985081e-05,
    'acbyhs5.r8b1': -4.02874597858e-05,
    'acbyhs5.r8b2': -4.73283800199e-06},
    'on_xip1b1': {'acbch5.r1b1': -0.00010695556865921357,
    'acbch6.l1b1': -3.0217482187889205e-05,
    'acbyh4.l1b1': 6.856138213387039e-05,
    'acbyhs4.r1b1': 0.0001391794798175955},
    'on_xip1b2': {'acbch5.l1b2': -9.241987776514907e-05,
    'acbch6.r1b2': -3.103986623665818e-05,
    'acbyh4.r1b2': 7.116645148310001e-05,
    'acbyhs4.l1b2': 0.00011935165190059529},
    'on_xip2b1': {'acbchs5.r2b1': -0.00010712188353610386,
    'acbyh4.r2b1': 6.699916191143782e-05,
    'acbyh5.l2b1': -4.3755233051432573e-05,
    'acbyhs4.l2b1': 0.0001130012471953883},
    'on_xip2b2': {'acbch5.r2b2': -5.02569928150234e-05,
    'acbyh4.l2b2': 3.903747975130236e-05,
    'acbyhs4.r2b2': 0.00011947754342607242,
    'acbyhs5.l2b2': -6.915878820476851e-05},
    'on_xip5b1': {'acbch5.r5b1': -6.90438229941417e-05,
    'acbch6.l5b1': -3.4122763497305895e-05,
    'acbyh4.l5b1': 4.305846686316669e-05,
    'acbyhs4.r5b1': 0.00012182670006994758},
    'on_xip5b2': {'acbch5.l5b2': -7.883531897078955e-05,
    'acbch6.r5b2': -3.283478503588004e-05,
    'acbyh4.r5b2': 4.426343111081036e-05,
    'acbyhs4.l5b2': 0.00013290206219095492},
    'on_xip8b1': {'acbch5.l8b1': -4.451344020420723e-05,
    'acbch6.r8b1': -1.772515768038751e-05,
    'acbyh4.r8b1': -1.3161405565541242e-06,
    'acbyhs4.l8b1': 0.00010534085207737146},
    'on_xip8b2': {'acbchs5.l8b2': -6.655636938812533e-05,
    'acbyh4.l8b2': 4.0116240399150104e-05,
    'acbyh5.r8b2': -4.4338431146318484e-05,
    'acbyhs4.r8b2': 0.00010554338873316548},
    'on_yip1b1': {'acbcv5.l1b1': -9.226682040626933e-05,
    'acbcv6.r1b1': -3.125540718932119e-05,
    'acbyv4.r1b1': 7.095340242677776e-05,
    'acbyvs4.l1b1': 0.00011936980924063663},
    'on_yip1b2': {'acbcv5.r1b2': -0.00010695549276253395,
    'acbcv6.l1b2': -3.0042502923549863e-05,
    'acbyv4.l1b2': 6.880556134947911e-05,
    'acbyvs4.r1b2': 0.0001391790416970699},
    'on_yip2b1': {'acbcv5.r2b1': -5.0654087072343794e-05,
    'acbyv4.l2b1': 2.2322593752092693e-05,
    'acbyvs4.r2b1': 0.00011745759301569891,
    'acbyvs5.l2b1': -6.944024443477368e-05},
    'on_yip2b2': {'acbcvs5.r2b2': -0.00010791908518872986,
    'acbyv4.r2b2': 6.233838288087331e-05,
    'acbyv5.l2b2': -4.403924983532336e-05,
    'acbyvs4.l2b2': 0.00010597017184973377},
    'on_yip5b1': {'acbcv5.l5b1': -7.839845878125118e-05,
    'acbcv6.r5b1': -3.2240678693847206e-05,
    'acbyv4.r5b1': 4.0605110058233945e-05,
    'acbyvs4.l5b1': 0.0001356822376765277},
    'on_yip5b2': {'acbcv5.r5b2': -6.906799725731989e-05,
    'acbcv6.l5b2': -3.468990844473363e-05,
    'acbyv4.l5b2': 4.6802558808403085e-05,
    'acbyvs4.r5b2': 0.0001189312950278371},
    'on_yip8b1': {'acbcvs5.l8b1': -6.842111090808824e-05,
    'acbyv4.l8b1': 4.053123643411973e-05,
    'acbyv5.r8b1': -4.3689926858664866e-05,
    'acbyvs4.r8b1': 0.00011079456790525628},
    'on_yip8b2': {'acbcv5.l8b2': -4.569813567560489e-05,
    'acbyv4.r8b2': 3.999948490422221e-05,
    'acbyvs4.l8b2': 0.00010553487983680055,
    'acbyvs5.r8b2': -6.864872704879031e-05}}


    configs = {}

    for nn in dct.keys():

        configs[nn] = {}

        purpose = (
                'xipb1' if '_xip' in nn and 'b1' in nn
            else 'xipb2' if '_xip' in nn and 'b2' in nn
            else 'yipb1' if '_yip' in nn and 'b1' in nn
            else 'yipb2' if '_yip' in nn and 'b2' in nn
            else 'sep' if '_sep' in nn
            else 'o' if '_o' in nn
            else 'a' if '_a' in nn
            else 'x' if '_x' in nn
            else None)
        assert purpose is not None

        configs[nn]['purpose'] = purpose

        # Determine the plane
        one_corrector = list(dct[nn].keys())[0]
        if 'h' in one_corrector and 'v' in one_corrector:
            raise ValueError(f'Cannot determine plane for {nn}')
        plane = 'x' if 'h' in one_corrector else 'y'
        configs[nn]['plane'] = plane

        # Determine ip
        tmp = one_corrector.split('.')[1][1]
        assert tmp in ['1', '2', '5', '8']
        ip = int(tmp)
        configs[nn]['ip'] = ip

        # generate targets
        if purpose == 'sep':
            targets = {
                'lhcb1': {plane:  1e-3, 'p' + plane: 0},
                'lhcb2': {plane: -1e-3, 'p' + plane: 0},
            }
        elif purpose == 'o':
            targets = {
                'lhcb1': {plane:  1e-3, 'p' + plane: 0},
                'lhcb2': {plane:  1e-3, 'p' + plane: 0},
            }
        elif purpose == 'x':
            targets = {
                'lhcb1': {plane:  0, 'p' + plane: 1e-6},
                'lhcb2': {plane:  0, 'p' + plane: -1e-6},
            }
        elif purpose == 'a':
            targets = {
                'lhcb1': {plane:  0, 'p' + plane: 1e-6},
                'lhcb2': {plane:  0, 'p' + plane: 1e-6},
            }
        elif purpose == 'xipb1':
            targets = {
                'lhcb1': {plane:  1e-3, 'p' + plane: 0},
                'lhcb2': None
            }
        elif purpose == 'xipb2':
            targets = {
                'lhcb1': None,
                'lhcb2': {plane:  1e-3, 'p' + plane: 0},
            }
        elif purpose == 'yipb1':
            targets = {
                'lhcb1': {plane:  1e-3, 'p' + plane: 0},
                'lhcb2': None
            }
        elif purpose == 'yipb2':
            targets = {
                'lhcb1': None,
                'lhcb2': {plane:  1e-3, 'p' + plane: 0},
            }
        else:
            raise ValueError(f'Unknown purpose {purpose}')

        correctors = dct[nn].copy()
        for cc in correctors:
            if not cc.startswith('acbx'):
                correctors[cc] = None

        configs[nn]['targets'] = targets
        configs[nn]['correctors'] = correctors


    # All orbit correctors off
    corrector_names = collider.vars.get_table().rows['acb.*'].name
    for knob_name in corrector_names:
        collider.vars[knob_name] = 0.0

    twflat = collider.twiss()

    assert_allclose = np.testing.assert_allclose
    assert_allclose(twflat.lhcb1.x, 0.0, atol=1e-14)
    assert_allclose(twflat.lhcb1.y, 0.0, atol=1e-14)
    assert_allclose(twflat.lhcb2.x, 0.0, atol=1e-14)
    assert_allclose(twflat.lhcb2.y, 0.0, atol=1e-14)



    knob_name = list(configs.keys())[0] # To be replaced by a loop over all knobs
    for knob_name in configs.keys():

        conf = configs[knob_name]
        ipn = conf['ip']

        # Build targets
        targets = []
        for lname in ['lhcb1', 'lhcb2']:
            if conf['targets'][lname] is None:
                continue

            for tname in conf['targets'][lname]:
                targets.append(xt.Target(tname, line=lname, at='ip'+str(ipn),
                                        value=conf['targets'][lname][tname]))

        start_b1 = f'e.ds.l{ipn}.b1'
        start_b2 = f'e.ds.l{ipn}.b2'
        end_b1 = f's.ds.r{ipn}.b1'
        end_b2 = f's.ds.r{ipn}.b2'

        plane = conf['plane']
        targets += [
            xt.Target(plane, 0, line='lhcb1', at=xt.END),
            xt.Target(plane, 0, line='lhcb2', at=xt.END),
            xt.Target('p' + plane, 0, line='lhcb2', at=xt.END),
            xt.Target('p' + plane, 0, line='lhcb1', at=xt.END),
        ]

        default_tols = {'x': 1e-8, 'y': 1e-8, 'px': 1e-10, 'py': 1e-10}
        for tt in targets:
            nnn = tt.tar[0]
            tt.tol = default_tols[nnn] # set tolerances

        # Build vary
        vary = []
        for cc in conf['correctors']:
            vary.append(xt.Vary(cc, step=1e-8))

        opt = collider.match_knob(
            knob_name=knob_name,
            knob_value_start=0.,
            knob_value_end=1., # I assume that the targets from conf are set got knob=1
            run=False,
            start=(start_b1, start_b2),
            end=(end_b1, end_b2),
            init=[xt.TwissInit(), xt.TwissInit()], # Zero orbit
            vary=vary,
            targets=targets)

        # Force correctors that have values in the config (mcbx)
        for vv in opt.vary:
            value_from_conf = conf['correctors'][vv.name.split('_from_')[0]]
            if value_from_conf is not None:
                collider.vars[vv.name] = value_from_conf
                vv.active = False

        opt.solve()
        opt.generate_knob()



