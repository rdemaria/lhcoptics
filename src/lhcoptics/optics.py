import json
from pprint import pp
import re
from pathlib import Path
import gzip

import numpy as np

import matplotlib.pyplot as plt
from cpymad.madx import Madx

from .arcs import LHCArc
from .circuits import LHCCircuits
from .ir1 import LHCIR1
from .ir2 import LHCIR2
from .ir3 import LHCIR3
from .ir4 import LHCIR4
from .ir5 import LHCIR5
from .ir6 import LHCIR6
from .ir7 import LHCIR7
from .ir8 import LHCIR8
from .model_xsuite import LHCMadxModel, LHCXsuiteModel
from .aperture import LHCAperture
from .section import Knob
from .utils import (
    deliver_list_str,
    print_diff_dict_float,
    print_diff_dict_objs,
    read_knob_structure,
    find_comparable_values,
)

_opl = ["_op", "_sq", ""]


def set_ip_labels(ax, tw):
    ips = tw.rows["ip[1-8]"].s
    lbl = [s.upper() for s in tw.rows["ip[1-8]"].name]
    ax.set_xticks(ips, labels=lbl)
    ax.set_xlabel(None)


class LHCOptics:
    """
    Optics containts global knobs, global parameters and sections
    Section contains strengths, local knobs, local parameters
    """

    _arcs = ["a12", "a23", "a34", "a45", "a56", "a67", "a78", "a81"]
    _irs = [LHCIR1, LHCIR2, LHCIR3, LHCIR4, LHCIR5, LHCIR6, LHCIR7, LHCIR8]

    knob_names = [f"dq{x}.b{b}{op}" for x in "xy" for b in "12" for op in _opl]
    knob_names += [f"dqp{x}.b{b}{op}" for x in "xy" for b in "12" for op in _opl]
    knob_names += [f"cm{x}s.b{b}{op}" for x in "ir" for b in "12" for op in _opl]
    knob_names += [
        f"{kk}.b{b}" for kk in ["on_mo", "phase_change", "dp_trim"] for b in "12"
    ]
    knob_names += ["on_ssep1_h", "on_xx1_v", "on_ssep5_v", "on_xx5_h"]
    knob_names += ["dqxdjy.b1", "dqxdjy.b1"]

    @classmethod
    def get_default_knob_names(cls):
        out = cls.knob_names[:]
        for ss in cls._irs:
            out.extend(ss.knob_names)
        return out

    @classmethod
    def from_dict(
        cls,
        data,
        xsuite_model=None,
        madx_model=None,
        circuits=None,
        aperture=None,
        verbose=False,
        name=None,
    ):
        irs = [
            globals()[f"LHCIR{n + 1}"].from_dict(d) for n, d in enumerate(data["irs"])
        ]
        arcs = [LHCArc.from_dict(d) for d in data["arcs"]]
        if isinstance(xsuite_model, str) or isinstance(xsuite_model, Path):
            xsuite_model = LHCXsuiteModel.from_json(xsuite_model)
        out = cls(
            name=data.get("name", name),
            irs=irs,
            arcs=arcs,
            params=data["params"],
            knobs={k: Knob.from_dict(d) for k, d in data["knobs"].items()},
            model=xsuite_model,
            variant=data.get("variant", "2025"),
        )
        if xsuite_model is not None:
            out.update_model(verbose=verbose)
        elif madx_model is not None:
            out.set_madx_model(madx_model)
        if circuits is not None:
            out.set_circuits(circuits)
        if aperture is not None:
            if isinstance(aperture, str) or isinstance(aperture, Path):
                out.aperture = LHCAperture.from_json(aperture)
            else:
                out.aperture = LHCAperture.from_json(aperture)
        return out

    @classmethod
    def from_json(
        cls,
        filename,
        name=None,
        xsuite_model=None,
        madx_model=None,
        circuits=None,
        aperture=None,
        verbose=False,
    ):
        if filename.endswith(".gz"):
            fh = gzip.open(filename, "rt")
        else:
            fh = open(filename, "r")
        with fh as f:
            data = json.load(f)
            if name is None:
                name = Path(filename).stem
            out = cls.from_dict(
                data,
                xsuite_model=xsuite_model,
                madx_model=madx_model,
                circuits=circuits,
                verbose=verbose,
                aperture=aperture,
                name=name,
            )
            return out

    @classmethod
    def from_madx(
        cls,
        madx,
        knob_structure,
        name=None,
        sliced=False,
        make_model=None,
        xsuite_model=None,
        circuits=None,
        verbose=False,
        variant="2025",
    ):
        """
        Create an LHCOptics object from a MADX model.

        Steps:
            1. Create a MADX model from the provided `madx` object.
            2. Read the knob structure from the `knob_structure` parameter.
            3. Create knobs using the MADX model and the knob structure.
               This sets also the knob values to zero in the model
            4. Populate the sections using from_model
                1. collect stregnth names
                2. make and set knob values to zero
                3. get strengths values
                4. put knobs back
                5. get params from variables if they exist
            5. Restores the knob values
            6. If ask create an xsuite or madx model, or uses one
            7. Set the model
            8. Set the ciruits if needed
        """
        madmodel = LHCMadxModel(madx)
        knob_structure = read_knob_structure(knob_structure)
        knobs = madmodel.make_and_set0_knobs(
            knob_structure.get("global", []), variant=variant
        )
        irs = [
            ir.from_model(
                madmodel, knob_names=knob_structure.get(ir.name), variant=variant
            )
            for ir in cls._irs
        ]
        arcs = [
            LHCArc.from_model(madmodel, arc, knob_names=knob_structure.get(arc))
            for arc in cls._arcs
        ]
        for k, knob in knobs.items():
            madmodel[k] = knob.value
        self = cls(name, irs, arcs, knobs=knobs, variant=variant)
        if make_model == "xsuite":
            xsuite_model = LHCXsuiteModel.from_madx(
                madx, sliced=sliced, knob_structure=knob_structure
            )
        elif make_model == "madx":
            self.model = madmodel
        elif make_model is None:
            pass
        else:
            raise ValueError(
                f"Unknown make_model {make_model}, use 'xsuite' or 'madx' or None"
            )
        if xsuite_model is not None:
            self.set_xsuite_model(xsuite_model, verbose=verbose)
        if circuits is not None:
            self.set_circuits(circuits)
        return self

    @classmethod
    def from_xsuite(
        cls,
        xsuite_model,
        name=None,
        circuits=None,
        knob_structure=None,
        variant="2025",
        params_mode="from_twiss_init",
        verbose=False,
    ):
        """
        Create an LHCOptics object from an Xsuite model.
        Parameters
        ----------
        xsuite_model : LHCXsuiteModel or str or Path
            The Xsuite model or the path to the json file.
        name : str
            The name of the optics object.
        circuits : LHCCircuits or str or Path
            The circuits object or the path to the json file.
        knob_structure : dict or str or Path
            The knob structure or the path to the yaml file.
        variant : str
            The optics variant such as '2025' or 'hl'.
        params_mode : str
            The mode to get the parameters from the model.
            Options are 'from_twiss_init', 'from_variables'.
        Returns
        -------
        LHCOptics
            The LHCOptics object.
        """

        if isinstance(xsuite_model, str) or isinstance(xsuite_model, Path):
            xsuite_model = LHCXsuiteModel.from_json(xsuite_model)
        elif hasattr(xsuite_model, "to_json"):
            xsuite_model = LHCXsuiteModel(xsuite_model)
        if knob_structure is None:
            knob_structure = read_knob_structure(
                xsuite_model.env.metadata["knob_structure"]
            )
        elif isinstance(knob_structure, str) or isinstance(knob_structure, Path):
            knob_structure = read_knob_structure(knob_structure)
        else:
            print("No knob_structure provided")
        knobs = xsuite_model.make_and_set0_knobs(
            knob_structure.get("global", []), variant=variant
        )
        irs = [
            ir.from_model(
                xsuite_model, knob_names=knob_structure.get(ir.name), variant=variant
            )
            for ir in cls._irs
        ]
        arcs = [
            LHCArc.from_model(
                xsuite_model, arc, knob_names=knob_structure.get(arc), variant=variant
            )
            for arc in cls._arcs
        ]
        self = cls(name, irs, arcs, knobs=knobs, variant=variant)
        self.model = (
            xsuite_model  #  TODO replace with set_xsuite_model to make more checks
        )
        if circuits is not None:
            self.set_circuits(circuits)
        self.create_knobs(verbose=verbose)
        self.set_params(mode=params_mode, full=True)
        for k, knob in knobs.items():
            xsuite_model[k] = knob.value
        return self

    @classmethod
    def from_madxfile(
        cls,
        filename,
        name=None,
        sliced=False,
        make_model=None,
        xsuite_model=None,
        stdout=False,
        verbose=False,
        knob_structure=None,
        variant="2025",
    ):
        """
        Create an LHCOptics object from a MADX file.

        Parameters
        ----------
        filename : str
            The name of the MADX file.
        name : str
            The name of the optics object.
        sliced : bool
            If True, the optics will be sliced.
        make_model : bool
            If True, a MADX model will be created.
        xsuite_model : bool
            If True, an Xsuite model will be created.
        stdout : bool
            If True, the MADX output will be printed to stdout.
        verbose : bool
            If True, the MADX output will be printed to stdout.
        knob_names : list
            A list of knob names to be used.

        Returns
        -------
        LHCOptics
            The LHCOptics object.
        """
        madx = Madx(stdout=stdout)
        madx.call(filename)
        if name is None:
            name = str(filename)
        return cls.from_madx(
            madx,
            name=name,
            knob_structure=knob_structure,
            sliced=sliced,
            make_model=make_model,
            xsuite_model=xsuite_model,
            verbose=verbose,
            variant=variant,
        )

    def __init__(
        self,
        name,
        irs=None,
        arcs=None,
        params=None,
        knobs=None,
        model=None,
        circuits=None,
        aperture=None,
        variant="2025",
    ):
        if name is None:
            name = "lhcoptics"
        self.name = name
        if irs is None:
            irs = [IR(variant=variant) for IR in self._irs]
        if arcs is None:
            arcs = [LHCArc(arc, variant=variant) for arc in self._arcs]
        for ir in irs:
            setattr(self, ir.name, ir)
            ir.parent = self
        for arc in arcs:
            setattr(self, arc.name, arc)
            arc.parent = self
        if params is None:
            params = {}
        if knobs is None:
            knobs = {}
        self.params = params
        self.knobs = knobs
        self.model = model
        self.circuits = circuits
        self.aperture = aperture
        self.variant = variant
        # print(f"Optics {self.name} created")
        for knob in knobs.values():
            knob.parent = self
        for ss in irs + arcs:
            for knob in ss.knobs.values():
                knob.parent = self

    def __getitem__(self, k):
        if k in self.params:
            return self.params[k]
        if k in self.knobs:
            return self.knobs[k]
        for ss in self.irs + self.arcs:
            if k in ss:
                return ss[k]
        raise KeyError(f"{k} not found in {self}")

    def __contains__(self, k):
        return k in self.params or k in self.knobs

    def __repr__(self) -> str:
        return f"<LHCOptics {self.name!r}>"

    @property
    def arcs(self):
        """List of arcs"""
        return [getattr(self, arc) for arc in self._arcs]

    @property
    def irs(self):
        """List of IRs"""
        return [getattr(self, ir.name) for ir in self._irs]

    def twissip(self):
        """Compute twiss and print orbit at IPs, tunes and chromaticity"""
        tw1, tw2 = self.twiss(chrom=True, strengths=False)
        header = True
        cols = "betx bety dx dpx px*1e6 py*1e6 x*1e3 y*1e3"
        for ip in ["ip1", "ip2", "ip5", "ip8"]:
            tw1.rows[ip].cols[cols].show(digits=4, fixed="f", header=header)
            if header:
                header = False
            tw2.rows[ip].cols[cols].show(digits=4, fixed="f", header=header)
        print("         HB1         HB2         VB1         VB2")
        print(f"Tunes:  {tw1.qx:11.6f} {tw2.qx:11.6f} {tw1.qy:11.6f} {tw2.qy:11.6f}")
        print(
            f"Chroma: {tw1.dqx:11.6f} {tw2.dqx:11.6f} {tw1.dqy:11.6f} {tw2.dqy:11.6f}"
        )
        return self

    def check(self, verbose=False):
        """Compute twiss and print orbit at IPs, tunes and chromaticity"""
        self.twissip()

    def check_knobs(self, fail=False, verbose=True):
        report = []
        for knob in self.find_knobs():
            if not hasattr(knob, "check"):
                msg = f"Knob {knob.name} has no check method, skipping"
                if verbose:
                    print(msg)
                report.append(msg)
                continue
            try:
                res = knob.check(verbose=verbose)
                if res:
                    msg = f"Knob {knob.name} OK"
                else:
                    msg = f"Knob {knob.name} FAILED"
            except Exception as e:
                msg = f"Knob {knob.name} RAISED EXCEPTION: {e}"
                res = False
                if verbose:
                    print(f"Knob {knob.name} raised {e}")
                if fail:
                    raise e
            if verbose:
                print(msg)
            if fail and not res:
                raise ValueError(f"Knob {knob.name} failed the check")
            report.append(msg)
        if verbose:
            print("Knob check report:")
            print("\n".join(report))
        return self

    def check_match(self, verbose=False):
        out = {}
        for ss in self.irs + self.arcs:
            res = ss.check_match(verbose=verbose)
            out[ss.name] = res
        if verbose:
            for k, v in out.items():
                print(f" Section {k} match status: {v}")
        return np.all(out.values())

    def check_phase_params(self, verbose=False, tol=1e-9, fail=False, correct=False):
        ret = True
        for xy in "xy":
            for beam in "12":
                data = {}
                for arc in self.arcs:
                    nn = f"mu{xy}{arc.name}b{beam}"
                    data[nn] = arc.params[nn]
                for ir in self.irs:
                    nn = f"mu{xy}ip{ir.irn}b{beam}"
                    data[nn] = ir.params[nn]
                sum = 0
                for k, v in data.items():
                    sum += v
                qq = self.params[f"q{xy}b{beam}"]
                if abs(sum - qq) > tol:
                    if verbose:
                        print(
                            f"mu{xy}b{beam}: Sum={sum} Q={qq} Diff={sum - qq} > tol={tol} FAIL"
                        )
                    ret = False
                    if fail:
                        raise ValueError(
                            f"Phase parameter mu{xy}b{beam} check failed: Sum={sum}, Q={qq}"
                        )
                nn = f"mu{xy}a34b{beam}"
                if correct and abs(sum - qq) > tol:
                    old = data[nn]
                    new = old + qq - sum
                    if verbose:
                        print(
                            f"Correcting {nn} from {old} to {new} to enforce sum of phases equal to Q"
                        )
                    self.arcs[2].params[nn] = new
                    ret = True
        return ret

    def check_params(self, verbose=True, tol=5e-8, fail=False):
        ret = True
        rows = []
        tw1, tw2 = self.twiss(strengths=False, chrom=True)
        params = self.get_params_from_twiss(tw1, tw2)
        for k, v in self.params.items():
            if k in params:
                vtw = params[k]
                diff = v - vtw
                if abs(diff) > tol:
                    rows.append(("global", k, v, vtw, diff))
                    ret = False
                    if fail:
                        raise ValueError(
                            f"Parameter {k} check failed: Optics={v}, Twiss={vtw}"
                        )
        for ss in self.irs + self.arcs:
            mismatches = ss.get_param_mismatches(tol=tol)
            if mismatches:
                ret = False
                for k, v, vtw, diff in mismatches:
                    rows.append((ss.name, k, v, vtw, diff))
                if fail:
                    k, v, vtw, _ = mismatches[0]
                    raise ValueError(
                        f"Parameter {k} check failed: Optics={v}, Twiss={vtw}"
                    )
        if verbose and rows:
            print(
                f"{'Section':8s} {'Param':20s} {'Optics':>18s} {'Twiss':>18s} {'Diff':>12s} {'Tol':>10s} {'Status':>6s}"
            )
            for section, k, v, vtw, diff in rows:
                print(
                    f"{section:8s} {k:20s} {v:18.10g} {vtw:18.10g} {diff:12.3e} {tol:10.1e} {'FAIL':>6s}"
                )
        return ret

    def check_quad_strengths(
        self,
        verbose=False,
        p0c=None,
        ratio=1.5,
        margin=0.1,
    ):
        out = {}
        for ir in self.irs:
            if verbose:
                print(f"       Check {ir.name.upper()}")
            irout = ir.check_quad_strengths(
                verbose=verbose,
                p0c=p0c,
                ratio=ratio,
                margin=margin,
            )
            out.update(irout)
        return out

    def compare_init_ir15(self, tol=1e-6):
        """
        Compare the initial ATS at the left of IR1 and IR5 for both beams.
        """
        self.ir1.set_init()
        self.ir5.set_init()
        for beam in [1, 2]:
            for var in "betx bety alfx alfy dx dpx".split():
                for lr in ["left", "right"]:
                    ini1 = getattr(self.ir1, "init_" + lr)[beam]
                    ini5 = getattr(self.ir5, "init_" + lr)[beam]
                    v1 = getattr(ini1, var)
                    v5 = getattr(ini5, var)
                    diff = abs(v1 - v5)
                    print(
                        f"{lr.capitalize():<5} b{beam} {var:<4}: IR1={v1:15.6e} IR5={v5:15.6e} diff={diff:15.6e}"
                    )

    def compare_twiss_ir15(self, stol=1e-6):
        out = {}
        for beam in [1, 2]:
            tw1 = self.ir1.twiss_from_params(beam)
            tw5 = self.ir5.twiss_from_params(beam)
            ia, ib = find_comparable_values(tw1.s, tw5.s, tol=stol)
            for var in "s betx bety alfx alfy dx dpx".split():
                v1 = tw1[var][ia]
                v5 = tw5[var][ib]
                out[f"{var}b{beam}_ir{1}"] = v1
                out[f"{var}b{beam}_ir{5}"] = v5
        return out

    def copy(self, name=None):
        """
        Copy the optics object, including all sections and knobs.
        """
        if name is None:
            name = self.name
        other = self.__class__(
            name=name,
            params=self.params.copy(),
            knobs={k: v.copy() for k, v in self.knobs.items()},
            irs=[ir.copy() for ir in self.irs],
            arcs=[arc.copy() for arc in self.arcs],
            circuits=self.circuits,
            model=self.model,
        )
        return other

    def copy_ir1_to_ir5(self):
        """
        Copy the IR1 parameters, strengths and knobs to IR5.
        """
        for name, value in self.ir1.params.items():
            if name.replace("ip1", "ip5") in self.ir5.params:
                self.ir5.params[name.replace("ip1", "ip5")] = value
        for name, value in self.ir1.strengths.items():
            name5 = name.replace(".l1", ".l5").replace(".r1", ".r5")
            if name5 in self.ir5.strengths:
                print(f"Copying strength {name} to {name5}: {value}")
                self.ir5.strengths[name5] = value

    def create_knobs(self, verbose=False):
        """Create knobs in the model from the optics knobs."""
        self.model.create_knobs(self.knobs, verbose=verbose)
        for ss in self.irs + self.arcs:
            ss.create_knobs(verbose=verbose)
        return self

    def diff(self, other=None, full=True):
        """
        Compare the optics with another optics object or a json file.
        """
        if other is None:
            self.diff_model()
        else:
            if isinstance(other, str) or isinstance(other, Path):
                other = self.__class__.from_json(other)
            self.diff_knobs(other)
            self.diff_params(other)
            if full:
                for ss, so in zip(self.irs + self.arcs, other.irs + other.arcs):
                    ss.diff_strengths(so)
                    ss.diff_knobs(so)
                    ss.diff_params(so)

    def diff_model(self, model=None):
        """Display differences in strengths with respect to a model."""
        model = self.model if model is None else model
        for ss in self.irs + self.arcs:
            ss.diff_model(model=model)

    def diff_knobs(self, other):
        print_diff_dict_objs(self.knobs, other.knobs)

    def diff_params(self, other):
        print_diff_dict_float(self.params, other.params)

    def find_strengths(self, regexp=None):
        strengths = {}
        for ss in self.irs + self.arcs:
            strengths.update(ss.strengths)
        if regexp is not None:
            strengths = {k: v for k, v in strengths.items() if re.match(regexp, k)}
        return strengths

    def find_knobs(self, regexp=None):
        """Find all knobs in the optics and its sections."""
        knobs = {}
        for ss in self.irs + self.arcs:
            knobs.update(ss.knobs)
        knobs.update(self.knobs)
        if regexp is not None:
            knobs = {k: v for k, v in knobs.items() if re.match(regexp, k)}
        return knobs.values()

    def find_knobs_null(self):
        """Find all knobs in the optics and its sections that are empty."""
        knobs = self.find_knobs()
        return {
            knob for knob in knobs.items() if sum(map(abs, knob.weights.values())) == 0
        }

    def get(self, k, default=None, full=True):
        """
        Get a parameter, strengths or knobs from the optics or its sections.
        If full is True, search in all sections.
        """
        if k in self:
            return self[k]
        if full:
            for ss in self.irs + self.arcs:
                if k in ss:
                    return ss[k]
        return default

    def get_knobs_active(self):
        """Return the active knobs in the model."""
        model = self.model
        return {k: v for k in self.find_knobs() if (v := model[k.name]) != 0}

    def get_bumps(self):
        """Return the orbit bumps values from model."""
        out = {}
        for ss in self.irs:
            for k, v in ss.knobs.items():
                if re.match(r"on_(sep|x|o|a)", k):
                    out[k] = v.value
        for ss in self.knobs:
            if re.match(r"on_d(sep|x|o|a)", ss):
                out[ss] = self.model[ss]
        return out

    def get_knob_structure(self):
        """Return the names and locations of the knobs in the optics."""
        out = {}
        out["global"] = list(self.knobs.keys())
        for ss in self.irs + self.arcs:
            out[ss.name] = list(ss.knobs.keys())
        return out

    def get_mkdtct(self, tw1=None, tw2=None):
        """
        Compute the TCT and MKD phase advances at IP1, IP5 and IP8

        B1: IP1 TCT IP5       MKD IP6      TCP IP7        TCT IP1
        B2: IP1     IP5 TCT       IP6 MKD      IP7 TCP        IP1 TCT
        """
        if tw1 is None:
            tw1 = self.twiss(1, strengths=False)
        if tw2 is None:
            tw2 = self.twiss(2, strengths=False)
        mux_tcphb1 = tw1["mux", "tcp.b6l7.b1"]
        muy_tcpvb1 = tw1["muy", "tcp.d6l7.b1"]
        mux_tct5b1 = tw1["mux", "tctph.4l5.b1"]
        muy_tct5b1 = tw1["muy", "tctpv.4l5.b1"]
        mux_tct1b1 = tw1["mux", "tctph.4l1.b1"]
        muy_tct1b1 = tw1["muy", "tctpv.4l1.b1"]
        mux_tct8b1 = tw1["mux", "tctph.4l8.b1"]
        muy_tct8b1 = tw1["muy", "tctpv.4l8.b1"]
        mux_mkdob1 = tw1["mux", "mkd.o5l6.b1"]
        mux_mkdab1 = tw1["mux", "mkd.a5l6.b1"]

        mux_tcphb2 = tw2["mux", "tcp.b6r7.b2"]
        muy_tcpvb2 = tw2["muy", "tcp.d6r7.b2"]
        mux_tct5b2 = tw2["mux", "tctph.4r5.b2"]
        muy_tct5b2 = tw2["muy", "tctpv.4r5.b2"]
        mux_tct1b2 = tw2["mux", "tctph.4r1.b2"]
        muy_tct1b2 = tw2["muy", "tctpv.4r1.b2"]
        mux_tct8b2 = tw2["mux", "tctph.4r8.b2"]
        muy_tct8b2 = tw2["muy", "tctpv.4r8.b2"]
        mux_mkdob2 = tw2["mux", "mkd.o5r6.b2"]
        mux_mkdab2 = tw2["mux", "mkd.a5r6.b2"]

        qx = 61.31
        qy = 60.32

        out = {
            "mkda_tct1_b1": mux_tct1b1 - mux_mkdab1,
            "mkdo_tct1_b1": mux_tct1b1 - mux_mkdob1,
            "mkda_tct5_b1": mux_tct5b1 - mux_mkdab1 + qx,
            "mkdo_tct5_b1": mux_tct5b1 - mux_mkdob1 + qx,
            "mkda_tct8_b1": mux_tct8b1 - mux_mkdab1,
            "mkdo_tct8_b1": mux_tct8b1 - mux_mkdob1,
            "mkda_tct1_b2": -mux_tct1b2 + mux_mkdab2,
            "mkdo_tct1_b2": -mux_tct1b2 + mux_mkdob2,
            "mkda_tct5_b2": -mux_tct5b2 + mux_mkdab2,
            "mkdo_tct5_b2": -mux_tct5b2 + mux_mkdob2,
            "mkda_tct8_b2": -mux_tct8b2 + mux_mkdab2 + qx,
            "mkdo_tct8_b2": -mux_tct8b2 + mux_mkdob2 + qx,
            "tcph_tct1_b1": mux_tct1b1 - mux_tcphb1,
            "tcpv_tct1_b1": muy_tct1b1 - muy_tcpvb1,
            "tcph_tct5_b1": mux_tct5b1 - mux_tcphb1 + qx,
            "tcpv_tct5_b1": muy_tct5b1 - muy_tcpvb1 + qy,
            "tcph_tct8_b1": mux_tct8b1 - mux_tcphb1,
            "tcpv_tct8_b1": muy_tct8b1 - muy_tcpvb1,
            "tcph_tct1_b2": -mux_tct1b2 + mux_tcphb2,
            "tcpv_tct1_b2": -muy_tct1b2 + muy_tcpvb2,
            "tcph_tct5_b2": -mux_tct5b2 + mux_tcphb2,
            "tcpv_tct5_b2": -muy_tct5b2 + muy_tcpvb2,
            "tcph_tct8_b2": -mux_tct8b2 + mux_tcphb2 + qx,
            "tcpv_tct8_b2": -muy_tct8b2 + muy_tcpvb2 + qy,
        }
        return out

    def get_params(self, mode="from_twiss", full=False, verbose=False):
        """
        Get the parameters from the optics and its sections.
        """
        if verbose:
            print(f"Getting parameters from mode {mode} with full={full}")
        if mode == "from_twiss":
            tw1 = self.model.b1.twiss(
                compute_chromatic_properties=True, strengths=False
            )
            tw2 = self.model.b2.twiss(
                compute_chromatic_properties=True, strengths=False
            )
            return self.get_params_from_twiss(tw1, tw2, full=full)
        elif mode == "from_twiss_init":
            tw1 = self.ir1.twiss_from_params(1)
            tw2 = self.ir1.twiss_from_params(2)
            ret = self.get_params_from_twiss(tw1, tw2, full=False)
            for ss in self.irs + self.arcs:
                retss = ss.get_params(mode="from_twiss_init")
                ret = ret and retss
        elif mode == "from_variables":
            return self.get_params_from_variables(full=full)
        else:
            raise ValueError("mode must be 'from_twiss' or 'from_variables'")

    def get_params_from_twiss(self, tw1, tw2, full=False):
        """
        Get the parameters from the twiss object.
        """
        params = {
            "p0c": tw1.p0c,
            "qxb1": tw1.qx,
            "qyb1": tw1.qy,
            "qxb2": tw2.qx,
            "qyb2": tw2.qy,
            "qpxb1": tw1.dqx,
            "qpyb1": tw1.dqy,
            "qpxb2": tw2.dqx,
            "qpyb2": tw2.dqy,
        }
        temp = {}
        if full:
            for ss in self.irs + self.arcs:
                pp = ss.get_params_from_twiss(tw1, tw2)
                params.update(pp)
            temp.update(params)
        else:
            temp.update(self.ir1.get_params())
            temp.update(self.ir5.get_params())
        for irn in [1, 5]:
            for xy in "xy":
                rname = f"r{xy}_ip{irn}"
                rr_ipb1 = temp[f"bet{xy}ip{irn}b1"] / tw1[f"bet{xy}", f"ip{irn}"]
                rr_ibb2 = temp[f"bet{xy}ip{irn}b2"] / tw2[f"bet{xy}", f"ip{irn}"]
                params[rname] = (rr_ipb1 + rr_ibb2) / 2
                if abs(rr_ipb1 - rr_ibb2) > 0.00001:
                    print(
                        f"Warning: r{xy}_ip{irn} from beam 1 and beam 2 differ by more than 0.00001: {rr_ipb1} vs {rr_ibb2}"
                    )
        return params

    def get_params_from_variables(self, full=False, verbose=False):
        """
        Get the parameters from the model variables.
        """
        params = {}
        if "p0c" in self.model:
            params["p0c"] = self.model["p0c"]
        elif "nrj" in self.model:
            params["p0c"] = self.model["nrj"] * 1e9
        else:
            params["p0c"] = self.model.get_p0c()
        var_names = (
            "qxb1",
            "qyb1",
            "qxb2",
            "qyb2",
            "qpxb1",
            "qpyb1",
            "qpxb2",
            "qpyb2",
        )
        for irn in [1, 5]:
            for plane in "xy":
                params[f"r{plane}_ip{irn}"] = self.model.get(f"r{plane}_ip{irn}", 1)

        for var_name in var_names:
            if var_name in self.model:
                params[var_name] = self.model[var_name]
        if full:
            if verbose:
                print("Getting parameters from variables:")
            for ss in self.irs + self.arcs:
                pp = ss.get_params_from_variables(verbose=verbose)
                print(len(pp))
                params.update(pp)
        return params

    def get_phase_arcs(self):
        """
        Get the phase advances from the arcs.
        """
        phases = {}
        for arc in self.arcs:
            phases.update(arc.get_phase())
        return phases

    def get_phase_irs(self):
        """
        Get the phase advances from the IRs.
        """
        phases = {}
        for ir in self.irs:
            phases.update(ir.get_phase())
        return phases

    def get_quad_max_ratio(self, verbose=False, ratio=1.5):
        """
        Get the maximum ratio of the quadrupole strengths in the IRs.
        """
        ratios = np.array(
            [ir.get_quad_max_ratio(verbose=verbose, ratio=ratio) for ir in self.irs]
        )
        return ratios.max()

    def get_quad_margin(self, name, verbose=False, p0c=None, absvalue=False):
        """
        Get the margin of the quadrupole strengths in the IRs.
        """
        if p0c is None:
            p0c = self.params["p0c"]
        v = self.model[name]
        p0c = self.params["p0c"]
        limits = self.circuits.get_klimits(name, p0c)
        if absvalue:
            kmin = min(abs(limits[0]), abs(limits[1]))
            kmax = max(abs(limits[0]), abs(limits[1]))
            margin0 = abs(abs(v) - kmin) / kmax
            margin1 = abs(kmax - abs(v)) / kmax
        else:
            maxv = max(abs(limits[0]), abs(limits[1]))
            margin0 = (v - limits[0]) / maxv
            margin1 = (limits[1] - v) / maxv
        return (margin0, margin1)

    def is_ats(self):
        """
        Check if the optics has ATS factors different from 1.
        """
        return (
            self.params.get("rx_ip1", 1) != 1
            or self.params.get("ry_ip1", 1) != 1
            or self.params.get("rx_ip5", 1) != 1
            or self.params.get("ry_ip5", 1) != 1
        )

    def match(self, verbose=False):
        """
        Rematch the entire optics
        """
        if not self.check_phase_params(verbose=verbose):
            raise ValueError(
                "Phase parameters do not sum up to the tunes, correct them first or set correct=True in check_phase_params()"
            )
        print("Match LHC optics")
        for aa in self.arcs:
            print(f"Match {aa.name.upper()}", end="")
            aa.match(verbose=verbose)
            print(" - done")
        for ir in self.irs:
            print(f"Match {ir.name.upper()}", end="")
            ir.match(verbose=verbose)
            print(" - done")

        print("Match chroma")
        self.model.match_chroma(arcs="weak", verbose=verbose)
        print("Match w")
        self.model.match_w(verbose=verbose)
        print("Match chroma")
        self.model.match_chroma(arcs="weak", verbose=verbose)
        self.check_params(verbose=verbose)
        self.match_knobs(verbose=verbose, fail=True)
        self.check()

    def match_chroma(self, arcs="weak", verbose=False):
        """
        match chroma and regenerate knobs

        arcs: weak, strong, all
        """
        self.model.match_chroma(arcs=arcs, beam=1, verbose=verbose, solve=True)
        self.model.match_chroma(arcs=arcs, beam=2, verbose=verbose, solve=True)
        for knob in self.find_knobs(f"dqp.*"):
            self.model.update_knob(knob)

    def match_knobs(self, verbose=True, fail=False):
        result = {}
        for knob in self.find_knobs():
            if hasattr(knob, "match"):
                try:
                    print(f"Match knob {knob.name}", end="")
                    knob.match(verbose=verbose)
                    if not verbose:
                         print(" - done")
                    result[knob.name] = "matched"
                except Exception as e:
                    print(f"Error matching knob {knob.name}: {e}")
                    result[knob.name] = e
                    if fail:
                        raise e
        return result

    def match_phase_arcs(self, newphases):
        for arc in self.arcs:
            phases = [k for k in newphases if arc.name in k]
            if len(phases) > 0:
                for k in phases:
                    print(f"Set {k!r} from {arc.params[k]} to {newphases[k]}")
                    arc.params[k] = newphases[k]
                arc.match_phase()

    def match_tune(self, qx=62.31, qy=60.32, arcs="all"):
        tw1, tw2 = self.twiss(strengths=False)
        dmuxb1 = qx - tw1.qx
        dmuyb1 = qy - tw1.qy
        dmuxb2 = qx - tw2.qx
        dmuyb2 = qy - tw2.qy
        if arcs == "all":
            arcs = self.arcs
        elif arcs == "noats":
            arcs = [self.a23, self.a34, self.a67, self.a78]
        print(f"Apply dmu: {dmuxb1:.3f} {dmuyb1:.3f} {dmuxb2:.3f} {dmuyb2:.3f}")
        narcs = len(arcs)
        print(f"in {narcs} arcs")
        self.check()
        for arc in arcs:
            arc.shift_phase(
                dmuxb1 / narcs, dmuyb1 / narcs, dmuxb2 / narcs, dmuyb2 / narcs
            )
            self.check()

    def plot(
        self,
        beam=None,
        figlabel=None,
        yr=None,
        yl=None,
        filename=None,
        iplabels=False,
        figsize=(6.4 * 1.2, 4.8),
    ):
        if beam is None:
            return [
                self.plot(
                    beam=1,
                    figlabel=figlabel,
                    yr=yr,
                    yl=yl,
                    iplabels=iplabels,
                    filename=filename,
                    figsize=figsize,
                ),
                self.plot(
                    beam=2,
                    figlabel=figlabel,
                    yr=yr,
                    yl=yl,
                    iplabels=iplabels,
                    filename=filename,
                    figsize=figsize,
                ),
            ]
        else:
            if figlabel is None:
                figlabel = f"LHCB{beam}"
            tw = self.twiss(beam=beam)
            plot = tw.plot(figlabel=figlabel, yr=yr, yl=yl, figsize=figsize)
            plot.ax.set_title(figlabel)
            if filename is not None:
                plot.savefig(filename.format(figlabel=figlabel))
            if iplabels:
                twip = tw.rows["ip."]
                plot.left.set_xticks(twip.s, map(str.upper, twip.name))
                plot.ax.set_xlabel(None)
            plot.figure.tight_layout()
        return plot

    def plot_crabbing(
        self, figlabel="Crabbing", filename=None, ax=None, figsize=(8, 4)
    ):
        tw1, tw2 = self.twiss()
        idx_ip = {
            "1b1": tw1.rows.indices["ip1"][0],
            "1b2": tw2.rows.indices["ip1"][0],
            "5b1": tw1.rows.indices["ip5"][0],
            "5b2": tw2.rows.indices["ip5"][0],
        }
        out = {}

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, num=figlabel)

        emit = 2.5e-6 / tw1.particle_on_co.gamma0[0]
        sigma_z = 0.075
        for s, bet, dx, hv, beam in [
            (tw1.s, tw1.betx, tw1.dx_zeta, "H", 1),
            (tw1.s, tw1.bety, tw1.dy_zeta, "V", 1),
            (tw2.s, tw2.betx, tw2.dx_zeta, "H", 2),
            (tw2.s, tw2.bety, tw2.dy_zeta, "V", 2),
        ]:
            sigma_x = np.sqrt(bet * emit)
            crabbing = dx * sigma_z / sigma_x
            ax.plot(s, crabbing, label=f"Beam {beam} {hv}")
            out[f"ip{beam}_{hv}_1"] = dx[idx_ip[f"1b{beam}"]] * 1e6
            out[f"ip{beam}_{hv}_5"] = dx[idx_ip[f"5b{beam}"]] * 1e6

        ax.set_xlabel("s [m]")
        ax.set_ylabel("$x\\;\\text{or}\\;y(z=\\sigma_z)/\\sigma_x$")
        plt.legend()
        for ip in ["1", "5"]:
            for beam in ["1", "2"]:
                print(
                    f"IP{ip} B{beam} Cxy= {out[f'ip{beam}_H_{ip}']:9.3f}, {out[f'ip{beam}_V_{ip}']:9.3f} urad"
                )
        return ax

    def plot_w(self, beam=None):
        if beam is None:
            return self.plot_w(beam=1), self.plot_w(beam=2)
        else:
            fig, ax = plt.subplots(
                figsize=(8, 4), num=f"LHCOptics {self.name} W B{beam}"
            )
            fig.clear()
            tw = self.model.twiss(beam=beam, strengths=False, chrom=True, start="ip3")
            tw.plot("wx_chrom wy_chrom", ax=ax)
            set_ip_labels(ax, tw)

    def round_params(
        self, full=True, verbose=True, dryrun=False, qx=62.31, qy=60.32, qp=0.0
    ):
        if dryrun:
            verbose = True
        if full:
            for ss in self.irs + self.arcs:
                ss.round_params(verbose=verbose, dryrun=dryrun)
        for xy in "xy":
            for beam in [1, 2]:
                if verbose:
                    if self.params[f"q{xy}b{beam}"] != (qx if xy == "x" else qy):
                        print(
                            f"Round q{xy}b{beam} from {self.params[f'q{xy}b{beam}']} to {qx if xy == 'x' else qy}"
                        )
                    if self.params[f"qp{xy}b{beam}"] != qp:
                        print(
                            f"Round qp{xy}b{beam} from {self.params[f'qp{xy}b{beam}']} to {qp}"
                        )
                if not dryrun:
                    self.params[f"q{xy}b{beam}"] = qx if xy == "x" else qy
                    self.params[f"qp{xy}b{beam}"] = qp
        ## round ATS factors is not straightforward because they are computed from the 
        ## ratio of the beta functions at the IPs, so we need to compute them from the 
        ## twiss and check that the values from beam 1 and beam 2 are consistent
        ## also the value cannot be rounded as 1/3 is common 
        # for xy in "xy":
        #     for irn in [1, 5]:
        #         rname = f"r{xy}_ip{irn}"
        #         tw1 = self.twiss(1, strengths=False, chrom=False)
        #         tw2 = self.twiss(2, strengths=False, chrom=False)
        #         rr_ipb1 = self.irs[irn - 1][f"bet{xy}ip{irn}b1"] / np.round(
        #             tw1[f"bet{xy}", f"ip{irn}"], 8
        #         )
        #         rr_ibb2 = self.irs[irn - 1][f"bet{xy}ip{irn}b2"] / np.round(
        #             tw2[f"bet{xy}", f"ip{irn}"], 8
        #         )
        #         if abs(rr_ipb1 - rr_ibb2) > 0.00001:
        #             raise ValueError(
        #                 f"Cannot round r{xy}_ip{irn} to 1.0 because beam 1 and beam 2 values differ by more than 0.00001: {rr_ipb1} vs {rr_ibb2}"
        #             )
        #         if verbose:
        #             print(f"Round {rname} from {self.params[rname]} to {rr_ipb1}")
        #         if not dryrun:
        #             self.params[rname] = rr_ipb1

    def set_ats_params(self, bet_cross, bet_sep=None, flat="hv", verbose=True):
        """
        Set the ATS parameters from the beta functions at the IPs.
        """
        if bet_sep is None:
            bet_sep = bet_cross
        if flat[0]=="h":
            self.params["rx_ip1"] = self.ir1.params['betxip1b1'] / bet_cross
            self.params["ry_ip1"] = self.ir1.params['betxip1b1'] / bet_sep
        elif flat[0]=="v":
            self.params["rx_ip1"] = self.ir1.params['betxip1b1'] / bet_sep
            self.params["ry_ip1"] = self.ir1.params['betxip1b1'] / bet_cross
        else:
            raise ValueError("Flat an only be 'h' or 'v'")
        if flat[1]=="v":
            self.params["rx_ip5"] = self.ir5.params['betxip5b1'] / bet_sep
            self.params["ry_ip5"] = self.ir5.params['betxip5b1'] / bet_cross
        elif flat[1]=="h":
            self.params["rx_ip5"] = self.ir5.params['betxip5b1'] / bet_cross
            self.params["ry_ip5"] = self.ir5.params['betxip5b1'] / bet_sep
        else:
            raise ValueError("Flat an only be 'h' or 'v'")
        if verbose:
            print(
                f"Set rx_ip1={self.params['rx_ip1']}, ry_ip1={self.params['ry_ip1']}, rx_ip5={self.params['rx_ip5']}, ry_ip5={self.params['ry_ip5']}"
            )




    def set_bumps(self, bumps, verbose=False):
        """Set the bump parameters."""
        model = self.model
        for k, v in bumps.items():
            if verbose:
                if k in model and model[k] != v:
                    print(f"Set bump {k} from {model[k]} to {v}")
            model[k] = v

    def set_bumps_off(self):
        for ir in self.irs:
            ir.set_bumps_off()
        for k in self.knobs:
            if re.match(r"on_d(sep|x|o|a)", k):
                self.model[k] = 0

    def set_circuits(self, circuits):
        if isinstance(circuits, str) or isinstance(circuits, Path):
            self.circuits = LHCCircuits.from_json(circuits)
        else:
            self.circuits = circuits
        return self

    def set_init(self):
        for ir in self.irs:
            ir.set_init()

    def set_knobs_off(self):
        for k in self.knobs:
            self.model[k] = 0
        for sec in self.irs + self.arcs:
            sec.set_knobs_off()

    def set_knobs_on(self):
        for k, knob in self.knobs.items():
            self.model[k] = knob.value
        for sec in self.irs + self.arcs:
            sec.set_knobs_on()

    def set_madx_model(self, model):
        if Path(model).exists():
            model = LHCMadxModel.from_madxfile(model)
        self.model = model
        self.update_model()
        return self

    def set_params(self, full=True, mode="from_twiss_init", verbose=False):
        """
        Copy all parameters from get_params() into params
        """
        if verbose:
            print(f"Setting parameters from mode {mode} with full={full}")
        self.params.update(self.get_params(mode=mode, verbose=verbose))
        if full:
            for ss in self.irs + self.arcs:
                ss.set_params(mode=mode, verbose=verbose)
        return self

    def set_xsuite_model(self, model, verbose=False):
        if isinstance(model, str) or isinstance(model, Path):
            model = LHCXsuiteModel.from_json(model)
        self.model = model
        self.update_model(verbose=verbose)
        return self

    def test_coupling_knobs(self):
        self.model.cycle("ip7")
        for ri, ii in zip("ri", (0, 1)):
            for beam in [1, 2]:
                for ext in ["_op", "_sq", ""]:
                    name = f"cm{ri}s.b{beam}{ext}"
                    oldvalue = self.model[name]
                    self.model[name] = 0.001
                    cmim = self.get_cmin(beam=beam)[ii]
                    print(f"{name:10} {cmim - 0.001:.3f}")
                    self.model[name] = oldvalue

    def test_tune_knobs(self):
        for beam in [1, 2]:
            for ext in ["_op", "_sq", ""]:
                for xy in "xy":
                    name = f"dq{xy}.b{beam}{ext}"
                    oldtune = self.twiss(beam=beam)[f"q{xy}"]
                    oldvalue = self.model[name]
                    self.model[name] = 0.01
                    tune = self.twiss(beam=beam)[f"q{xy}"]
                    print(
                        f"{name:10} {oldtune:.3f} {tune:.3f} {tune - oldtune - 0.01:.3f}"
                    )
                    self.model[name] = oldvalue

    def to_dict(self):
        return {
            "name": self.name,
            "irs": [ir.to_dict() for ir in self.irs],
            "arcs": [arc.to_dict() for arc in self.arcs],
            "params": self.params,
            "knobs": {n: k.to_dict() for n, k in self.knobs.items()},
            "variant": self.variant,
        }

    def to_json(self, filename):
        with open(filename, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def to_madx(self, output=None):
        out = []
        out.append(f"! {self.name.upper()}\n")

        if len(self.params) > 0:
            out.append("! Main Parameters")
            for k, v in self.params.items():
                out.append(f"{k:30} = {v:19.16f};")
            out.append("")

        for ss in self.irs + self.arcs:
            out.extend(ss.to_madx(output=list, knobs=False))

        knobs = self.find_knobs()
        if len(knobs) > 0:
            strengths = self.find_strengths()
            out.append("! Knobs")
            for expr in LHCMadxModel.knobs_to_expr(knobs, strengths):
                out.append(expr)
            out.append("")

        out.append("! Constant definitions\n")
        if self.variant.startswith("hl"):
            out.append(LHCMadxModel.extra_defs_hllhc)
        else:
            out.append(LHCMadxModel.extra_defs)
        return deliver_list_str(out, output)

    def to_table(self, *rows):
        from .opttable import LHCOpticsTable

        return LHCOpticsTable([self.copy()] + list(rows))

    def twiss(self, beam=None, chrom=False, strengths=True):
        if beam is None:
            return [
                self.twiss(beam=1, strengths=strengths, chrom=chrom),
                self.twiss(beam=2, strengths=strengths, chrom=chrom),
            ]
        return getattr(self.model, f"b{beam}").twiss(
            compute_chromatic_properties=chrom, strengths=strengths
        )

    def update(
        self,
        src=None,
        verbose=False,
        full=True,
        add_params=True,
        knobs=True,
        params=True,
        strengths=True,
    ):
        if isinstance(src, str) or isinstance(src, Path):
            src = self.from_json(src)
        if full:
            if src is None:
                if verbose:
                    print(f"Update {self} from model")
                for ss in self.irs + self.arcs:
                    ss.update(
                        src,
                        verbose=verbose,
                        add_params=add_params,
                        knobs=knobs,
                        params=params,
                        strengths=strengths,
                    )
            else:
                for ss in self.irs + self.arcs:
                    if hasattr(src, ss.name):
                        src_ss = getattr(src, ss.name)
                        if verbose:
                            print(f"Update {ss.name} from {src}.{ss.name}")
                        ss.update(
                            src=src_ss,
                            verbose=verbose,
                            add_params=add_params,
                            knobs=knobs,
                            params=params,
                            strengths=strengths,
                        )
                    elif ss.name in src:
                        src_ss = src[ss.name]
                        if verbose:
                            print(f"Update {ss.name} from {src}[{ss.name}]")
                        ss.update(
                            src=src_ss,
                            verbose=verbose,
                            add_params=add_params,
                            knobs=knobs,
                            params=params,
                            strengths=strengths,
                        )
        if params:
            self.update_params(src, verbose=verbose, full=False, add=True)
        if knobs:
            self.update_knobs(src, verbose=verbose, full=False)
        return self

    def update_knobs(self, src=None, full=True, verbose=False):
        """
        Update optics knobs from src, if full incluiding all sections knobs
        """
        if src is None:
            src = self.model
        if hasattr(src, "knobs"):
            knobs_dict = src.knobs
        elif hasattr(src, "get_knob"):
            knobs_dict = {k: src.get_knob(knob) for k, knob in self.knobs.items()}
        elif src == "default":
            if verbose:
                print("Update knobs from default list")
            knobs_dict = {
                k: self.model.get_knob_by_probing(k)
                for k in self.get_default_knob_names()
            }
        for k in self.knobs:
            if k in knobs_dict:
                if verbose:
                    self.knobs[k].print_update_diff(knobs_dict[k])
                self.knobs[k] = Knob.from_src(knobs_dict[k])
                self.knobs[k].parent = self
        if full:
            for ss in self.irs + self.arcs:
                if hasattr(src, ss.name):
                    src_ss = getattr(src, ss.name)
                    ss.update_knobs(src=src_ss, verbose=verbose)
                elif ss.name in src:
                    src_ss = src[ss.name]
                    ss.update_knobs(src=src_ss, verbose=verbose)
                else:
                    ss.update_knobs(src=src, verbose=verbose)
        return self

    def update_model(
        self,
        src=None,
        full=True,
        verbose=False,
        set_init=True,
        knobs="create",
        set_knob_values=False,
    ):
        """
        Update model from an optics or a dict.
        If full incluiding all sections strengths and knobs.
        """
        if self.model is None:
            raise ValueError("Model not set")
        if src is None:
            src = self
        if full:
            for ss in self.irs + self.arcs:
                if hasattr(src, ss.name):
                    src_ss = getattr(src, ss.name)
                    lbl = f"{src}.{ss.name}"
                elif ss.name in src:
                    src_ss = src[ss.name]
                    lbl = f"{src}[{ss.name!r}]"
                if verbose:
                    print(f"Update strengths and knobs in {ss.name!r} from {lbl}")
                ss.update_model(
                    src=src_ss,
                    verbose=verbose,
                    knobs=knobs,
                    set_knob_values=set_knob_values,
                )
        # knobs must be updated after the strengths
        if knobs == "create":
            if verbose:
                print(f"Update knobs from {src}")
            self.model.create_knobs(
                src.knobs,
                verbose=verbose,
                set_value=set_knob_values,
            )
        elif knobs == "update":
            if verbose:
                print(f"Update knobs from {src}")
            self.model.update_knobs(
                src.knobs,
                verbose=verbose,
                set_value=set_knob_values,
            )
        elif knobs == None or knobs == False:
            pass
        else:
            raise ValueError(
                f"knobs must be 'create', 'update', None or False not {knobs!r}"
            )

        if "p0c" in self.params:
            self.model.p0c = self.params["p0c"]
        if set_init:
            self.set_init()
        return self

    def update_params(self, src=None, add=False, verbose=False, full=True):
        """
        Update existing params from self.model or src.params or src
        """
        # if verbose:
        #    print(f"Update params in {self}")
        if src is None:
            src = self.get_params()
        elif hasattr(src, "params"):
            src = src.params
        if add:
            self.params.update(src)
        else:
            for k in self.params:
                if k in src:
                    if verbose and self.params[k] != src[k]:
                        print(
                            f"Updating {k!r:15} from {self.params[k]:15.6g} to {src[k]:15.6g}"
                        )
                    self.params[k] = src[k]
        if full:
            for ss in self.irs + self.arcs:
                # if verbose:
                # print(f"Update params in {ss}")
                ssrc = src.get(ss.name)
                ss.update_params(ssrc, verbose=verbose, add=add)
        return self
