import json
import re
from pathlib import Path

import numpy as np
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
from .opttable import LHCOpticsTable
from .section import Knob
from .utils import (
    deliver_list_str,
    print_diff_dict_float,
    print_diff_dict_objs,
    git_get_current_branch,
    git_set_branch,
)

_opl = ["_op", "_sq", ""]


class LHCOptics:
    """
    Optics containts global knobs, global parameters and sections
    Section contains strengths, local knobs, local parameters
    """

    _arcs = ["a12", "a23", "a34", "a45", "a56", "a67", "a78", "a81"]
    _irs = [LHCIR1, LHCIR2, LHCIR3, LHCIR4, LHCIR5, LHCIR6, LHCIR7, LHCIR8]

    knob_names = [f"dq{x}.b{b}{op}" for x in "xy" for b in "12" for op in _opl]
    knob_names += [
        f"dqp{x}.b{b}{op}" for x in "xy" for b in "12" for op in _opl
    ]
    knob_names += [
        f"cm{x}s.b{b}{op}" for x in "ir" for b in "12" for op in _opl
    ]
    knob_names += [
        f"{kk}.b{b}"
        for kk in ["on_mo", "phase_change", "dp_trim"]
        for b in "12"
    ]
    knob_names += ["on_ssep1_h", "on_xx1_v", "on_ssep5_v", "on_xx5_h"]

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
            globals()[f"LHCIR{n+1}"].from_dict(d)
            for n, d in enumerate(data["irs"])
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
        with open(filename) as f:
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
        name=None,
        sliced=False,
        gen_model=None,
        xsuite_model=None,
        circuits=None,
        verbose=False,
    ):
        madmodel = LHCMadxModel(madx)
        knobs = madmodel.make_and_set0_knobs(cls.knob_names)
        irs = [ir.from_madx(madx) for ir in cls._irs]
        arcs = [LHCArc.from_madx(madx, arc) for arc in cls._arcs]
        for k, knob in knobs.items():
            madx.globals[k] = knob.value
        self = cls(name, irs, arcs, knobs=knobs)
        if gen_model == "xsuite":
            xsuite_model = LHCXsuiteModel.from_madx(madx, sliced=sliced)
        elif gen_model == "madx":
            self.model = madmodel
        if xsuite_model is not None:
            self.set_xsuite_model(xsuite_model, verbose=verbose)
        if circuits is not None:
            self.set_circuits(circuits)
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
    ):
        madx = Madx(stdout=stdout)
        madx.call(filename)
        if name is None:
            name = str(filename)
        return cls.from_madx(
            madx,
            name=name,
            sliced=sliced,
            gen_model=make_model,
            xsuite_model=xsuite_model,
            verbose=verbose,
        )

    @staticmethod
    def set_repository(version="2024"):
        import subprocess

        version = str(version)

        accmodels = Path("acc-models-lhc")
        if accmodels.exists():
            if not (accmodels / "lhc.seq").exists():
                raise FileNotFoundError("acc-models-lhc/lhc.seq not found")
            else:
                if (accmodels / ".git").exists():
                    if git_get_current_branch(accmodels) != version:
                        git_set_branch(accmodels, version)
        elif (
            lcl := (Path.home() / "local" / "acc-models-lhc" / version)
        ).exists():
            accmodels.symlink_to(lcl)
        else:
            subprocess.run(
                [
                    "git",
                    "clone",
                    "https://gitlab.cern.ch/acc-models/lhc.git",
                    f"-b {version}" "acc-models-lhc",
                ]
            )

    def __init__(
        self,
        name,
        irs,
        arcs,
        params=None,
        knobs=None,
        model=None,
        circuits=None,
        aperture=None,
    ):
        if name is None:
            name = "lhcoptics"
        self.name = name
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
        print(f"Optics {self.name} created")
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
        return [getattr(self, arc) for arc in self._arcs]

    @property
    def irs(self):
        return [getattr(self, ir.name) for ir in self._irs]

    def check(self):
        tw1, tw2 = self.twiss(chrom=True, strengths=False)
        header = True
        cols = "betx bety dx dpx px*1e6 py*1e6 x*1e3 y*1e3"
        for ip in ["ip1", "ip2", "ip5", "ip8"]:
            tw1.rows[ip].cols[cols].show(digits=4, fixed="f", header=header)
            if header:
                header = False
            tw2.rows[ip].cols[cols].show(digits=4, fixed="f", header=header)
        print(
            f"Tunes:  {tw1.qx:9.6f} {tw2.qx:9.6f} {tw1.qy:9.6f} {tw2.qy:9.6f}"
        )
        print(
            f"Chroma: {tw1.dqx:9.6f} {tw2.dqx:9.6f} {tw1.dqy:9.6f} {tw2.dqy:9.6f}"
        )
        return self

    def check_quad_strengths(
        self, verbose=False, p0c=None, ratio_threshold=1.5
    ):
        for ir in self.irs:
            ir.check_quad_strengths(
                verbose=verbose, p0c=p0c, ratio_threshold=ratio_threshold
            )

    def check_match(self):
        for ss in self.irs + self.arcs:
            opt = ss.match()
            print(f"{ss.name} {opt.within_tol}")

    def copy(self):
        other = self.__class__(
            self.name,
            params=self.params.copy(),
            knobs={k: v.copy() for k, v in self.knobs.items()},
            irs=[ir.copy() for ir in self.irs],
            arcs=[arc.copy() for arc in self.arcs],
            circuits=self.circuits,
        )
        return other

    def diff(self, other, full=True):
        self.diff_knobs(other)
        self.diff_params(other)
        if full:
            for ss, so in zip(self.irs + self.arcs, other.irs + other.arcs):
                ss.diff_strengths(so)
                ss.diff_knobs(so)
                ss.diff_params(so)

    def diff_knobs(self, other):
        print_diff_dict_objs(self.knobs, other.knobs)

    def diff_params(self, other):
        print_diff_dict_float(self.params, other.params)

    def find_strengths(self, regexp=None):
        strengths = {}
        for ss in self.irs + self.arcs:
            strengths.update(ss.strengths)
        if regexp is not None:
            strengths = {
                k: v for k, v in strengths.items() if re.match(regexp, k)
            }
        return strengths

    def find_knobs(self, regexp=None):
        knobs = {}
        for ss in self.irs + self.arcs:
            knobs.update(ss.knobs)
        knobs.update(self.knobs)
        if regexp is not None:
            knobs = {k: v for k, v in knobs.items() if re.match(regexp, k)}
        return knobs.values()

    def find_knobs_null(self):
        knobs = self.find_knobs()
        return {
            knob
            for knob in knobs.items()
            if sum(map(abs, knob.weights.values())) == 0
        }

    def get(self, k, default=None, full=True):
        if k in self:
            return self[k]
        if full:
            for ss in self.irs + self.arcs:
                if k in ss:
                    return ss[k]
        return default

    def get_cmin(self, beam=None, pos="ip1"):
        if beam is None:
            return [
                self.get_cmin(beam=1, pos=pos),
                self.get_cmin(beam=2, pos=pos),
            ]
        line = self.model.sequence[beam]
        if line.element_names[0] != pos:
            line.cycle(pos, inplace=True)
        tw = line.twiss(compute_chromatic_properties=False, strengths=True)
        # print(tw.name)
        k1sl = tw["k1sl"]
        pi2 = 2 * np.pi
        j2pi = 1j * pi2
        cmin = (
            np.sum(
                k1sl
                * np.sqrt(tw.betx * tw.bety)
                * np.exp(j2pi * (tw.mux - tw.muy))
            )
            / pi2
        )
        if line.element_names[0] != "ip1":
            line.cycle("ip1", inplace=True)
        return cmin.real, cmin.imag

    def get_params(self):
        tw1 = self.model.b1.twiss(
            compute_chromatic_properties=True, strengths=False
        )
        tw2 = self.model.b2.twiss(
            compute_chromatic_properties=True, strengths=False
        )
        return self.get_params_from_twiss(tw1, tw2)

    def get_params_from_twiss(self, tw1, tw2):
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
        # for ss in self.irs + self.arcs:
        #    params.update(ss.get_params_from_twiss(tw1, tw2))
        return params

    def get_phase_arcs(self):
        phases = {}
        for arc in self.arcs:
            phases.update(arc.get_phase())
        return phases

    def get_quad_max_ratio(self, verbose=False, ratio_threshold=1.5):
        ratios = np.array(
            [
                ir.get_quad_max_ratio(
                    verbose=verbose, ratio_threshold=ratio_threshold
                )
                for ir in self.irs
            ]
        )
        return ratios.max()

    def is_ats(self):
        return (
            self.params.get("rx_ip1", 1) != 1
            or self.params.get("ry_ip1", 1) != 1
            or self.params.get("rx_ip5", 1) != 1
            or self.params.get("ry_ip5", 1) != 1
        )

    def match_chroma(self, beam=None, dqx=0, dqy=0, arcs="all", solve=True):
        """
        NB: breaks knobs and restore them
        """
        if beam is None:
            for beam in [1, 2]:
                self.match_chroma(beam, dqx, dqy, arcs, solve=solve)
        else:
            model = self.model
            xt = model._xt
            beam = f"b{beam}"
            line = getattr(model, beam)
            for fd in "fd":
                for ks in self.find_strengths(f"ks{fd}.*{beam}"):
                    tmp = f"ks{fd}_{beam}"
                    model[tmp] = model[ks]
                    print(f"Set {tmp} from {ks} to {model[tmp]}")
                    model.vars[ks] = model.vars[tmp]  # expr
            mtc = line.match(
                solve=solve,
                vary=[xt.VaryList([f"ksf_{beam}", f"ksd_{beam}"], step=1e-9)],
                targets=[xt.TargetSet(dqx=dqx, dqy=dqy, tol=1e-6)],
                strengths=False,
                compute_chromatic_properties=True,
                n_steps_max=50,
            )
            mtc.target_status()
            mtc.vary_status()
            for knob in self.find_knobs(f"dqp.*{beam}"):
                model.update_knob(knob)
            return mtc

    def match_knobs(self):
        for knob in self.find_knobs():
            if hasattr(knob, "match"):
                knob.match()

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
        print(
            f"Apply dmu: {dmuxb1:.3f} {dmuyb1:.3f} {dmuxb2:.3f} {dmuyb2:.3f}"
        )
        narcs = len(arcs)
        print(f"in {narcs} arcs")
        self.check()
        for arc in arcs:
            arc.shift_phase(
                dmuxb1 / narcs, dmuyb1 / narcs, dmuxb2 / narcs, dmuyb2 / narcs
            )
            self.check()

    def plot(self, beam=None):
        if beam is None:
            for beam in [1, 2]:
                self.plot(beam)
        else:
            self.twiss(beam=beam).plot()
        return self

    def round_params(self, full=False):
        if full:
            for ss in self.irs + self.arcs:
                ss.round_params()
        for k, v in self.params.items():
            self.params[k] = round(v, 6)

    def set_bumps_off(self):
        for ir in self.irs:
            ir.set_bumps_off()

    def set_circuits(self, circuits):       
        if isinstance(circuits, str) or isinstance(circuits, Path):
            from .circuits import LHCCircuits
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

    def set_params(self, full=True):
        """
        Copy all parameters from get_params() into params
        """
        self.params.update(self.get_params())
        if full:
            for ss in self.irs + self.arcs:
                ss.set_params()

    def set_xsuite_model(self, model, verbose=False):
        if isinstance(model, str):
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
                    print(f"{name:10} {cmim-0.001:.3f}")
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
                        f"{name:10} {oldtune:.3f} {tune:.3f} {tune-oldtune-0.01:.3f}"
                    )
                    self.model[name] = oldvalue

    def to_dict(self):
        return {
            "name": self.name,
            "irs": [ir.to_dict() for ir in self.irs],
            "arcs": [arc.to_dict() for arc in self.arcs],
            "params": self.params,
            "knobs": {n: k.to_dict() for n, k in self.knobs.items()},
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
        out.append(LHCMadxModel.extra_defs)
        return deliver_list_str(out, output)

    def to_table(self, *rows):
        return LHCOpticsTable([self] + list(rows))

    def twiss(self, beam=None, chrom=False, strengths=True):
        if beam is None:
            return [
                self.twiss(beam=1, strengths=strengths, chrom=chrom),
                self.twiss(beam=2, strengths=strengths, chrom=chrom),
            ]
        return getattr(self.model, f"b{beam}").twiss(
            compute_chromatic_properties=chrom, strengths=strengths
        )

    def update(self, src=None, verbose=False, full=True):
        if isinstance(src, str) or isinstance(src, Path):
            src = self.from_json(src)
        if full:
            if src is None:
                if verbose:
                    print(f"Update {self} from model")
                for ss in self.irs + self.arcs:
                    ss.update(getattr(src,ss.name), verbose=verbose)
            else:
                for ss in self.irs + self.arcs:
                    if hasattr(src, ss.name):
                        src_ss = getattr(src, ss.name)
                        if verbose:
                            print(f"Update {ss.name} from {src}.{ss.name}")
                        ss.update(src=src_ss, verbose=verbose)
                    elif ss.name in src:
                        src_ss = src[ss.name]
                        if verbose:
                            print(f"Update {ss.name} from {src}[{ss.name}]")
                        ss.update(src=src_ss, verbose=verbose)
        self.update_params(src, verbose=verbose, full=False)
        self.update_knobs(src, verbose=verbose, full=False)
        return self

    def update_knobs(self, src=None, full=True, verbose=False):
        """
        Update optics knobs from src, if full incluiding all sections nobs
        """
        if src is None:
            src = self.model
        if hasattr(src, "knobs"):
            src = src.knobs
        elif hasattr(src, "get_knob"):
            src = {k: src.get_knob(knob) for k, knob in self.knobs.items()}
        elif src == "default":
            if verbose:
                print("Update knobs from default list")
            src = {
                k: self.model.get_knob_by_probing(k)
                for k in self.get_default_knob_names()
            }
        for k in self.knobs:
            if k in src:
                if verbose:
                    self.knobs[k].print_update_diff(src[k])
                self.knobs[k] = Knob.from_src(src[k])
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
        knobs_off=False,
        set_init=True,
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
                    print(
                        f"Update strengths and knobs in {ss.name!r} from {lbl}"
                    )
                ss.update_model(
                    src=src_ss, verbose=verbose, knobs_off=knobs_off
                )
        # knobs must be updated after the strengths
        if verbose:
            print(f"Update knobs from {src}")
        self.model.update_knobs(
            src.knobs, verbose=verbose, knobs_off=knobs_off
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
                ss.update_params(ssrc, verbose=verbose)
        return self
