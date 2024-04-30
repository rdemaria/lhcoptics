from cpymad.madx import Madx
import json
from pathlib import Path

from .ir1 import LHCIR1
from .ir2 import LHCIR2
from .ir3 import LHCIR3
from .ir4 import LHCIR4
from .ir5 import LHCIR5
from .ir6 import LHCIR6
from .ir7 import LHCIR7
from .ir8 import LHCIR8
from .arcs import LHCArc

from .section import Knob
from .model_xsuite import LHCXsuiteModel, LHCMadModel
from .circuits import LHCCircuits

irs = [LHCIR1, LHCIR2]

_opl = ["_op", "_sq", ""]


class LHCOptics:
    """
    Optics containts global knobs, global parameters and sections
    Section contains strengths, local knobs, local parameters
    """

    @staticmethod
    def set_repository(version="2024"):
        import subprocess
        import os

        accmodels = Path("acc-models-lhc")
        if accmodels.exists():
            if not (accmodels / "lhc.seq").exists():
                raise FileNotFoundError("acc-models-lhc/lhc.seq not found")
            else:
                if (accmodels / ".git").exists():
                    subprocess.run(["git", "switch", version], cwd=accmodels)
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
                    "acc-models-lhc",
                ]
            )

    _irs = [LHCIR1, LHCIR2, LHCIR3, LHCIR4, LHCIR5, LHCIR6, LHCIR7, LHCIR8]
    _arcs = ["a12", "a23", "a34", "a45", "a56", "a67", "a78", "a81"]

    knobs = [f"dq{x}.b{b}{op}" for x in "xy" for b in "12" for op in _opl]
    knobs += [f"dqp{x}.b{b}{op}" for x in "xy" for b in "12" for op in _opl]
    knobs += [f"cm{x}s.b{b}{op}" for x in "ir" for b in "12" for op in _opl]
    knobs += [
        f"{kk}.b{b}"
        for kk in ["on_mo", "phase_change", "dp_trim"]
        for b in "12"
    ]
    knobs += ["on_ssep1_h", "on_xx1_v", "on_ssep5_v", "on_xx5_h"]

    def __init__(self, name, irs, arcs, params=None, knobs=None, model=None):
        self.name = name
        for ir in irs:
            setattr(self, ir.name, ir)
            ir.parent = self
        for arc in arcs:
            setattr(self, arc.name, arc)
            arc.parent = self
        self.irs = irs
        self.arcs = arcs
        if params is None:
            params = {}
        if knobs is None:
            knobs = {}
        self.params = params
        self.knobs = knobs
        self.model = model

    @classmethod
    def from_madxfile(
        cls,
        filename,
        name="lhcoptics",
        sliced=False,
        gen_model=None,
        xsuite_model=None,
    ):
        madx = Madx()
        madx.call(filename)
        return cls.from_madx(
            madx,
            name=name,
            sliced=sliced,
            gen_model=gen_model,
            xsuite_model=xsuite_model,
        )

    @classmethod
    def from_madx(
        cls,
        madx,
        name="lhcoptics",
        sliced=False,
        gen_model=None,
        xsuite_model=None,
        circuits=None,
    ):
        madmodel = LHCMadModel(madx)
        knobs = madmodel.make_and_set0_knobs(cls.knobs)
        irs = [ir.from_madx(madx) for ir in cls._irs]
        arcs = [LHCArc.from_madx(madx, arc) for arc in cls._arcs]
        for k, knob in knobs.items():
            madx.globals[k] = knob.value
        self = cls(name, irs, arcs, knobs=knobs)
        if gen_model == "xsuite":
            gen_model = LHCXsuiteModel.from_madx(madx, sliced=sliced)
            self.model = gen_model
        elif gen_model == "madx":
            self.model = madmodel
        elif xsuite_model is not None:
            self.set_xsuite_model(xsuite_model)
        if circuits is not None:
            self.set_circuits_from_json(circuits)
        return self

    @classmethod
    def from_dict(cls, data, xsuite_model=None, circuits=None):
        irs = [
            globals()[f"LHCIR{n+1}"].from_dict(d)
            for n, d in enumerate(data["irs"])
        ]
        arcs = [LHCArc.from_dict(d) for d in data["arcs"]]
        if isinstance(xsuite_model, str) or isinstance(xsuite_model, Path):
            xsuite_model = LHCXsuiteModel.from_json(xsuite_model)
        if isinstance(circuits, str) or isinstance(circuits, Path):
            circuits = LHCCircuits.from_json(circuits)
        return cls(
            name=data["name"],
            irs=irs,
            arcs=arcs,
            params=data["params"],
            knobs={k: Knob.from_dict(d) for k, d in data["knobs"].items()},
            model=xsuite_model)

    @classmethod
    def from_json(cls, filename, xsuite_model=None, circuits=None):
        with open(filename) as f:
            data = json.load(f)
            out=cls.from_dict(data, xsuite_model=xsuite_model, circuits=circuits)
            out.update_model()
            return out

    def __repr__(self) -> str:
        return f"<LHCOptics {self.name!r}>"

    def to_dict(self):
        return {
            "name": self.name,
            "irs": [ir.to_dict() for ir in self.irs],
            "arcs": [arc.to_dict() for arc in self.arcs],
            "params": self.params,
            "knobs": {n: k.to_dict() for n, k in self.knobs.items()},
        }

    def copy(self):
        return self.__class__.from_dict(self.to_dict())

    def to_json(self, filename):
        with open(filename, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def update_model(self, src=None, full=True):
        """
        Update model knobs, if full incluiding all sections strengths and knobs
        """
        if self.model is None:
            raise ValueError("Model not set")
        if src is None:
            src = self
        src.model.update_knobs(self.knobs)
        if full:
            for ss in src.irs + src.arcs:
                if hasattr(src, ss.name):
                    src_ss = getattr(src, ss.name)
                    ss.update_model(src=src_ss)
                elif ss.name in src:
                    src_ss = src[ss.name]
                    ss.update_model(src=src_ss)
        return self

    def update_knobs(self, src=None, full=True):
        """
        Update optics knobs from src, if full incluiding all sections nobs
        """
        if src is None:
            src = self.model
        if hasattr(src, "knobs"):
            src=src.knobs
        elif hasattr(src, "get_knob"):
            src={k: src.get_knob(knob) for k,knob in self.knobs.items()}
        for k in self.knobs:
            self.knobs[k] = Knob.from_src(src[k])
        if full:
            for ss in self.irs + self.arcs:
                if hasattr(src, ss.name):
                    src_ss = getattr(src, ss.name)
                    ss.update_knobs(src=src_ss)
                elif ss.name in src:
                    src_ss = src[ss.name]
                    ss.update_knobs(src=src_ss)
        return self

    def update(self,src=None):
        self.update_knobs(src)
        self.update_params(src)
        for ss in self.irs + self.arcs:
            ss.update(src)
        return self
    
    def update_params(self, src=None, add=False):
        """
        Update existing params from self.model or src.params or src
        """
        if src is None:
            src = self.get_params()
        elif hasattr(src, "params"):
            src = src.params
        if add:
            self.params.update(src)
        else:
            for k in self.params:
                if k in src:
                    self.params[k] = src[k]
        return self

    def set_xsuite_model(self, model):
        if isinstance(model, str):
            model = LHCXsuiteModel.from_json(model)
        self.model = model
        self.update_model()
        return self

    def set_madx_model(self, model):
        if Path(model).exists():
            model = LHCMadModel.from_madxfile(model)
        self.model = model
        self.update_model()
        return self

    def set_circuits_from_json(self, filename):
        from .circuits import LHCCircuits

        self.circuits = LHCCircuits.from_json(filename)
        return self

    def get_params(self):
        tw1 = self.model.b1.twiss(compute_chromatic_properties=True,strengths=False)
        tw2 = self.model.b2.twiss(compute_chromatic_properties=True,strengths=False)
        return self.get_params_from_twiss(tw1, tw2)

    def get_params_from_twiss(self, tw1, tw2):
        params = {
            "pc0": tw1.p0c,
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

    def set_params(self, full=True):
        """
        Copy all parameters from get_params() into params
        """
        self.params.update(self.get_params())
        if full:
            for ss in self.irs + self.arcs:
                ss.set_params()

    def twiss(self, beam=None, chrom=False,strengths=True):
        if beam is None:
            return [self.twiss(beam=1,strenghts=strengths), self.twiss(beam=2,strengths=strengths)]
        return getattr(self.model,f"b{beam}").twiss(compute_chromatic_properties=chrom,strenghts=strengths)

    def plot(self, beam=None):
        if beam is None:
            for beam in [1, 2]:
                self.plot(beam)
        else:
            self.twiss(beam=beam).plot()
        return self

    def disable_bumps(self):
        for ir in self.irs:
            ir.disable_bumps()



