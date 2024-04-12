from cpymad.madx import Madx
import json

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

irs = [LHCIR1, LHCIR2]

_opl = ["_op", "_sq", ""]


class LHCOptics:
    """
    Optics containts global knobs, global parameters and sections
    Section contains strengths, local knobs, local parameters
    """

    @staticmethod
    def set_repository(version="2024"):
        from pathlib import Path
        import subprocess
        import os
        accmodels = Path("acc-models-lhc")
        if accmodels.exists():
            if not (accmodels/"lhc.seq").exists():
                raise FileNotFoundError("acc-models-lhc/lhc.seq not found")
            else:
                if  (accmodels/".git").exists():
                    subprocess.run(["git", "switch", version], cwd=accmodels)
        elif (lcl:=(Path.home()/"local"/"acc-models-lhc"/version)).exists():
            accmodels.symlink_to(lcl)
        else:
            subprocess.run(["git", "clone", "https://gitlab.cern.ch/acc-models/lhc.git", "acc-models-lhc"])


    irs = [LHCIR1, LHCIR2, LHCIR3, LHCIR4, LHCIR5, LHCIR6, LHCIR7, LHCIR8]
    arcs = ["a12", "a23", "a34", "a45", "a56", "a67", "a78", "a81"]

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
        self._irs = irs
        self._arcs = arcs
        if params is None:
            params = {}
        if knobs is None:
            knobs = {}
        self.params = params
        self.knobs = knobs
        self.model = model

    @classmethod
    def from_madxfile(
        cls, filename, name="lhcoptics", sliced=False, model=None
    ):
        madx = Madx()
        madx.call(filename)
        return cls.from_madx(madx, name=name, sliced=sliced, model=model)

    @classmethod
    def from_madx(cls, madx, name="lhcoptics", sliced=False, model=None):
        madmodel = LHCMadModel(madx)
        knobs = madmodel.make_and_set0_knobs(cls.knobs)
        irs = [ir.from_madx(madx) for ir in cls.irs]
        arcs = [LHCArc.from_madx(madx, arc) for arc in cls.arcs]
        for k, knob in knobs.items():
            madx.globals[k] = knob.value
        self = cls(name, irs, arcs, knobs=knobs)
        if model=='xsuite':
            model = LHCXsuiteModel.from_madx(madx,sliced=sliced)
            self.model=model
        elif model=='madx':
            self.model = madmodel
        return self


    @classmethod
    def from_dict(cls, data):
        irs = [
            globals()[f"LHCIR{n+1}"].from_dict(d)
            for n, d in enumerate(data["irs"])
        ]
        arcs = [LHCArc.from_dict(d) for d in data["arcs"]]
        return cls(
            name=data["name"],
            irs=irs,
            arcs=arcs,
            params=data["params"],
            knobs={k: Knob.from_dict(d) for k, d in data["knobs"].items()},
        )

    @classmethod
    def from_json(cls, filename):
        with open(filename) as f:
            data = json.load(f)
            return cls.from_dict(data)

    def __repr__(self) -> str:
        return f"<LHCOptics {self.name!r}>"

    def to_dict(self):
        return {
            "name": self.name,
            "irs": [ir.to_dict() for ir in self._irs],
            "arcs": [arc.to_dict() for arc in self._arcs],
            "params": self.params,
            "knobs": {n: k.to_dict() for n, k in self.knobs.items()},
        }

    def copy(self):
        return self.__class__.from_dict(self.to_dict())

    def to_json(self, filename):
        with open(filename, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def update_model(self):
        for ss in self._irs + self._arcs:
            ss.update_model()
        return self

    def update_from_model(self):
        for ss in self._irs + self._arcs:
            ss.update_from_model()
        return self

    def set_xsuite_model(self, model):
        if os.path.exists(model):
            model = LHCXsuiteModel.from_json(model)
        self.model = model
        self.update_model()
        return self

    def set_madx_model(self, model):
        if os.path.exists(model):
            model = LHCMadModel.from_madxfile(model)
        self.model = model
        self.update_model()
        return self

    def get_params(self):
        tw1 = self.model.b1.twiss(compute_chromatic_properties=True)
        tw2 = self.model.b2.twiss(compute_chromatic_properties=True)
        params = {
            "qxb1": tw1.mux[-1],
            "qyb1": tw1.muy[-1],
            "qxb2": tw2.mux[-1],
            "qyb2": tw2.muy[-1],
            "qpxb1": tw1.mux[-1],
            "qpyb1": tw1.muy[-1],
            "qpxb2": tw2.mux[-1],
            "qpyb2": tw2.muy[-1],
        }
        for ss in self._irs + self._arcs:
            params.update(ss.get_params_from_twiss(tw1, tw2))
        return params
