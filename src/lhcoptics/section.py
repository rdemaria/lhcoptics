import re
import json
from pathlib import Path


from .model_madx import LHCMadModel
from .knob import Knob

_lb = [(l, b) for l in "lr" for b in "12"]
_ac = {
    0: [("h", "l", "1"), ("v", "l", "1"), ("h", "r", "2"), ("v", "r", "2")],
    1: [("h", "l", "2"), ("v", "l", "2"), ("h", "r", "1"), ("v", "r", "1")],
}


def lhcprev(n):
    return (n - 2) % 8 + 1


def lhcsucc(n):
    return n % 8 + 1


def sort_n(lst):
    out = []
    for s in lst:
        if m := re.match(r"[a-z]*([0-9][0-9]?)", s):
            out.append((int(m.group(1)), s))
        else:
            out.append((0, s))
    return [s for _, s in sorted(out)]


class LHCSection:
    """
    Model a section of the machine.
    It contains local strengths, local parameters and local knobs.

    It can twiss, plot and match the section using the model in the parent.

    It can be updated from the model and update the model with the local values.

    Needs:
       twiss_<method>() to return a twiss table
       get_params() method to return the parameters at the end of the section
       get_params_from_twiss() method to return the parameters from a twiss table

    """

    def __init__(
        self,
        name,
        start,
        end,
        strengths=None,
        params=None,
        knobs=None,
        parent=None,
        filename=None,
    ):
        self.name = name
        self.start = start
        self.end = end
        if strengths is None:
            strengths = {}
        if params is None:
            params = {}
        if knobs is None:
            knobs = {}
        self.strengths = strengths
        self.params = params
        self.knobs = knobs
        self.parent = parent
        self.filename = filename

    def __repr__(self):
        return f"<LHCSection {self.name} {self.start}/{self.end}>"

    @property
    def model(self):
        if hasattr(self, "model"):
            return self.model
        else:
            return self.parent.model

    @property
    def circuits(self):
        if hasattr(self, "circuits"):
            return self.circuits
        else:
            return self.parent.circuits

    @classmethod
    def from_dict(cls, data, filename=None):
        return cls(
            name=data["name"],
            start=data["start"],
            end=data["end"],
            strengths=data["strengths"],
            params=data["params"],
            knobs={k: Knob.from_dict(d) for k, d in data["knobs"].items()},
            filename=filename,
        )

    @classmethod
    def from_json(cls, filename):
        with open(filename) as f:
            data = json.load(f)
            return cls.from_dict(data, filename=filename)

    @classmethod
    def from_madxfile(cls, filename):
        model = LHCMadModel.from_madxfile(filename)
        return cls.from_madx(model.madx, filename=filename)

    def update_from_madxfile(self, filename):
        self.__class__.from_madxfile(filename)
        self.update(self.__class__.from_madxfile(filename))

    def to_dict(self):
        return {
            "name": self.name,
            "start": self.start,
            "end": self.end,
            "strengths": self.strengths,
            "params": self.params,
            "knobs": {n: k.to_dict() for n, k in self.knobs.items()},
        }

    def to_json(self, filename):
        with open(filename, "w") as f:
            json.dump(self.to_dict(), f)
        return self

    def to_madx(self, out=str):
        out = []
        out.append(f"! {self.name}\n")
        out.append(f"! Parameters\n")
        for k, v in self.params.items():
            out.append(f"{k:30} = {v:19.16f};")
        out.append(f"! Strengths\n")
        for k, v in self.strengths.items():
            out.append(f"{k:30} = {v:19.16f};")
        out.append(f"! Knobs\n")
        for expr in LHCMadModel.knobs_to_expr(self.knobs, self.strengths):
            out.append(expr)
        if out is str:
            return "\n".join(out)
        elif hasattr(self, "input"):
            for ll in out:
                if ll[0] != "!":
                    self.input(ll)
        elif hasattr(self, "writelines"):
            self.writelines(out)
        elif isinstance(out, str) or isinstance(out, Path):
            with open(out, "w") as f:
                f.writelines(out)

    def copy(self, src=None):
        other = self.__class__(
            strengths=self.strengths.copy(),
            params=self.params.copy(),
            knobs={k: v.copy() for k, v in self.knobs.items()},
        )
        if src is not None:
            other.update(src)
        return other

    def set_params(self):
        """
        Copy all parameters from get_params() to self.params
        """
        self.params.update(self.get_params())
        return self

    def update_model(self, src=None):
        if self.model is None:
            raise ValueError("Model not set")
        if src is None:
            src = self
        if hasattr(src, "strengths"):
            strength = src.strengths
        elif "strengths" in src:
            strength = src["strengths"]
        else:
            strength = src
        self.model.update_vars(strength)
        if hasattr(src, "knobs"):
            knobs = src.knobs
        elif "knobs" in src:
            knobs = src["knobs"]
        else:
            knobs = {}
        self.model.update_vars(strength)
        self.model.update_knobs(knobs)
        return self

    def update_strengths(self, src=None):
        """
        Update existing stregnths from self. model or src.strengths or src
        """
        if src is None:
            src = self.model
        elif hasattr(src, "strengths"):
            src = src.strengths
        for k in self.strengths:
            self.strengths[k] = src[k]
        return self

    def update_knobs(self, src=None):
        """
        Update existing knobs from self. model or src.knobs or src
        """
        if src is None:
            src = self.model
            raise NotImplementedError
        elif hasattr(src, "knobs"):
            src = src.knobs
        for k in self.knobs:
            self.knobs[k] = src[k]
        return self

    def update_params(self, src=None, add=False):
        """
        Update existing params from self. model or src.params or src
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

    def update(self, src=None):
        """
        Update existing strengths, knobs, params from self. model or src.params or src
        """
        if isinstance(src, str):
            src = self.__class__.from_json(src)
        self.update_strengths(src)
        self.update_knobs(src)
        self.update_params(src)
        return self

    def twiss(self, beam=None, method=None):
        """Return twiss table from the model using specific methods."""
        if method is None:
            method = self.default_twiss_method
        return getattr(self, "twiss_" + method)(beam)

    def plot(self, beam=None, method="periodic", figlabel=None):
        if beam is None:
            return [self.plot(beam=1), self.plot(beam=2)]
        else:
            twiss = self.twiss(beam, method=method)
            if figlabel is None:
                figlabel = f"{self.name}b{beam}"
            return twiss.plot(figlabel=figlabel)

    def __getitem__(self, key):
        if key in self.strengths:
            return self.strengths[key]
        elif key in self.params:
            return self.params[key]
        elif key in self.knobs:
            return self.knobs[key]
        else:
            raise KeyError(f"{key} not found in {self}")

    def get_current(self, kname, pc0=7e12):
        if self.parent.circuit is None:
            raise ValueError("Circuit not set")
        return self.parent.circuit.get_current(kname, self[kname], pc0)
