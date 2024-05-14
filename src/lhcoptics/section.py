import json
import re
from pathlib import Path

from .knob import Knob
from .model_madx import LHCMadxModel
from .utils import (deliver_list_str, print_diff_dict_float,
                    print_diff_dict_objs)

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
        self._model = None

    @property
    def model(self):
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
        model = LHCMadxModel.from_madxfile(filename)
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

    def to_madx(self, output=None, knobs=True):
        out = []
        out.append(f"! {self.name.upper()}\n")
        if len(self.strengths) > 0:
            out.append(f"! Strengths of {self.name.upper()}")
            for k, v in self.strengths.items():
                out.append(f"{k:30} = {v:19.16f};")
            out.append("")
        if len(self.params) > 0:
            out.append(f"! Parameters of {self.name.upper()}")
            for k, v in self.params.items():
                out.append(f"{k:30} = {v:19.16f};")
            out.append("")
        if knobs and len(self.knobs) > 0:
            out.append(f"! Knobs of {self.name.upper()}")
            for expr in LHCMadxModel.knobs_to_expr(self.knobs, self.strengths):
                out.append(expr)
            out.append("")
        return deliver_list_str(out, output)

    def copy(self):
        return self.__class__(
            name=self.name,
            start=self.start,
            end=self.end,
            strengths=self.strengths.copy(),
            params=self.params.copy(),
            knobs={k: knob.copy() for k, knob in self.knobs.items()},
        )

    def set_params(self):
        """
        Copy all parameters from get_params() to self.params
        """
        self.params.update(self.get_params())
        return self

    def update_model(self, src=None, verbose=False,knobs_off=False):
        """Update the model with the local strengths, knobs
        If a src is provided, it will be used to update the local values, else self will be used.
        If src is a dict containing strengths, knobs or a LHCSection, they will be used to update the model.
        if src is dict of values, they will be used to update the model variables.
        """
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
        self.model.update_vars(strength, verbose=verbose)
        if hasattr(src, "knobs"):
            knobs = src.knobs
        elif "knobs" in src:
            knobs = src["knobs"]
        else:
            knobs = {}
        self.model.update_knobs(knobs, verbose=verbose,knobs_off=knobs_off)
        return self

    def update_strengths(self, src=None, verbose=False):
        """
        Update existing stregnths from self. model or src.strengths or src
        """
        if src is None:
            src = self.model
        elif hasattr(src, "strengths"):
            src = src.strengths
        for k in self.strengths:
            if k in src:
                if verbose and self.strengths[k] != src[k]:
                    print(
                        f"Updating {k!r} from {self.strengths[k]} to {src[k]}"
                    )
                self.strengths[k] = src[k]
        return self

    def update_knobs(self, src=None, verbose=False):
        """
        Update existing knobs from self. model or src.knobs or src
        """
        if src is None:
            src = self.model
        if hasattr(src, "get_knob"):
            src = {k: src.get_knob(knob) for k, knob in self.knobs.items()}
        elif hasattr(src, "knobs"):
            src = src.knobs
        for k in self.knobs:
            if k in src:
               if verbose:
                     self.knobs[k].print_update_diff(src[k])   
               self.knobs[k] = Knob.from_src(src[k])
               self.knobs[k].parent = self.parent
        return self

    def update_params(self, src=None, add=False, verbose=False):
        """
        Update existing params from self.model or src.params or src
        """
        if src is None:
            src = self.get_params()
        elif hasattr(src, "params"):
            src = src.params
        if add:
            for k in src:
                if k not in self.params:
                    if verbose:
                        print(f"Adding {k!r:18} = {src[k]}")
                    self.params[k] = src[k]
        else:
            for k in self.params:
                if k in src:
                    if verbose and self.params[k] != src[k]:
                        print(
                            f"Updating {k!r:15} from {self.params[k]:15.6g} to {src[k]:15.6g}"
                        )
                    self.params[k] = src[k]
        return self

    def update(
        self, src=None, verbose=False, knobs=True, strengths=True, params=True
    ):
        """
        Update existing strengths, knobs, params from self. model or src.params or src
        """
        if isinstance(src, str):
            src = self.__class__.from_json(src)
        if strengths:
            self.update_strengths(src, verbose=verbose)
        if knobs:
            self.update_knobs(src, verbose=verbose)
        if params:
            self.update_params(src, verbose=verbose)
        return self

    def twiss(self, beam=None, method=None, strengths=True):
        """Return twiss table from the model using specific methods."""
        if method is None:
            method = self.default_twiss_method
        return getattr(self, "twiss_" + method)(beam, strengths=strengths)

    def plot(self, beam=None, method="periodic", figlabel=None):
        if beam is None:
            return [self.plot(beam=1), self.plot(beam=2)]
        else:
            twiss = self.twiss(beam, method=method, strengths=True)
            if figlabel is None:
                figlabel = f"{self.name}b{beam}"
            return twiss.plot(figlabel=figlabel)

    def get_current(self, kname, p0c=7e12):
        if self.parent.circuit is None:
            raise ValueError("Circuit not set")
        return self.parent.circuit.get_current(kname, self[kname], p0c)

    def disable_bumps(self):
        pass

    def knobs_off(self):
        for k in self.knobs:
            self.parent.model[k] = 0

    def knobs_on(self):
        for k,knob in self.knobs.items():
            self.parent.model[k] = knob.value

    def diff(self, other):
        self.diff_strengths(other)
        self.diff_knobs(other)
        self.diff_params(other)

    def diff_strengths(self, other):
        print_diff_dict_float(self.strengths, other.strengths)

    def diff_params(self, other):
        print_diff_dict_float(self.params, other.params)

    def diff_knobs(self, other):
        print_diff_dict_objs(self.knobs, other.knobs)

    def __getitem__(self, key):
        if key in self.strengths:
            return self.strengths[key]
        elif key in self.params:
            return self.params[key]
        elif key in self.knobs:
            return self.knobs[key]
        else:
            raise KeyError(f"{key} not found in {self}")

    def __contains__(self, key):
        return key in self.strengths or key in self.params or key in self.knobs

    def __repr__(self):
        return f"<LHCSection {self.name} {self.start}/{self.end}>"
