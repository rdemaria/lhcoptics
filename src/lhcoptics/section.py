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

    def __repr__(self):
        return f"<LHCSection {self.name} {self.start}/{self.end}>"

    @property
    def model(self):
        return self.parent.model

    @classmethod
    def from_dict(cls, data):
        return cls(
            name=data["name"],
            start=data["start"],
            end=data["end"],
            strengths=data["strengths"],
            params=data["params"],
            knobs={k: Knob.from_dict(d) for k, d in data["knobs"].items()},
        )

    @classmethod
    def from_json(cls, filename):
        with open(filename) as f:
            data = json.load(f)
            return cls.from_dict(data)

    @classmethod
    def from_madxfile(cls, filename):
        model=LHCMadModel.from_madxfile(filename)
        return cls.from_madx(model.madx)

    def update_from_madxfile(self,filename):
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
        return {
            "name": self.name,
            "start": self.start,
            "end": self.end,
            "strengths": self.strengths,
            "params": self.params,
            "knobs": self.knobs,
        }

    
    def to_madx(self,out=str):
        out=[]
        out.append(f"! {self.name}\n")
        out.append(f"! Parameters\n")
        for k, v in self.params.items():
            out.append(f"{k:30} = {v:19.16f};")
        out.append(f"! Strengths\n")
        for k, v in self.strengths.items():
            out.append(f"{k:30} = {v:19.16f};")
        out.append(f"! Knobs\n")
        for expr in LHCMadModel.knobs_to_expr(self.knobs,self.strengths):
            out.append(expr)
        if out is str:
            return "\n".join(out)
        elif hasattr(self,'input'):
            for ll in out:
                if ll[0]!='!':
                    self.input(ll)
        elif hasattr(self,'writelines'):
            self.writelines(out)
        elif isinstance(out,str) or isinstance(out,Path):
            with open(out,'w') as f:
                f.writelines(out)    


    def copy(self):
        return self.__class__.from_dict(self.to_dict())

    def update_model(self, src=None):
        if self.model is None:
            raise ValueError("Model not set")
        self.model.update_vars(self.strengths)
        self.model.update_knobs(self.knobs)
        if src is not None:
            self.model.update_vars(src)
        return self

    def update_strengths(self,src=None):
        """
        Update existing stregnths from self. model or src.strengths or src
        """
        if src is None:
            src = self.model
        elif hasattr(src,"strengths"):
            src = src.strengths
        for k in self.strengths:
                self.strengths[k] = src[k]

    def update_knobs(self,src=None):
        """
        Update existing knobs from self. model or src.knobs or src
        """
        if src is None:
            src = self.model
            raise NotImplementedError
        elif hasattr(src,"knobs"):
            src=src.knobs
        for k in self.knobs:
            self.knobs[k] = src[k]

    def update_params(self,src=None):
        """
        Update existing params from self. model or src.params or src
        """
        if src is None:
            src = self.model
        elif hasattr(src,"params"):
            src = src.params
        for k in self.params:
            self.params[k] = src[k]

    def update(self, src=None):
        """
        Update existing strengths, knobs, params from self. model or src.params or src
        """
        self.update_strengths(src)
        self.update_knobs(src)
        self.update_params(src)
        return self

    def twiss(self, beam=None, method=None):
        """Return twiss table from the model using specific methods."""
        if method is None:
            method = self.default_twiss_method
        return getattr(self,"twiss_"+method)(beam)


    def plot(self, beam=None, method="periodic", figlabel=None):
        if beam is None:
            return [self.plot(beam=1), self.plot(beam=2)]
        else:
            twiss = self.twiss(beam,method=method)
            if figlabel is None:
                figlabel = f"{self.name}b{beam}"
            return twiss.plot(figlabel=figlabel)
        
    