from .utils import print_diff_dict_float

import xtrack as xt

class Knob:
    @classmethod
    def from_dict(cls, data):
        return cls(data["name"], data["value"], data["weights"])

    @classmethod
    def from_src(cls, src):
        if (
            hasattr(src, "name")
            and hasattr(src, "value")
            and hasattr(src, "weights")
        ):
            return cls(src.name, src.value, src.weights)
        else:
            return cls.from_dict(src)

    def __init__(self, name, value=0, weights=None, parent=None):
        self.name = name
        self.value = value
        self.weights = weights
        self.parent = parent

    def from_madx(self, madx, redefine_weights=False):
        if weights is None:
            weights = {}

    def __repr__(self):
        return f"Knob({self.name!r}, {self.value})"

    def copy(self):
        return Knob(self.name, self.value, self.weights.copy())

    def to_dict(self):
        return {
            "name": self.name,
            "value": self.value,
            "weights": self.weights,
        }

    def print_diff(self, other):
        if self.value != other.value:
            print(f"{self.name} {self.value} != {other.value}")
        print_diff_dict_float(self.weights, other.weights)

    def print_update_diff(self, other):
        if self.value != other.value:
            print(f"Knob {self.name}  frpm {self.value} to  {other.value}")
        for key, value in self.weights.items():
            if key in other.weights and value != other.weights[key]:
                print(
                    f"Knob {self.name}: {key} from {value} to {other.weights[key]}"
                )

    def get_weight_knob_names(self):
        return [f"{key}_from_{key}" for key in self.weights.keys()]

class IPKnob(Knob):
    _zero_init=[xt.TwissInit(), xt.TwissInit()]

    def __init__(self, name, value=0, weights=None, parent=None, const=None,ip=None, plane=None, specs=None, max_value=1):
        super().__init__(name, value, weights, parent)
        self.const = const
        self.ip = ip
        self.plane = plane
        self.ipname = f"ip{ip}"
        self.tols= {"":1e-9, "p":1e-11}
        self.step= 1e-9
        self.specs=specs
        self.max_value=max_value

    def match(self):
        self.parent.model.update_knob(self)
        xt= self.parent.model._xt
        targets = [ xt.Target(self.plane,  value=self.specs[tt+self.plane+bb], line=bb, at=self.ipname, tol=self.xytol) for tt in ('','p') for bb in ('b1','b2') ]
        tagets+=[xt.Target(tt, value=0, line=bb,at=xt.END, tol=self.tols[tt]) for tt in ('','p') for bb in ('b1','b2') ]
        vary= [xt.Vary(
            self.get_weight_knob_names(),
            step=self.step()) for wv,wv in self.get_weight_knob_names()
        ]

        mtc=self.parent.model.match_knob(
            knob_name=self.name,
            knob_value_start=0.0, # before knobs are applied
            knob_value_end=self.max_value,  # after knobs are applied
            run=False,
            start=self.startb12,
            end=self.endb12,
            init=self._zero_init,  # Zero orbit
            vary=vary,
            targets=targets,
        )

        mtc.disable(vary_name=self.const)
        return mtc