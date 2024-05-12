from .utils import print_diff_dict_float
import re

import xtrack as xt


class Knob:
    @classmethod
    def from_dict(cls, data):
        if "class" in data:
            return globals()[data["class"]].from_dict(data)
        out = cls(data["name"], data["value"], data["weights"])
        out = IPKnob.specialize(out)  # try to specialize from name
        return out

    @classmethod
    def from_src(cls, src):
        if (
            hasattr(src, "name")
            and hasattr(src, "value")
            and hasattr(src, "weights")
        ):
            return cls.from_dict(src.__dict__)
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
        return [f"{key}_from_{self.name}" for key in self.weights.keys()]

    def specialize(self, knob):
        """Specialize the knob to a specific type."""
        knob = IPKnob.specialize(self)
        return knob


class IPKnob(Knob):
    _zero_init = [xt.TwissInit(), xt.TwissInit()]
    reorb = re.compile("on_([A-z]+)([0-9])_?([hv])?(b[12])?")

    @classmethod
    def specialize(cls, knob):
        """
        Specialize the knob to a specific type or return the original knob.
        """
        if isinstance(knob, cls):
            return knob
        mtc = cls.reorb.match(knob.name)
        if mtc is None:
            return knob
        else:
            kind, irn, hv, beam = mtc.groups()
            if kind in ["xx", "ssep"]:
                return knob
            if hv is None:
                if kind.startswith("x"):
                    hv = "h"
                elif kind.startswith("y"):
                    hv = "v"
                elif kind == "oh":
                    hv = "h"
                elif kind == "ov":
                    hv = "v"
                elif kind == "a" and irn in "12":
                    hv = "h"
                elif kind == "a" and irn in "58":
                    hv = "v"
                elif kind == "o" and irn in "12":
                    hv = "v"
                elif kind == "o" and irn in "58":
                    hv = "h"
                else:
                    raise ValueError(
                        f"Cannot determine plane for {knob.name!r}"
                    )
            xy = {"h": "x", "v": "y"}[hv]
            if kind in ["x", "sep", "oh", "ov", "a", "o"]:
                beams = ["b1", "b2"]
                if kind == "x":
                    dxy = 0
                    dpxy = 1e-6
                    ss = -1
                elif kind == "sep":
                    dxy = 1e-3
                    dpxy = 0
                    ss = -1
                elif kind == "a":
                    dxy = 0
                    dpxy = 1e-6
                    ss = 1
                elif kind.startswith("o"):
                    dxy = 1e-3
                    dpxy = 0
                    ss = 1
                specs = {
                    f"{xy}b1": dxy,
                    f"{xy}b2": ss * dxy,
                    f"p{xy}b1": dpxy,
                    f"p{xy}b2": ss * dpxy,
                }
                const = [
                    k for k in knob.weights.keys() if re.match(f"acbx{hv}", k)
                ]
            elif kind == "xip" or kind == "yip":
                specs = {kind[0] + beam: 1e-3, "p" + kind[0] + beam: 0.0}
                const = []
                beams = [beam]
            else:
                print(f"Warning: {cls} cannot specialize {knob.name!r}")
                return knob
            return cls(
                knob.name,
                value=knob.value,
                weights=knob.weights,
                parent=knob.parent,
                const=const,
                ip=irn,
                xy=xy,
                specs=specs,
                beams=beams,
                kind=kind,
            )

    def __init__(
        self,
        name,
        value=0,
        weights=None,
        parent=None,
        const=None,
        ip=None,
        xy=None,
        specs=None,
        max_value=1,
        beams=["b1", "b2"],
        kind=None,
    ):
        super().__init__(name, value, weights, parent)
        self.const = const
        self.ip = ip
        self.xy = xy
        self.hv = "h" if xy == "x" else "v"
        self.ipname = f"ip{ip}"
        self.tols = {"": 1e-9, "p": 1e-11}
        self.step = 1e-9
        self.specs = specs
        self.beams = beams
        self.max_value = max_value
        self.kind = kind

    def match(self):
        # for beam in self.beams:
        #    getattr(self.parent.model,beam).build_tracker()
        knob_start = self.parent.model[self.name]
        xt = self.parent.model._xt
        ir = getattr(self.parent, f"ir{self.ip}")
        targets = [
            xt.Target(
                tt + self.xy,
                value=self.specs[tt + self.xy + bb],
                line=bb,
                at=self.ipname,
                tol=self.tols[tt],
            )
            for tt in ("", "p")
            for bb in self.beams
        ]
        targets += [
            xt.Target(
                tt + self.xy, value=0, line=bb, at=xt.END, tol=self.tols[tt]
            )
            for tt in ("", "p")
            for bb in self.beams
        ]
        varyb1 = [
            xt.Vary(wn, step=self.step)
            for wn in self.get_weight_knob_names()
            if "b1" in wn
        ]
        varyb2 = [
            xt.Vary(wn, step=self.step)
            for wn in self.get_weight_knob_names()
            if "b2" in wn
        ]
        varycmn = [
            xt.Vary(wn, step=self.step)
            for wn in self.get_weight_knob_names()
            if wn.startswith(f"acbx{self.hv}")
        ]
        if len(self.beams) == 2:
            start = ir.startb12
            end = ir.endb12
            vary = varycmn + varyb1 + varyb2
        elif self.beams[0] == "b1":
            start = ir.startb1
            end = ir.endb1
            vary = varyb1
        elif self.beams[0] == "b2":
            start = ir.startb2
            end = ir.endb2
            vary = varyb2

        mtc = self.parent.model.match(
            solve=False,
            start=start,
            end=end,
            init=self._zero_init,  # Zero orbit
            vary=vary,
            targets=targets,
        )

        mtc.disable(vary_name=self.const)
        # get present target values
        mtc._err(None, check_limits=False)
        # add offsets
        for val, tt in zip(mtc._err.last_res_values, mtc.targets):
            tt.value += val
        # update definition, potentially mismatched
        self.parent.model.update_knob(self)
        # add offset in the knobs
        self.parent.model[self.name] += self.max_value
        mtc.target_status()
        try:
            mtc.solve()
            mtc.vary_status()
            self.parent.model[self.name] = knob_start
        except:
            print(f"Failed to match {self.name}")
        return mtc

    def get_mcbx_preset(self):
        left = [k for k in self.weights if re.match(f"acbx{self.hv}\d.l", k)]
        right = [k for k in self.weights if re.match(f"acbx{self.hv}\d.r", k)]
        vleft = sum([self.weights[k] for k in left])
        vright = sum([self.weights[k] for k in right])
        return vleft, vright

    def set_mcbx_preset(self, vleft, vright=None):
        if vright is None:
            if self.kind == "x":
                vright = -vleft
            else:
                vright = vleft
        left = [k for k in self.weights if re.match(f"acbx{self.hv}\d.l", k)]
        right = [k for k in self.weights if re.match(f"acbx{self.hv}\d.r", k)]
        for k in left:
            self.weights[k] = vleft / len(left)
        for k in right:
            self.weights[k] = vright / len(right)

    def plot(self, value=1):
        aux = self.value
        self.parent.model[self.name] = value
        ir = getattr(self.parent, f"ir{self.ip}")
        if len(self.beams) == 2:
            ir.plot(yl="x y")
        else:
            ir.plot(beam=int(self.beams[0][1]), yl="x y")
        self.parent.model[self.name] = aux

    def __repr__(self):
        return f"IPKnob({self.name!r}, {self.value})"
