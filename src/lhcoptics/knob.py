import re

import numpy as np
import xtrack as xt

from .utils import print_diff_dict_float


class Knob:

    def __init__(self, name, value=0, weights=None, parent=None):
        self.name = name
        self.value = value
        self.weights = weights
        self.parent = parent

    @classmethod
    def from_dict(cls, data):
        if "class" in data:
            return globals()[data["class"]].from_dict(data)
        out = cls(data["name"], data["value"], data["weights"])
        # specializations
        out = IPKnob.specialize(out)
        out = DispKnob.specialize(out)
        out = TuneKnob.specialize(out)
        out = ChromaKnob.specialize(out)
        out = CouplingKnob.specialize(out)
        # octupole knobs
        # dpp knmob
        # disp knob
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

    def from_madx(self, madx, redefine_weights=False):
        raise NotImplementedError

    def check(self, threshold=1e-10, test_value=1):
        print(f"Warning knob {self.name!r} check not implemented")

    def copy(self):
        return Knob(self.name, self.value, self.weights.copy())

    def to_dict(self):
        return {
            "name": self.name,
            "value": self.value,
            "weights": self.weights,
        }

    def diff(self, other):
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

    def __repr__(self):
        return f"<Knob {self.name!r} = {self.value}>"


class IPKnob(Knob):
    _zero_init = [xt.TwissInit(), xt.TwissInit()]
    reorb = re.compile("on_([A-z]+)([0-9])_?([hv])?(b[12])?")

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
        match_value=1,
        beams=["b1", "b2"],
        kind=None,
    ):
        super().__init__(name, value, weights, parent)
        self.const = const
        self.ip = ip
        self.xy = xy
        self.hv = "h" if xy == "x" else "v"
        self.ipname = f"ip{ip}"
        self.tols = {"": 1e-8, "p": 1e-10}
        self.step = 1e-9
        self.specs = specs
        self.beams = beams
        self.match_value = match_value
        self.kind = kind

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
            match_value = 1
            if kind in ["x", "sep", "oh", "ov", "a", "o"]:
                beams = ["b1", "b2"]
                if kind == "x":
                    dxy = 0
                    dpxy = 1e-6
                    ss = -1
                    match_value = 170
                elif kind == "sep":
                    if (irn == "5" and hv == "h") or (
                        irn == "1" and hv == "v"
                    ):
                        dxy = -1e-3
                    else:
                        dxy = 1e-3
                    dpxy = 0
                    ss = -1
                elif kind == "a":
                    dxy = 0
                    dpxy = 1e-6
                    ss = 1
                    match_value = 30
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
                match_value=match_value,
            )

    def check(self, threshold=1e-9, test_value=1):
        print(f"Checking knob {self.name!r}")
        model = self.parent.model
        old_value = model[self.name]
        model[self.name] = test_value
        msg = []
        cols = ["x", "y", "px", "py"]
        ipname = f"ip{self.ip}"
        if len(self.beams) == 2:
            tw1, tw2 = self.parent.twiss(strengths=False)
            expected = {
                (col, f"ip{ip}"): (0, 0) for ip in range(1, 9) for col in cols
            }
            if self.kind in ["x", "a"]:
                pp = f"p{self.xy}"
            else:
                pp = self.xy
            expected[pp, ipname] = (
                self.specs[pp + "b1"] * test_value,
                self.specs[pp + "b2"] * test_value,
            )
            for (col, ips), (v1, v2) in expected.items():
                for tw, vv in zip([tw1, tw2], [v1, v2]):
                    if abs(tw[col, ips] - vv) > threshold:
                        msg.append(
                            f"Error: {col} at {ips} = {tw[col,ips]:23.15g} != {vv:23.15g}"
                        )
        else:
            tw = self.parent.twiss(strengths=False, beam=int(self.beams[0][1]))
            expected = {
                (col, f"ip{ip}"): 0 for ip in range(1, 9) for col in cols
            }
            expected[self.xy, ipname] = (
                self.specs[self.xy + self.beams[0]] * test_value
            )
            for (col, ips), vv in expected.items():
                if abs(tw[col, ips] - vv) > threshold:
                    msg.append(
                        f"Error: {col} at {ips} = {tw[col,ips]:23.15g} != {vv:23.15g}"
                    )

        model[self.name] = old_value
        if len(msg) > 0:
            print("\n".join(msg))
            raise ValueError(f"Knob {self.name!r} failed the check")

    def copy(self):
        return IPKnob(
            name=self.name,
            value=self.value,
            weights=self.weights.copy(),
            parent=self.parent,
            const=self.const,
            ip=self.ip,
            xy=self.xy,
            specs=self.specs,
            match_value=self.match_value,
            beams=self.beams,
            kind=self.kind,
        )

    def get_mcbx_preset(self):
        left = [k for k in self.weights if re.match(f"acbx{self.hv}\\d.l", k)]
        right = [k for k in self.weights if re.match(f"acbx{self.hv}\\d.r", k)]
        vleft = sum([self.weights[k] for k in left])
        vright = sum([self.weights[k] for k in right])
        return vleft, vright

    def match(self):
        model = self.parent.model
        """
        In general the problem is to find

        find dv/dk such that

           t(v0+k0*dv/dk) + dt/dk*(k1-k0) == t(v0+k1*dv/dk)

        Here we solve

           t(v0) + dt/dk*k = t(v0+k*dv/dk)

        needs to get limits in vary (limits will be check during the dry-run with currents)
        """

        xt = model._xt
        ir = getattr(self.parent, f"ir{self.ip}")
        targets = [
            xt.Target(
                tt + self.xy,
                value=self.specs[tt + self.xy + bb] * self.match_value,
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
            if "b1" in wn and self.weights[wn.split("_from_")[0]] != 0
        ]
        varyb2 = [
            xt.Vary(wn, step=self.step)
            for wn in self.get_weight_knob_names()
            if "b2" in wn and self.weights[wn.split("_from_")[0]] != 0
        ]
        varycmn = [
            xt.Vary(wn, step=self.step)
            for wn in self.get_weight_knob_names()
            if wn.startswith(f"acbx{self.hv}")
            and self.weights[wn.split("_from_")[0]] != 0
        ]
        if len(self.beams) == 2:
            vary = varycmn + varyb1 + varyb2
        elif self.beams[0] == "b1":
            vary = varyb1
        elif self.beams[0] == "b2":
            vary = varyb2

        # assumes using multilines anyway
        # TODO use only the relevant lines
        start = ir.startb12
        end = ir.endb12
        init = self._zero_init

        mtc = model.match(
            solve=False,
            start=start,
            end=end,
            init=init,  # Zero orbit
            vary=vary,
            targets=targets,
            strengths=False,
            compute_chromatic_properties=False,
        )

        mtc.disable(vary_name=self.const)
        knob_start = model[self.name]
        try:
            model[self.name] = 0
            # get present target values
            mtc._err(None, check_limits=False)
            # add offsets
            for val, tt in zip(mtc._err.last_res_values, mtc.targets):
                tt.value += val
            # update definitions, potentially mismatched
            model.update_knob(self)
            # add offset in the knobs
            model[self.name] = self.match_value
            mtc.target_status()
            mtc.solve()
        except Exception as ex:
            mtc.vary_status()
            print(f"Failed to match {self.name}")
            model.update_knob(self)
            raise (ex)
        model[self.name] = knob_start
        return mtc

    def plot(self, value=None):
        model = self.parent.model
        aux = self.value
        if value is None:
            value = self.value
        model[self.name] = value
        ir = getattr(self.parent, f"ir{self.ip}")
        if len(self.beams) == 2:
            ir.plot(yl="x y")
        else:
            ir.plot(beam=int(self.beams[0][1]), yl="x y")
        model[self.name] = aux

    def set_mcbx_preset(self, vleft, vright=None):
        if vright is None:
            if self.kind == "x":
                vright = -vleft
            else:
                vright = vleft
        left = [k for k in self.weights if re.match(f"acbx{self.hv}\\d.l", k)]
        right = [k for k in self.weights if re.match(f"acbx{self.hv}\\d.r", k)]
        for k in left:
            self.weights[k] = vleft / len(left)
        for k in right:
            self.weights[k] = vright / len(right)

    def __repr__(self):
        return f"<IPKnob {self.name!r} = {self.value}>"


class TuneKnob(Knob):
    retune = re.compile(r"dq([xy])\.(b[12])_?([a-z]+)?")

    def __init__(
        self,
        name,
        value=0,
        weights=None,
        parent=None,
        xy="x",
        beam="b1",
        kind=None,
        match_value=0.01,
    ):
        super().__init__(name, value, weights, parent)
        self.xy = xy
        self.beam = beam
        self.kind = kind
        self.match_value = match_value

    @classmethod
    def specialize(cls, knob):
        """
        Specialize the knob to a specific type or return the original knob.
        """
        if isinstance(knob, cls):
            return knob
        mtc = cls.retune.match(knob.name)
        if mtc is None:
            return knob
        else:
            xy, beam, kind = mtc.groups()
            return cls(
                knob.name,
                value=knob.value,
                weights=knob.weights,
                xy=xy,
                beam=beam,
                kind=kind,
                match_value=0.01,
            )

    def check(self, threshold=1e-9, test_value=0.01):
        print(f"Checking knob {self.name!r}")
        model = self.parent.model
        old_value = model[self.name]
        model[self.name] = 0
        tw = self.parent.twiss(strengths=False, beam=int(self.beam[1]))
        zero_qxy = tw.qx, tw.qy
        model[self.name] = test_value
        tw = self.parent.twiss(strengths=False, beam=int(self.beam[1]))
        delta = tw.qx - zero_qxy[0], tw.qy - zero_qxy[1]
        if self.xy == "x":
            expected = (test_value, 0)
        else:
            expected = (0, test_value)
        msg = []
        if abs(delta[0] - expected[0]) > threshold:
            msg.append(
                f"Error: qx = {tw.qx:23.15g}, delta = {delta[0]:23.15g} != {expected[0]:23.15g}"
            )
        if abs(delta[1] - expected[1]) > threshold:
            msg.append(
                f"Error: qy = {tw.qy:23.15g}, delta = {delta[0]:23.15g} != {expected[1]:23.15g}"
            )
        model[self.name] = old_value
        if len(msg) > 0:
            print("\n".join(msg))
            raise ValueError(f"Knob {self.name!r} failed the check")
        if len(msg) > 0:
            print("\n".join(msg))
            raise ValueError(f"Knob {self.name!r} failed the check")

    def copy(self):
        return TuneKnob(
            name=self.name,
            value=self.value,
            weights=self.weights.copy(),
            parent=self.parent,
            xy=self.xy,
            beam=self.beam,
            kind=self.kind,
            match_value=self.match_value,
        )

    def match(self, solve=True):
        model = self.parent.model
        xt = model._xt
        knob_start = model[self.name]
        line = getattr(model, self.beam)
        vary = [
            xt.Vary(wn, step=1e-9)
            for wn in self.get_weight_knob_names()
            if self.weights[wn.split("_from_")[0]] != 0
        ]
        if self.xy == "x":
            dq = {"x": self.match_value, "y": 0}
        else:
            dq = {"x": 0, "y": self.match_value}

        # find baseline
        model[self.name] = 0
        tw = line.twiss(strengths=False, compute_chromatic_properties=False)
        q0 = {"x": tw["qx"], "y": tw["qy"]}
        targets = [
            tw.target(
                f"q{xy}",
                value=q0[xy] + dq[xy],
                tol=1e-9,
            )
            for xy in "xy"
        ]
        model[self.name] = self.match_value
        mtc = line.match(
            solve=False,
            vary=vary,
            targets=targets,
            strengths=False,
            compute_chromatic_properties=False,
        )
        mtc.target_status()
        mtc.vary_status()
        if solve:
            mtc.solve()
        model[self.name] = knob_start
        model.get_knob(self).diff(self)
        return mtc

    def __repr__(self):
        return f"TuneKnob({self.name!r}, {self.value})"


class ChromaKnob(Knob):
    retune = re.compile(r"dqp([xy])\.(b[12])_?([a-z]+)?")

    def __init__(
        self,
        name,
        value=0,
        weights=None,
        parent=None,
        xy="x",
        beam="b1",
        kind=None,
        match_value=10,
    ):
        super().__init__(name, value, weights, parent)
        self.xy = xy
        self.beam = beam
        self.kind = kind
        self.match_value = match_value

    def check(self, threshold=1e-4, test_value=1):
        print(f"Checking knob {self.name!r}")
        model = self.parent.model
        old_value = model[self.name]
        model[self.name] = 0
        tw = self.parent.twiss(
            strengths=False, chrom=True, beam=int(self.beam[1])
        )
        zero_dqxy = tw.dqx, tw.dqy
        model[self.name] = test_value
        tw = self.parent.twiss(
            strengths=False, chrom=True, beam=int(self.beam[1])
        )
        delta = tw.dqx - zero_dqxy[0], tw.dqy - zero_dqxy[1]
        if self.xy == "x":
            expected = (test_value, 0)
        else:
            expected = (0, test_value)
        msg = []
        if abs(delta[0] - expected[0]) > threshold:
            msg.append(
                f"Error: dqx = {tw.dqx:23.15g}, delta = {delta[0]:23.15g} != {expected[0]:23.15g}"
            )
        if abs(delta[1] - expected[1]) > threshold:
            msg.append(
                f"Error: dqy = {tw.dqy:23.15g}, delta = {delta[0]:23.15g} != {expected[1]:23.15g}"
            )
        model[self.name] = old_value
        if len(msg) > 0:
            print("\n".join(msg))
            raise ValueError(f"Knob {self.name!r} failed the check")
        if len(msg) > 0:
            print("\n".join(msg))
            raise ValueError(f"Knob {self.name!r} failed the check")

    def copy(self):
        return ChromaKnob(
            name=self.name,
            value=self.value,
            weights=self.weights.copy(),
            parent=self.parent,
            xy=self.xy,
            beam=self.beam,
            kind=self.kind,
            match_value=self.match_value,
        )

    @classmethod
    def specialize(cls, knob):
        """
        Specialize the knob to a specific type or return the original knob.
        """
        if isinstance(knob, cls):
            return knob
        mtc = cls.retune.match(knob.name)
        if mtc is None:
            return knob
        else:
            xy, beam, kind = mtc.groups()
            return cls(
                knob.name,
                value=knob.value,
                weights=knob.weights,
                xy=xy,
                beam=beam,
                kind=kind,
            )

    def match(self):
        model = self.parent.model
        xt = model._xt
        knob_start = model[self.name]
        line = getattr(model, self.beam)
        for wn in self.get_weight_knob_names():
            if self.weights[wn.split("_from_")[0]] != 0:
                family = re.match(r"ks([fd]).*", wn).group(1)
                tmp = f"ks{family}_temp"
                model[tmp] = model[wn]
                model.vars[wn] = model.vars[tmp]
                print(f"Setting {wn} := {tmp};")

        vary = [xt.VaryList(["ksf_temp", "ksd_temp"], step=1e-9)]
        if self.xy == "x":
            dq = {"x": self.match_value, "y": 0}
        else:
            dq = {"x": 0, "y": self.match_value}

        # find baseline
        model[self.name] = 0
        tw = self.parent.twiss(int(self.beam[1]), chrom=True, strengths=False)
        q0 = {"x": tw["dqx"], "y": tw["dqy"]}
        targets = [
            tw.target(
                f"dq{xy}",
                value=q0[xy] + dq[xy],
                tol=1e-5,
            )
            for xy in "xy"
        ]
        model[self.name] = self.match_value
        mtc = line.match(
            solve=False,
            vary=vary,
            targets=targets,
            strengths=False,
            compute_chromatic_properties=True,
            n_steps_max=50,
        )
        mtc.target_status()
        mtc.vary_status()
        mtc.step(20)
        mtc.solve()
        model[self.name] = knob_start
        # reset weights
        for wn in self.get_weight_knob_names():
            if self.weights[wn.split("_from_")[0]] != 0:
                model.vars[wn] = model[wn]
        model.get_knob(self).diff(self)
        return mtc

    def __repr__(self):
        return f"ChromaKnob({self.name!r}, {self.value})"


class ActionCmin(xt.Action):
    # see Eq. 47 in https://cds.cern.ch/record/522049/files/lhc-project-report-501.pdf
    def __init__(self, line):
        self.line = line
        self.twiss = self.line.twiss(compute_chromatic_properties=False)
        self.betx = self.twiss.betx[:-1]
        self.bety = self.twiss.bety[:-1]
        self.mux = self.twiss.mux[:-1]
        self.muy = self.twiss.muy[:-1]
        self.reverse = self.line.twiss_default.get("reverse", False)

    def run(self):
        k1sl = self.line.attr["k1sl"]
        if self.reverse:
            k1sl = -k1sl[::-1]
        c_min = (
            1
            / (2 * np.pi)
            * np.sum(
                k1sl
                * np.sqrt(self.betx * self.bety)
                * np.exp(1j * 2 * np.pi * (self.mux - self.muy))
            )
        )
        return {"r": c_min.real, "i": c_min.imag}


class CouplingKnob(Knob):
    retune = re.compile(r"cm([ri])s\.(b[12])_?([a-z]+)?")

    def __init__(
        self,
        name,
        value=0,
        weights=None,
        parent=None,
        ri="r",
        beam="b1",
        kind=None,
        match_value=1e-4,
    ):
        super().__init__(name, value, weights, parent)
        self.ri = ri
        self.beam = beam
        self.kind = kind
        self.match_value = match_value
        assert self.ri in ["r", "i"]

    def copy(self):
        return CouplingKnob(
            name=self.name,
            value=self.value,
            weights=self.weights.copy(),
            parent=self.parent,
            ri=self.ri,
            beam=self.beam,
            kind=self.kind,
            match_value=self.match_value,
        )

    def check(self, threshold=1e-3, test_value=0.0001):
        print(f"Checking knob {self.name!r}")
        model = self.parent.model
        old_value = model[self.name]
        model[self.name] = 0
        zero_cmin = self.parent.get_cmin(beam=int(self.beam[1]))
        model[self.name] = test_value
        new_cmin = self.parent.get_cmin(beam=int(self.beam[1]))
        delta = zero_cmin[0] - new_cmin[0], zero_cmin[1] - new_cmin[1]
        if self.ri == "r":
            expected = (test_value, 0)
        else:
            expected = (0, test_value)
        msg = []
        if abs(delta[0] - expected[0]) > threshold:
            msg.append(
                f"Error: cmin_re = {new_cmin[0]:23.15g}, delta = {delta[0]:23.15g} != {expected[0]:23.15g}"
            )
        if abs(delta[1] - expected[1]) > threshold:
            msg.append(
                f"Error: cmin_im = {new_cmin:23.15g}, delta = {delta[0]:23.15g} != {expected[1]:23.15g}"
            )
        model[self.name] = old_value
        if len(msg) > 0:
            print("\n".join(msg))
            raise ValueError(f"Knob {self.name!r} failed the check")
        if len(msg) > 0:
            print("\n".join(msg))
            raise ValueError(f"Knob {self.name!r} failed the check")

    @classmethod
    def specialize(cls, knob):
        """
        Specialize the knob to a specific type or return the original knob.
        """
        if isinstance(knob, cls):
            return knob
        mtc = cls.retune.match(knob.name)
        if mtc is None:
            return knob
        else:
            ri, beam, kind = mtc.groups()
            return cls(
                knob.name,
                value=knob.value,
                weights=knob.weights,
                ri=ri,
                beam=beam,
                kind=kind,
            )

    def match(self, limit=0.1, weights={}, reset=True):
        model = self.parent.model
        xt = model._xt

        line = getattr(model, self.beam)
        # line.cycle("ip7", inplace=True)

        kqsa = [
            f"{wm}_from_{self.name}"
            for wm, wv in self.weights.items()
            if re.match(r"kqs\.a", wm) and wv != 0
        ]
        kqslr = [
            f"{wm}_from_{self.name}"
            for wm, wv in self.weights.items()
            if re.match(r"kqs\.[lr]", wm) and wv != 0
        ]

        if self.ri == "r":
            dq = {"r": self.match_value, "i": 0}
        else:
            dq = {"r": 0, "i": self.match_value}

        if reset:
            for wn in kqsa + kqslr:
                model[wn] = 0

        act_cmin = ActionCmin(line)
        # find baseline
        knob_start = model[self.name]
        model[self.name] = 0
        q0 = {"r": act_cmin.run()["r"], "i": act_cmin.run()["i"]}
        targets = [
            act_cmin.target(ri, value=q0[ri] + dq[ri], tol=1e-8) for ri in "ri"
        ]
        model[self.name] = self.match_value
        mtc = line.match(
            solve=False,
            vary=[
                xt.VaryList(kqsa, step=5e-5, limits=(-limit, limit)),
                xt.VaryList(kqslr, step=5e-5, limits=(-limit, limit)),
            ],
            targets=targets,
            check_limits=False,
            compute_chromatic_properties=False,
        )
        for kk, val in weights.items():
            for vv in mtc.vary:
                if vv.name.startswith(kk):
                    vv.weight = val

        mtc.target_status()
        mtc.vary_status()
        mtc.solve()
        mtc.vary_status()
        model[self.name] = knob_start
        print("Knob new vs old")
        model.get_knob(self).diff(self)
        # line.cycle("ip1", inplace=True)
        return mtc

    def __repr__(self):
        return f"CouplingKnob({self.name!r}, {self.value})"


class DispKnob(Knob):
    reorb = re.compile("on_([A-z]+)([0-9])_?([hv])?")
    match_value = {"xx": 170, "ssep": 1}
    hv = {"h": "x", "v": "y"}

    def __init__(
        self,
        name,
        value=0,
        weights=None,
        parent=None,
        ip=None,
        xy=None,
        specs=None,
        match_value=1,
        beams=["b1", "b2"],
        kind=None,
    ):
        super().__init__(name, value, weights, parent)
        self.ip = ip
        self.xy = xy
        self.hv = "h" if xy == "x" else "v"
        self.ipname = f"ip{ip}"
        self.tols = {"": 1e-8, "p": 1e-10}
        self.step = 1e-9
        self.specs = specs
        self.beams = beams
        self.match_value = match_value
        self.kind = kind

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
            kind, irn, hv = mtc.groups()
            if kind not in cls.match_value:
                return knob
            else:
                xy = cls.hv[hv]
                return cls(
                    knob.name,
                    value=knob.value,
                    weights=knob.weights,
                    parent=knob.parent,
                    ip=irn,
                    xy=xy,
                    beams=["b1", "b2"],
                    kind=kind,
                    match_value=cls.match_value[kind],
                )

    def copy(self):
        return IPKnob(
            name=self.name,
            value=self.value,
            weights=self.weights.copy(),
            parent=self.parent,
            ip=self.ip,
            xy=self.xy,
            specs=self.specs,
            match_value=self.match_value,
            beams=self.beams,
            kind=self.kind,
        )

    def match(self, beam):
        model = self.parent.model
        """
        In general the problem is to find

        find dv/dk such that

           t(v0+k0*dv/dk) + dt/dk*(k1-k0) == t(v0+k1*dv/dk)

        Here we solve

           t(v0) + dt/dk*k = t(v0+k*dv/dk)

        needs to get limits in vary (limits will be check during the dry-run with currents)
        """

        xt = model._xt
        if self.ip == "1":
            e_arc_right = f"e.ds.r2.b{beam}"
        elif self.ip == "5":
            e_arc_right = f"e.ds.r1.b{beam}"
        targets = [
            xt.Target(
                tt + self.xy,
                value=0,
                at=at,
                tol=self.tols[tt],
            )
            for tt in ("", "p")
            for at in (self.ipname, e_arc_right)
        ]
        targets += [
            xt.Target(
                f"d{tt}{self.xy}", value=0, at=self.ipname, tol=self.tols[tt]
            )
            for tt in ("", "p")
        ]
        vary = [
            xt.Vary(wn, step=self.step)
            for wn in self.get_weight_knob_names(beam)
            if "b1" in wn and self.weights[wn.split("_from_")[0]] != 0
        ]

        line = getattr(model, "b" + beam)

        mtc = line.match(
            solve=False,
            vary=vary,
            targets=targets,
            strengths=False,
            compute_chromatic_properties=False,
        )

        knob_start = model[self.name]
        try:
            model[self.name] = 0
            # get present target values
            mtc._err(None, check_limits=False)
            # add offsets
            for val, tt in zip(mtc._err.last_res_values, mtc.targets):
                tt.value += val
            # update definitions, potentially mismatched
            model.update_knob(self)
            # add offset in the knobs
            model[self.name] = self.match_value
            mtc.target_status()
            mtc.solve()
            mtc.vary_status()
        except Exception as ex:
            print(f"Failed to match {self.name}")
            model.update_knob(self)
            raise (ex)
        model[self.name] = knob_start
        return mtc

    def plot(self, value=None):
        model = self.parent.model
        aux = self.value
        if value is None:
            value = self.value
        model[self.name] = value
        if self.ip == "1":
            start,end="ip8","ip2"
        elif self.ip == "5":
            start,end="ip2","ip8"
        model.b1.twiss(start=start).rows[:end].plot(yl="x y")
        model.b2.twiss(start=start).rows[:end].plot(yl="x y")
        model[self.name] = aux

    def set_mcbx_preset(self, vleft, vright=None):
        if vright is None:
            if self.kind == "x":
                vright = -vleft
            else:
                vright = vleft
        left = [k for k in self.weights if re.match(f"acbx{self.hv}\\d.l", k)]
        right = [k for k in self.weights if re.match(f"acbx{self.hv}\\d.r", k)]
        for k in left:
            self.weights[k] = vleft / len(left)
        for k in right:
            self.weights[k] = vright / len(right)

    def __repr__(self):
        return f"<IPKnob {self.name!r} = {self.value}>"
