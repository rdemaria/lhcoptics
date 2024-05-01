from .model_madx import LHCMadModel
from .knob import Knob

import xdeps

def termlist(ex,lst=[]):
    if isinstance(ex,xdeps.refs.AddExpr):
        return lst+termlist(ex._lhs)+termlist(ex._rhs)
    if isinstance(ex,xdeps.refs.SubExpr):
        if isinstance(ex._rhs,xdeps.refs.MulExpr):
            ex=ex._lhs+(-1*ex._rhs._lhs)*ex._rhs._rhs
        else:
            ex=ex._lhs+(-1)*ex._rhs
        return lst+termlist(ex._lhs)+termlist(ex._rhs)
    else:
        return [ex]

def pprint(expr):
    return "\n + ".join([str(t) for t in termlist(expr)])


class LHCXsuiteModel:
    @classmethod
    def from_madxfile(cls, madxfile, sliced=False):
        """
        Create a LHCXsuiteModel from a MAD-X input file
        """
        madmodel = LHCMadModel.from_madxfile(madxfile)

        self = cls.from_madx(madmodel.madx, sliced=sliced, madxfile=madxfile)
        self.madxfile = madxfile
        return self

    @classmethod
    def from_madx(cls, madx, sliced=False, madxfile=None):
        import xtrack as xt

        lines = xt.Multiline.from_madx(
            madx=madx, enable_layout_data=True, return_lines=True
        )
        lines["b1"] = lines.pop("lhcb1")
        lines["b2"] = lines.pop("lhcb2")
        lines["b1"].particle_ref = xt.Particles(
            mass0=xt.PROTON_MASS_EV, p0c=450e9
        )
        lines["b2"].particle_ref = xt.Particles(
            mass0=xt.PROTON_MASS_EV, p0c=450e9
        )
        lines["b1"].twiss_default["method"] = "4d"
        lines["b2"].twiss_default["method"] = "4d"
        lines["b1"].twiss_default["co_search_at"] = "ip7"
        lines["b2"].twiss_default["co_search_at"] = "ip7"
        lines["b1"].twiss_default["strengths"] = True
        lines["b2"].twiss_default["strengths"] = True
        lines["b1"].twiss_default["compute_chromatic_properties"] = False
        lines["b2"].twiss_default["compute_chromatic_properties"] = False
        if sliced:
            for ln, ll in list(lines.items()):
                ls = ll.copy()
                ls.slice_thick_elements(
                    slicing_strategies=[
                        xt.Strategy(slicing=None),
                        xt.Strategy(
                            slicing=xt.Uniform(8, mode="thick"), name="mb.*"
                        ),
                        xt.Strategy(
                            slicing=xt.Uniform(8, mode="thick"), name="mq.*"
                        ),
                    ]
                )
                lines[f"{ln}s"] = ls

        lhc = xt.Multiline(lines)
        lines["b1"].build_tracker()
        lines["b2"].build_tracker()
        out = cls(multiline=lhc, madxfile=madxfile)
        return out

    @classmethod
    def from_json(cls, jsonfile):
        import xtrack as xt

        lhc = xt.Multiline.from_json(jsonfile)
        return cls(lhc, jsonfile=jsonfile)

    def to_json(self, jsonfile="lhc.json"):
        self.multiline.to_json(jsonfile)

    def __init__(
        self,
        multiline=None,
        optics=None,
        settings=None,
        jsonfile=None,
        madxfile=None,
    ):
        self.multiline = multiline
        self.optics = optics
        self.settings = settings
        self.jsonfile = jsonfile
        self._var_values = multiline._xdeps_vref._owner
        self.vars = multiline._xdeps_vref
        self.mgr = multiline._xdeps_manager
        self.madxfile = madxfile
        self.sequence = {1: multiline.b1, 2: multiline.b2}

    def __repr__(self):
        if self.madxfile is not None:
            return f"<LHCXsuiteModel {self.madxfile!r}>"
        elif self.jsonfile is not None:
            return f"<LHCXsuiteModel {self.jsonfile!r}>"
        else:
            return f"<LHCXsuiteModel {id(self)}>"

    @property
    def b1(self):
        return self.multiline.b1

    @property
    def b2(self):
        return self.multiline.b2

    @property
    def b1s(self):
        return self.multiline.b1s

    @property
    def b2s(self):
        return self.multiline.b2s

    def __getitem__(self, key):
        return self._var_values[key]

    def __contains__(self, key):
        return key in self._var_values

    def __setitem__(self, key, value):
        self.vars[key] = value

    def update_vars(self, strengths, verbose=False):
        for k, v in strengths.items():
            if verbose:
                print(f"{k:20} {v:15.6g}")
            self[k] = v

    def update_knobs(self, knobs, verbose=False):
        for k, knob in knobs.items():
            if verbose:
                print(f"Update knob {k:20} {knob.value:15.6g}")
            self.update_knob(knob)

    def knob_check(self, knob, verbose=False):
        """
        Return True has the expeceted structure
        Return False has a different structure
        Return None if it does not exist
        """
        knobname = knob.name
        deps = self.mgr.rdeps.get(self.vars[knobname], {})
        if len(deps) == 0:
            if verbose:
                print(f"Missing knob {knobname}")
            return None
        depnames = {dep._key for dep in deps}
        if verbose:
            print(f"Check knob {knobname}")
            print(f"Dependencies {depnames!r}")
        if depnames != set(knob.weights.keys()):
            if verbose:
                print(
                    f"Dependencies do not match with {set(knob.weights.keys())}"
                )
            return False
        else:
            for depname in depnames:
                wname = f"{depname}_from_{knobname}"
                if wname not in self._var_values:
                    if verbose:
                        print(f"Missing weight {wname}")
                    return False
            return True

    def update_knob(self, knob, verbose=False):
        """
        Update the model with the knob

        Check that the knob exists, that is k has dependent targets.
        If it exists, check that has the same structure
        else raise an error.
        If it does not exist, create it.

        """
        knobname = knob.name
        check = self.knob_check(knob)
        if check is False:
            self.knob_check(knob, verbose=verbose)
            raise ValueError(f"Knob {knobname} has different structure")
        else:
            self[knobname] = knob.value
            for wtarget, value in knob.weights.items():
                wname = f"{wtarget}_from_{knobname}"
                self[wname] = value
                if check is None:
                    self.vars[wtarget] += (
                        self.vars[wname] * self.vars[knobname]
                    )

    def _erase_knob(self, knob):
        """
        Delete knobs and weights from the model, very unsafe, because it can break other knobs. To be used only in tests.
        """
        knobname = knob.name
        deps = list(self.mgr.rdeps.get(self.vars[knobname], {}))
        for dep in deps:
            wname = f"{dep._key}_from_{knobname}"
            self.mgr.set_value(dep, dep._get_value())
            if wname in self._var_values:
                del self._var_values[wname]

    def get_knob(self, knob):
        value = self._var_values[knob.name]
        weights = {}
        for wname in knob.weights:
            weights[wname] = self._var_values[f"{wname}_from_{knob.name}"]
        return Knob(knob.name, value, weights)
 
    def show_knob(self, knobname):
        print(f"Knob: {knobname} = {self[knobname]:15.6g}")
        for deps in self.mgr.rdeps.get(self.vars[knobname], {}):
            print("Target:", deps._key)
            print("     Expr:", pprint(deps._expr))
            wname=f"{deps._key}_from_{knobname}"
            if wname in self:
                print(f"    Weight {wname} = {self[wname]:15.6g}")

    def get_knob_by_naming(self, name):
        weights = {}
        wname = f"_from_{name}"
        for k in self._var_values:
            if k.endswith(wname):
                weights[k] = self._var_values[k]
        value = self._var_values[name]
        return Knob(name, value, weights)

    def update(self, src):
        if hasattr(src, "strengths"):
            self.update_vars(src.strengths)
        else:
            self.update_vars(src)
        if hasattr(src, "knobs"):
            self.update_knobs(src.knobs)

    def twiss(
        self,
        start=None,
        end=None,
        init=None,
        beam=None,
        full=False,
        chrom=False,
    ):
        if beam is None:
            return self.twiss(
                start=start, end=end, init=init, beam=1, full=full, chrom=chrom
            ), self.twiss(
                start=start, end=end, init=init, beam=2, full=full, chrom=chrom
            )
        line = self.sequence[beam]
        startout = line.element_names[0] if start is None else start
        endout = line.element_names[-1] if end is None else end
        startmac = line.element_names[0] if full is None else start
        endmac = line.element_names[-1] if full is None else end
        if init is None:
            tw = line.twiss(start=startmac, end=endmac, init="periodic")
            if full:
                tw = tw.rows[startout:endout]  # can fail because of cycle
        else:
            tw = line.twiss(start=startout, end=endout, init=init)

    def copy(self):
        return self.__class__(
            multiline=self.multiline.copy(),
            settings=self.settings,
            jsonfile=self.jsonfile,
            madxfile=self.madxfile,
        )

    def diff(self, other):
        allk = set(self._var_values.keys()) | set(other._var_values.keys())
        for k in allk:
            if k not in self._var_values:
                print(f"{k:20} {other._var_values[k]:15.6g} only in other")
            elif k not in other._var_values:
                print(f"{k:20} {self._var_values[k]:15.6g} only in self")
            elif self._var_values[k] != other._var_values[k]:
                print(
                    f"{k:20} {self._var_values[k]:15.6g} {other._var_values[k]:15.6g}"
                )
