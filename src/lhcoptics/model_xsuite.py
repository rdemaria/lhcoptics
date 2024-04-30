from .model_madx import LHCMadModel
from .knob import Knob


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
    def from_madx(cls, madx, sliced=False):
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
        out = cls(multiline=lhc)
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
        self.vref = multiline._xdeps_vref
        self.mgr = multiline._xdeps_manager
        self.sequence = {1: multiline.b1, 2: multiline.b2}

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

    def __setitem__(self, key, value):
        self.vref[key] = value

    def update_vars(self, strengths):
        for k, v in strengths.items():
            self[k] = v

    def update_knobs(self, knobs):
        for k, knob in knobs.items():
            self[k] = knob.value
            for wn, value in knob.weights.items():
                name = f"{wn}_from_{k}"
                self[name] = value
                task = self.mgr.tasks.get(self.vref[wn])
                if (
                    task is not None
                    and k in task.dependencies
                    and name not in task.dependencies
                ):
                    raise ValueError(f"{wn} depends on {k} but not on {name}")
                if task is None or (
                    task is not None and k not in task.dependencies
                ):
                    self.vref[wn] += self.vref[name] * self.vref[k]

    def get_knob(self,knob):
        value= self._var_values[knob.name]
        weights = {}
        for wname in knob.weights:
            weights[wname]= self._var_values[f"{wname}_from_{knob.name}"]
        return Knob(knob.name,value,weights)


    def get_knob_by_naming(self, name):
        weights = {}
        wname= f"_from_{name}"
        for k in self._var_values:
            if k.endswith(wname):
                weights[k]= self._var_values[k]
        value = self._var_values[name]
        return Knob(name,value,weights)

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
