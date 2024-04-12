from .model_madx import LHCMadModel

class LHCXsuiteModel:
    @classmethod
    def from_madxfile(cls, madxfile, sliced=False):
        """
        Create a LHCXsuiteModel from a MAD-X input file
        """
        madmodel=LHCMadModel.from_madxfile(madxfile)

        self=cls.from_madx(madmodel.madx, sliced=sliced, madxfile=madxfile)
        self.madxfile=madxfile
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

    def update(self, src):
        if hasattr(src, "strengths"):
            self.update_vars(src.strengths)
        else:
            self.update_vars(src)
        if hasattr(src, "knobs"):
            self.update_knobs(src.knobs)



