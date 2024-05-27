import numpy as np
import matplotlib.pyplot as plt
import json


class Rectellipse:
    """
    Intersect a rectangle with an ellipse.

    ah, av: half-widths of the rectangle
    rh, rv: half-widths of the ellipse
    """

    @classmethod
    def from_dict(cls, data):
        return cls(data["ah"], data["av"], data["rh"], data["rv"])

    @classmethod
    def from_layout_spec(cls, spec):
        return cls(*spec)

    def __init__(self, ah, av, rh, rv):
        self.ah = ah
        self.av = av
        self.rh = rh
        self.rv = rv

    def bbox(self):
        return min(self.rh, self.ah), min(self.rv, self.av)

    def __repr__(self):
        return f"Rectellipse({self.ah}, {self.av}, {self.rh}, {self.rv})"

    def to_dict(self):
        return {
            "shape": "rectellipse",
            "ah": self.ah,
            "av": self.av,
            "rh": self.rh,
            "rv": self.rv,
        }


class Ellipse:
    @classmethod
    def from_dict(cls, data):
        return cls(data["rh"], data["rv"])

    @classmethod
    def from_layout_spec(cls, spec):
        return cls(spec[0], spec[1])

    def __init__(self, rh, rv):
        self.rh = rh
        self.rv = rv

    def bbox(self):
        return self.rh, self.rv

    def __repr__(self):
        return f"Ellipse({self.rh}, {self.rv})"

    def to_dict(self):
        return {
            "shape": "ellipse",
            "rh": self.rh,
            "rv": self.rv,
        }


class Rectangle:
    @classmethod
    def from_dict(cls, data):
        return cls(data["ah"], data["av"])

    @classmethod
    def from_layout_spec(cls, spec):
        return cls(spec[0], spec[1])

    def __init__(self, ah, av):
        self.ah = ah
        self.av = av

    def bbox(self):
        return self.ah, self.av

    def __repr__(self):
        return f"Rectangle({self.ah}, {self.av})"

    def to_dict(self):
        return {
            "shape": "rectangle",
            "ah": self.ah,
            "av": self.av,
        }


class Circle:
    @classmethod
    def from_dict(cls, data):
        return cls(data["r"])

    @classmethod
    def from_layout_spec(cls, spec):
        return cls(spec[0])

    def __init__(self, r):
        self.r = r

    def bbox(self):
        return self.r, self.r

    def __repr__(self):
        return f"Circle({self.r})"

    def to_dict(self):
        return {
            "shape": "circle",
            "r": self.r,
        }


class Octagon:
    """
    Octagon with sides parallel to the axes and 45 degrees.

    ah, av: horizontal, vertical half-widths
    ar: half-width at 45 degrees
    """

    @classmethod
    def from_dict(cls, data):
        return cls(data["ah"], data["av"], data["ar"])

    @classmethod
    def from_layout_spec(cls, spec):
        return cls(spec[0], spec[1], spec[3])

    def __init__(self, ah, av, ar):
        self.ah = ah
        self.av = av
        self.ar = ar

    def bbox(self):
        return self.ah, self.av

    def __repr__(self):
        return f"Octagon({self.ah}, {self.av}, {self.ar})"

    def to_dict(self):
        return {
            "shape": "octagon",
            "ah": self.ah,
            "av": self.av,
            "ar": self.ar,
        }


class Racetrack:
    @classmethod
    def from_dict(cls, data):
        return cls(data["ah"], data["av"], data["rh"], data["rv"])

    @classmethod
    def from_layout_spec(cls, spec):
        return cls(spec[0], spec[1], spec[2], spec[3])

    def __init__(self, ah, av, rh, rv):
        self.ah = ah
        self.av = av
        self.rh = rh
        self.rv = rv

    def bbox(self):
        return self.ah, self.av

    def __repr__(self):
        return f"Racetrack({self.ah}, {self.av}, {self.rh}, {self.rv})"

    def to_dict(self):
        return {
            "shape": "racetrack",
            "ah": self.ah,
            "av": self.av,
            "rh": self.rh,
            "rv": self.rv,
        }


class LHCAperture:
    """
    twiss: s,x,y,betx,bety,dx,dy,pt
    survey: su_x, su_y
    aperture: ap_h,ap_v,ap_x,ap_y,ap_phi,ap_profile
    """

    @classmethod
    def from_xsuite_model(cls, model):
        """
        Create a LHCApertureModel from layout data and a survey.

        """
        layout_data = [
            model.b1.metadata["layout_data"],
            model.b2.metadata["layout_data"],
        ]
        survey = model.get_survey_flat()
        profile_def = {}
        profiles = {}
        apertures = []
        for ld, su in zip(layout_data, survey):
            for ii in range(len(su)):
                name = su.name[ii]
                if name.split("..")[0].split("_")[0] in ld:
                    data = ld[name]
                    shape, spec, tols = data["aperture"]
                    tols = tuple(tols)
                    spec = tuple(spec)
                    aperture = (shape, spec)
                    if aperture in profile_def:
                        profile_id = profile_def[aperture]
                        profile = profiles[profile_id]
                    else:
                        profile_id = len(profiles)
                        profile_def[aperture] = profile_id
                        profile_cls = globals()[shape.capitalize()]
                        profile = profile_cls.from_layout_spec(spec)
                        profiles[len(profiles)] = profile
                    ah, av = profile.bbox()
                    offset = data.get("offset", (0, 0))
                    if len(offset) == 0:
                        dx, dy, dpsi = 0, 0, 0
                    elif len(offset) == 2:
                        dx, dy = offset[:2]
                        dpsi = 0
                    elif len(offset) == 3:
                        dx, dy, dpsi = offset
                    dx = su.X[ii] - dx
                    dy = su.Y[ii] - dy
                    apertures.append(
                        (name, (dx, dy, dpsi), (ah, av), profile_id, tols)
                    )

        return cls(
            np.array(apertures, dtype=cls.dtype), profiles, model, survey
        )

    dtype = np.dtype(
        [
            ("name", "U64"),
            ("offset", "3f8"),
            ("bbox", "2f8"),
            ("profile", "i4"),
            ("tols", "3f8"),
        ]
    )

    def __init__(self, apertures, profiles, model=None, survey=None):
        self.apertures = apertures
        self.profiles = profiles
        self.survey = survey
        self.model = model

    @property
    def name(self):
        return self.apertures["name"]

    @property
    def offset(self):
        return self.apertures["offset"]

    @property
    def bbox(self):
        return self.apertures["bbox"]

    @property
    def profile(self):
        return self.apertures["profile"]

    def to_dict(self):
        apertures = [
            (name, offset.tolist(), bbox.tolist(), int(profile), tols.tolist())
            for name, offset, bbox, profile, tols in self.apertures
        ]
        return {
            "apertures": apertures,
            "profiles": [p.to_dict() for p in self.profiles.values()],
        }

    def to_json(self, fn):
        return json.dump(self.to_dict(), open(fn, "w"), indent=2)

    @classmethod
    def from_json(cls, fn):
        data = json.load(open(fn))
        apertures = np.array(
            list(map(tuple, data["apertures"])), dtype=cls.dtype
        )
        profiles = {
            ii: globals()[d["shape"].capitalize()].from_dict(d)
            for ii, d in enumerate(data["profiles"])
        }
        return cls(apertures, profiles)
