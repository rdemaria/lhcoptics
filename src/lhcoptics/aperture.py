import numpy as np
import matplotlib.pyplot as plt
import json

import xdeps as xd


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


class AperList:
    dtype = np.dtype(
        [
            ("name", "U64"),
            ("ap_s", "f8"),
            ("offset", "3f8"),
            ("bbox", "2f8"),
            ("profile", "i4"),
            ("tols", "3f8"),
        ]
    )

    def __init__(self, apertures):
        self.apertures = np.array(apertures, np.dtype(self.dtype))
        self.lookup = {a["name"]: ii for ii, a in enumerate(self.apertures)}

    @property
    def name(self):
        return self.apertures["name"]

    @property
    def ap_s(self):
        return self.apertures["ap_s"]

    @property
    def offset(self):
        return self.apertures["offset"]

    @property
    def ap_x(self):
        return self.apertures["offset"][:, 0]

    @property
    def ap_y(self):
        return self.apertures["offset"][:, 1]

    @property
    def bbox(self):
        return self.apertures["bbox"]

    @property
    def profile(self):
        return self.apertures["profile"]

    @property
    def mask(self):
        return self.apertures["profile"] != -1

    def to_list(self):
        return [
            (
                name,
                s,
                offset.tolist(),
                bbox.tolist(),
                int(profile),
                tols.tolist(),
            )
            for name, s, offset, bbox, profile, tols in self.apertures
        ]

    def plotx(self, ax=None):
        """
        Plot the apertures for a given beam.
        """
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        mask = self.mask
        ap = self.apertures[mask]

        (line,) = plt.plot(ap["ap_s"], ap["offset"][:, 0], "k.-", label="ap_x")
        apx=ap["offset"][:, 0]+ap["bbox"][:, 0]
        plt.plot(ap["ap_s"], apx, "k.-", label="ap_x+")
        apx=ap["offset"][:, 0]-ap["bbox"][:, 0]
        plt.plot(ap["ap_s"], apx, "k.-", label="ap_x-")

        annot = ax.annotate(
            "",
            xy=(0, 0),
            xytext=(20, 20),
            textcoords="offset points",
            bbox=dict(boxstyle="round", fc="w"),
            arrowprops=dict(arrowstyle="->"),
        )
        annot.set_visible(False)

        def update_annot(event, ind):
            # pos = line.get_offsets()[ind["ind"][0]]
            annot.xy = event.xdata, event.ydata
            idxs = ind["ind"]
            ttplot = []
            ttprint = []
            for ii in idxs:
                row = ap[ii]
                ttplot.append(f"{row['name']}")
                ttprint.append(
                    f"{row['name']:20} {row['offset'][0]:5.3f} {row['offset'][1]:5.3f} {row['bbox'][0]:5.3f} {row['bbox'][1]:5.3f}"
                )
            print("\n".join(ttprint))
            annot.set_text("\n".join(ttplot))
            annot.get_bbox_patch().set_alpha(0.4)

        def hover(event):
            vis = annot.get_visible()
            if event.inaxes == ax:
                cont, ind = line.contains(event)
                if cont:
                    update_annot(event, ind)
                    annot.set_visible(True)
                    fig.canvas.draw_idle()
                else:
                    if vis:
                        annot.set_visible(False)
                        fig.canvas.draw_idle()

        fig.canvas.mpl_connect("motion_notify_event", hover)

        return ax

    @property
    def tab(self):
        dct = {
            "name": self.name,
            "s": self.ap_s,
            "x": self.ap_x,
            "y": self.ap_y,
        }
        return xd.Table(dct, index="name")


offsets1 = {
    "tanar.4r1": (0.08, 0.000, 0.000),
    "tanc.4l5": (-0.08, 0.000, 0.000),
    "tanc.4r5": (0.08, 0.000, 0.000),
    "tanal.4l1": (-0.08, 0.000, 0.000),
    "btvse.a4l6.b1": (-0.025, 0.000, 0.000),
    "bpmse.4l6.b1": (-0.025, 0.000, 0.000),
    "tcdsa.4l6.b1": (-0.025, 0.000, 0.000),
}

offsets2 = {
    "tanc.4l5": (-0.08, 0.000, 0.000),
    "tanc.4r5": (0.08, 0.000, 0.000),
    "tanal.4l1": (-0.08, 0.000, 0.000),
    "tanar.4r1": (0.08, 0.000, 0.000),
    "btvse.a4l6.b1": (0.025, 0.000, 0.000),
    "bpmse.4l6.b1": (0.025, 0.000, 0.000),
    "tcdsa.4l6.b1": (0.025, 0.000, 0.000),
}

offsets = (offsets1, offsets2)


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
        aplists = []
        for ld, su, off, beam in zip(layout_data, survey, offsets, (1, 2)):
            aplist = []
            for ii in range(len(su)):
                name = su.name[ii]
                # default values
                dx, dy, dpsi = 0, 0, 0
                ah, av = 0, 0
                profile_id = -1
                tols = (0, 0, 0)
                if name.split("..")[0].split("_")[0] in ld:
                    data = ld[name]
                    shape, spec, tols = data["aperture"]
                    if (
                        spec[0] < 1
                        and spec[0] > 0
                        and not name.startswith("x")
                        and not name.startswith("br")
                    ):
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
                        if name in off:
                            offset = off[name]
                        else:
                            offset = data.get("offset", (0, 0))
                        if len(offset) == 0:
                            dx, dy, dpsi = 0, 0, 0
                        elif len(offset) == 2:
                            dx, dy = offset[:2]
                            dpsi = 0
                        elif len(offset) == 3:
                            dx, dy, dpsi = offset
                        if beam == 2:  # layout data multiplied by bv!!!!!
                            dx = -dx
                            dy = -dy
                        dx = su.X[ii] - dx
                        dy = su.Y[ii] - dy
                aplist.append(
                    (
                        name,
                        su.s[ii],
                        (dx, dy, dpsi),
                        (ah, av),
                        profile_id,
                        tols,
                    )
                )
            aplists.append(AperList(aplist))
        return cls(aplists, profiles, model, survey)

    def __init__(self, apertures, profiles, model=None, survey=None):
        self.apertures = apertures
        self.profiles = profiles
        self.survey = survey
        self.model = model

    def to_dict(self):
        return {
            "apertures": [a.tolist() for a in self.apertures],
            "profiles": [p.to_dict() for p in self.profiles.values()],
        }

    def to_json(self, fn):
        return json.dump(self.to_dict(), open(fn, "w"), indent=2)

    @classmethod
    def from_json(cls, fn):
        data = json.load(open(fn))
        apertures = [
            AperList(list(map(tuple, aplist))) for aplist in data["apertures"]
        ]
        profiles = {
            ii: globals()[d["shape"].capitalize()].from_dict(d)
            for ii, d in enumerate(data["profiles"])
        }
        return cls(apertures, profiles)
