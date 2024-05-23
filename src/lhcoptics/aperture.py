import numpy as np
import matplotlib.pyplot as plt


class RectEllipse:
    def __init__(self, rh, rv, ah, av):
        self.rh = rh
        self.rv = rv
        self.ah = ah
        self.av = av

    def get_bbox(self):
        return min(self.rh, self.ah), min(self.rv, self.av)


class Ellipse:
    def __init__(self, rh, rv):
        self.rh = rh
        self.rv = rv

    def get_bbox(self):
        return self.rh, self.rv


class Rectangle:
    def __init__(self, ah, av):
        self.ah = ah
        self.av = av

    def get_bbox(self):
        return self.ah, self.av


class LHCApertureModel:
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
            for ii in len(su):
                name = su.name[ii]
                if name.split('..')[0].split('_')[0] in ld:
                    data = ld[name]
                    aperture = data["aperture"]
                    if aperture in profile_def:
                        profile_id = profile_def[aperture]
                        profile = profiles[profile_id]
                    else:
                        profile_def[aperture] = len(profiles)
                        profile = globals()[aperture[0].capitalize()](
                            *aperture[1:]
                        )
                        profiles[len(profiles)] = profile
                    ah, av = profile.get_bbox()
                    offsets = data["offsets"]
                    dx, dy = offsets[:2]
                    if len(offsets):
                        x = su.X[ii] - dx
                        y = su.Y[ii] - dy
                    apertures.append((name, x, y, ah, av, profile_id))
                else:
                    apertures.append((name, 0, 0, 0, 0, -1))
        return cls(apertures, profiles, model, survey)

    def __init__(self, apertures, profiles, model=None, survey=None):
        self.apertures = apertures
        self.profiles = profiles
        self.survey = survey
        self.model = model
