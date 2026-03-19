import re

from .irs import LHCIR


class LHCIR1(LHCIR):
    """IR1-specific optics model."""

    knob_names = [
        "on_x1_h",
        "on_sep1_h",
        "on_x1_v",
        "on_sep1_v",
        "on_xip1b1",
        "on_xip1b2",
        "on_oh1",
        "on_yip1b2",
        "on_yip1b1",
        "on_ov1",
    ]
    name = "ir1"

    @property
    def quads(self):
        if self.variant == "2025":
            return {
                k: v
                for k, v in self.strengths.items()
                if re.match("kt?q[^s]", k) and not k.startswith("kq4")
            }
        else:
            return LHCIR.quads.__get__(self)

    def has_ats_phase(self):
        return self.params["betxip1b1"] <= 2.5
