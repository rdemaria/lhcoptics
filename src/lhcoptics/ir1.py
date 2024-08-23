from .irs import LHCIR
import re


class LHCIR1(LHCIR):
    name = "ir1"
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

    @property
    def quads(self):
        return {
            k: v
            for k, v in self.strengths.items()
            if re.match("kt?q[^s]", k) and not k.startswith("kq4")
        }
