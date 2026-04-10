import re

from .irs import (
    LHCIR,
    gen_acb_full_names,
    gen_acbx_names,
    gen_crab_names,
    gen_param_names,
    gen_qq,
    gen_qt,
    gen_qtl,
    gen_tthl,
    gen_ttlhc,
)
from .section import gen_acb_alt_names


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
    stregnth_names = {}

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

    def gen_acb_names(self):
        out = []
        out.extend(gen_acbx_names(self.irn))
        if self.variant.startswith("hl"):
            out.extend(gen_acb_full_names("rd", "4", self.irn))
        out.extend(gen_acb_full_names("y", "s4", self.irn))
        if self.variant.startswith("hl"):
            out.extend(gen_acb_alt_names("y", [4], 1, "lr", self.irn))
            out.extend(gen_acb_alt_names("c", range(5, 10), 0, "lr", self.irn))
            out.extend(gen_acb_alt_names("", range(10, 14), 0, "lr", self.irn))
        else:
            out.extend(gen_acb_alt_names("y", [4], 1, "lr", self.irn))
            out.extend(gen_acb_alt_names("c", range(5, 11), 0, "lr", self.irn))
            out.extend(gen_acb_alt_names("", range(11, 14), 0, "lr", self.irn))
        return out

    def gen_crab_names(self):
        return gen_crab_names(self.irn)

    def gen_knob_names(self):
        out = []
        if self.variant.startswith("hl"):
            out.extend(
                f"on_{kk}{self.irn}{hv}" for kk in ["a", "o", "sep"] for hv in "hv"
            )
            out.extend(f"on_x{self.irn}{hv}{ls}" for ls in "ls" for hv in "hv")
            out.extend(f"on_{xy}ip{self.irn}b{beam}" for xy in "xy" for beam in "12")
            out.append(f"on_crab{self.irn}")
        else:
            out.extend(
                f"on_{kk}{self.irn}_{hv}" for kk in ["x", "sep"] for hv in "hv"
            )
            out.extend(f"on_o{hv}{self.irn}" for hv in "hv")
            out.extend(f"on_{xy}ip{self.irn}b{beam}" for xy in "xy" for beam in "12")
        return out

    def gen_quad_names(self):
        quads = []
        if self.variant.startswith("hl"):
            quads.extend(gen_tthl(self.irn))
        else:
            quads.extend(gen_ttlhc(self.irn))
        quads.extend(gen_qq(range(4, 11), self.irn))
        quads.extend(gen_qtl([11], self.irn))
        quads.extend(gen_qt([12, 13], self.irn))
        return quads

    def has_ats_phase(self):
        return self.params["betxip1b1"] <= 2.5
