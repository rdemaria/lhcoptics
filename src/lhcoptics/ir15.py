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
from .irs import gen_d12_names


class LHCIR15(LHCIR):
    """IR15-specific optics model."""

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
            out.extend(gen_acb_alt_names("y", [4], 0, "lr", self.irn))
            out.extend(gen_acb_alt_names("c", range(5, 11), 0, "lr", self.irn))
            out.extend(gen_acb_alt_names("", range(11, 14), 0, "lr", self.irn))
        return out

    def gen_bend_names(self):
        if self.variant.startswith("hl"):
            return gen_d12_names(self.irn, self.variant)
        else:
            out=[f"ad1.lr{self.irn}",f"ad2.l{self.irn}", f"ad2.r{self.irn}"]
            out.extend([f"kd1.lr{self.irn}",f"kd2.l{self.irn}", f"kd2.r{self.irn}"])
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
            out.extend(f"on_{kk}{self.irn}_{hv}" for kk in ["x", "sep"] for hv in "hv")
            out.extend(f"on_o{hv}{self.irn}" for hv in "hv")
            out.extend(f"on_{xy}ip{self.irn}b{beam}" for xy in "xy" for beam in "12")
        # TODO need to support energy dependent knobs
        # if self.name == "ir1":
        #    out.append(f"on_sol_atlas")
        # elif self.name == "ir5":
        #    out.append(f"on_sol_cms")
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

    def gen_strength_names(self):
        out = []
        out.extend(self.gen_quad_names())
        out.extend(self.gen_bend_names())
        out.extend(self.gen_acb_names())
        if self.variant.startswith("hl"):
            out.extend(self.gen_crab_names())
        out.extend(self.gen_experiment_names())
        return out

    def has_ats_phase(self):
        return self.params[f"betxip{self.irn}b1"] <= 2.5


class LHCIR1(LHCIR15):
    """IR1-specific optics model."""

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
            return LHCIR15.quads.__get__(self)

    def gen_experiment_names(self):
        return ["abas"]


class LHCIR5(LHCIR15):
    name = "ir5"

    def gen_experiment_names(self):
        return ["abcs"]
