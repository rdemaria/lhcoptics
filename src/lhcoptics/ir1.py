import re

from .ir15 import LHCIR15

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
