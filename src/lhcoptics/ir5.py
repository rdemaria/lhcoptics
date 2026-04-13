from .ir15 import LHCIR15

class LHCIR5(LHCIR15):
    name = "ir5"

    def gen_experiment_names(self):
        return ["abcs"]