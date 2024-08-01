from .irs import LHCIR


class LHCIR8(LHCIR):
    name = "ir8"
    knob_names = [
        "on_x8h",
        "on_sep8h",
        "on_x8v",
        "on_sep8v",
        "on_o8",
        "on_oh8",
        "on_xip8b2",
        "on_xip8b1",
        "on_a8",
        "on_ov8",
        "on_yip8b1",
        "on_yip8b2",
    ]

    def set_init(self):
        self.init_left = {
            1: self.arc_left.get_init_right(1),
            2: self.arc_left.get_init_right(2),
        }

        if self.parent.is_ats():
            self.init_right = {
                1: self.parent.ir1.get_init_ats_left(1),
                2: self.parent.ir1.get_init_ats_left(2),
            }
        else:
            self.init_right = {
                1: self.arc_right.get_init_left(1),
                2: self.arc_right.get_init_left(2),
            }
