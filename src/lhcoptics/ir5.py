from .irs import LHCIR


class LHCIR5(LHCIR):
    name = "ir5"
    knob_names = [
        "on_x5_h",
        "on_sep5_h",
        "on_x5_v",
        "on_sep5_v",
        "on_xip5b1",
        "on_xip5b2",
        "on_oh5",
        "on_yip5b2",
        "on_yip5b1",
        "on_ov5",
    ]
