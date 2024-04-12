from .irs import LHCIR


class LHCIR2(LHCIR):
    name = "ir2"
    knobs = [
        "on_x2h",
        "on_sep2h",
        "on_x2v",
        "on_sep2v",
        "on_a2",
        "on_oh2",
        "on_xip2b2",
        "on_xip2b1",
        "on_o2",
        "on_ov2",
        "on_yip2b1",
        "on_yip2b2",
    ]
