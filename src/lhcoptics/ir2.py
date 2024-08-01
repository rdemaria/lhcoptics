from .irs import LHCIR


class LHCIR2(LHCIR):
    name = "ir2"
    knob_names = [
        "on_a2",
        "on_oh2",
        "on_ov2",
        "on_sep2h",
        "on_sep2v",
        "on_x2h",
        "on_x2v",
        "on_xip2b1",
        "on_xip2b2",
        "on_yip2b1",
        "on_yip2b2",
    ]
    
    def set_init(self):
        if self.parent.is_ats():
            self.init_left={
                1: self.parent.ir1.get_init_ats_right(1),
                2: self.parent.ir1.get_init_ats_right(2),
            }
        else:
            self.init_left = {
               1: self.arc_left.get_init_right(1),
               2: self.arc_left.get_init_right(2),
        }        
    
        self.init_right = {
                1: self.arc_right.get_init_left(1),
                2: self.arc_right.get_init_left(2),
            }
