from .irs import LHCIR


class LHCIR4(LHCIR):
    name = "ir4"

    def set_init(self):
        self.init_left = {
               1: self.arc_left.get_init_right(1),
               2: self.arc_left.get_init_right(2),
        }        
    
        if self.parent.is_ats():
            self.init_right={
                1: self.parent.ir5.get_init_ats_left(1),
                2: self.parent.ir5.get_init_ats_left(2),
            }
        else:
            self.init_right = {
                1: self.arc_right.get_init_left(1),
                2: self.arc_right.get_init_left(2),
            }