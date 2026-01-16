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

    def has_ats_phase(self):
        return self.params['betxip5b1']<=2.5


    # def get_mux_left(self, beam):
    #     line = self.model.sequence[beam]
    #     ds = f"e.ds.r4.b{beam}"
    #     tw = line.twiss(
    #         start=f"e.ds.r4.b{beam}",
    #         end="ip5",
    #         init_at="ip5",
    #         betx=self.params[f"betxip5b{beam}"],
    #         bety=self.params[f"betyip5b{beam}"],
    #         backtracking=True,
    #     )
    #     return tw["mux", ds], tw["muy", ds]

    # def get_mux_right(self, beam):
    #     line = self.model.sequence[beam]
    #     ds = f"s.ds.l6.b{beam}"
    #     tw = line.twiss(
    #         start="ip5",
    #         end=ds,
    #         betx=self.params[f"betxip5b{beam}"],
    #         bety=self.params[f"betyip5b{beam}"],
    #     )
    #     return tw["mux", ds], tw["muy", ds]

    # def get_init_ats_left(self,beam):
    #     line = self.model.sequence[beam]
    #     ds = f"e.ds.r4.b{beam}"
    #     tw = line.twiss(
    #         start=ds,
    #         end="ip5",
    #         init_at="ip5",
    #         betx=self.params[f"betxip5b{beam}"],
    #         bety=self.params[f"betyip5b{beam}"],
    #     )
    #     init = tw.get_twiss_init(ds)
    #     mux, muy = self.get_mux_right(1)
    #     init.mux -= mux
    #     init.muy -= muy
    #     return init

    # def get_init_ats_right(self,beam):
    #     ds = f"s.ds.l6.b{beam}"
    #     line = self.model.sequence[beam]
    #     tw = line.twiss(
    #         start="ip5",
    #         end=ds,
    #         betx=self.params[f"betxip5b{beam}"],
    #         bety=self.params[f"betyip5b{beam}"],
    #     )
    #     init = tw.get_twiss_init(ds)
    #     mux, muy = self.get_mux_right(beam)
    #     init.mux -= mux
    #     init.muy -= muy
    #     return init


        


