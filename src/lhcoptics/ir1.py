from .irs import LHCIR

class LHCIR1(LHCIR):
    name = "ir1"
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

    def get_mux_left(self, beam):
        line = self.model.sequence[beam]
        ds = f"e.ds.r8.b{beam}"
        tw = line.twiss(
            start=f"e.ds.r8.b{beam}",
            end="ip1",
            init_at="ip1",
            betx=self.params[f"betxip1b{beam}"],
            bety=self.params[f"betyip1b{beam}"],
            backtracking=True,
        )
        return tw["mux", ds], tw["muy", ds]

    def get_mux_right(self, beam):
        line = self.model.sequence[beam]
        ds = f"s.ds.l2.b{beam}"
        tw = line.twiss(
            start="ip1",
            end=ds,
            betx=self.params[f"betxip1b{beam}"],
            bety=self.params[f"betyip1b{beam}"],
        )
        return tw["mux", ds], tw["muy", ds]

    def get_init_ats_left(self,beam):
        rx = self.parent.params["rx_ip1"]
        ry = self.parent.params["ry_ip1"]
        line = self.model.sequence[beam]
        ds = f"e.ds.r8.b{beam}"
        tw = line.twiss(
            start=ds,
            end="ip1",
            init_at="ip1",
            betx=self.params[f"betxip1b{beam}"] / rx,
            bety=self.params[f"betyip1b{beam}"] / ry,
        )
        init = tw.get_twiss_init(ds)
        mux, muy = self.get_mux_right(1)
        init.mux -= mux
        init.muy -= muy
        return init

    def get_init_ats_right(self,beam):
        rx = self.parent.params["rx_ip1"]
        ry = self.parent.params["ry_ip1"]
        line = self.model.sequence[beam]
        ds = f"s.ds.l2.b{beam}"
        tw = line.twiss(
            start="ip1",
            end=ds,
            betx=self.params[f"betxip1b{beam}"] / rx,
            bety=self.params[f"betyip1b{beam}"] / ry,
        )
        init = tw.get_twiss_init(ds)
        mux, muy = self.get_mux_right(beam)
        init.mux -= mux
        init.muy -= muy
        return init

