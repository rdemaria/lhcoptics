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

    def check_ats(self, beam):
        twa = self.twiss_ats_ip(beam)
        twb = self.twiss_ats_init(beam)
        print(f"beam {beam} ats check")
        for kk in "betx bety".split():
            vva = twa[kk, "ip1"]
            vvb = twb[kk, "ip1"]
            print(f"{kk:5} at ip1: {vva:9.6f} {vvb:9.6f}")
        for kk in "mux muy".split():
            vva = twa[kk, f"e.ds.r2.b{beam}"] - twa[kk][0]
            vvb = twb[kk, f"e.ds.r2.b{beam}"] - twb[kk][0]
            print(f"{kk:5} at ip1: {vva:9.6f} {vvb:9.6f}")

    def get_mux_ats(self, beam):
        line = self.model.sequence[beam]
        ir1 = self.parent.ir1
        ds = f"s.ds.l2.b{beam}"
        tw = line.twiss(
            start="ip1",
            end=ds,
            betx=ir1.params[f"betxip1b{beam}"],
            bety=ir1.params[f"betyip1b{beam}"],
        )
        return tw["mux", ds], tw["muy", ds]

    def get_init_ats(self, beam):
        rx = self.parent.params["rx_ip1"]
        ry = self.parent.params["ry_ip1"]
        line = self.model.sequence[beam]
        ir1 = self.parent.ir1
        ds = f"s.ds.l2.b{beam}"
        tw = line.twiss(
            start="ip1",
            end=ds,
            betx=ir1.params[f"betxip1b{beam}"] / rx,
            bety=ir1.params[f"betyip1b{beam}"] / ry,
        )
        init = tw.get_twiss_init(ds)
        mux, muy = self.get_mux_ats(beam)
        init.mux -= mux
        init.muy -= muy
        return init

    def set_init_ats(self, beam):
        self.init_left[beam] = self.get_init_ats(beam)

    def set_init_left(self, beam):
        if self.parent.is_ats():
            self.set_init_ats(beam)
        else:
            LHCIR.set_init_left(self, beam)

    def twiss_ats_init(self, beam):
        line = self.model.sequence[beam]
        tw = line.twiss(
            init=self.init_right[beam],
            start="ip1",
            end=self.init_right[beam].element_name,
        )
        return tw

    def twiss_ats_ip(self, beam):
        line = self.model.sequence[beam]
        ir1 = self.parent.ir1
        rx = self.parent.params["rx_ip1"]
        ry = self.parent.params["ry_ip1"]
        tw = line.twiss(
            start="ip1",
            end=f"e.ds.r2.b{beam}",
            init_at="ip1",
            betx=ir1.params[f"betxip1b{beam}"] / rx,
            bety=ir1.params[f"betyip1b{beam}"] / ry,
        )
        return tw
