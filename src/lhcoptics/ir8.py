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

    def check_ats(self, beam):
        twa = self.twiss_ats_ip(beam)
        twb = self.twiss_ats_init(beam)
        print(f"beam {beam} ats check")
        for kk in "betx bety".split():
            vva = twa[kk, "ip1"]
            vvb = twb[kk, "ip1"]
            print(f"{kk:5} at ip1: {vva:9.6f} {vvb:9.6f}")
        for kk in "mux muy".split():
            vva = twa[kk, "ip1"] - twa[kk][0]
            vvb = twb[kk, "ip1"] - twb[kk][0]
            print(f"{kk:5} at ip1: {vva:9.6f} {vvb:9.6f}")

    def get_mux_ats(self, beam):
        line = self.model.sequence[beam]
        ir1 = self.parent.ir1
        ds = f"e.ds.r8.b{beam}"
        tw = line.twiss(
            start=ds,
            end="ip1",
            init_at="ip1",
            betx=ir1.params[f"betxip1b{beam}"],
            bety=ir1.params[f"betyip1b{beam}"],
        )
        return -tw["mux", ds], -tw["muy", ds]

    def get_init_ats(self, beam):
        rx = self.parent.params["rx_ip1"]
        ry = self.parent.params["ry_ip1"]
        line = self.model.sequence[beam]
        ir1 = self.parent.ir1
        ds = f"e.ds.r8.b{beam}"
        tw = line.twiss(
            start=ds,
            end="ip1",
            init_at="ip1",
            betx=ir1.params[f"betxip1b{beam}"] / rx,
            bety=ir1.params[f"betyip1b{beam}"] / ry,
        )
        init = tw.get_twiss_init(ds)
        mux, muy = self.get_mux_ats(beam)
        dmux = -tw["mux", ds] - mux
        dmuy = -tw["muy", ds] - muy
        return init, dmux, dmuy

    def set_init_ats(self, beam):
        initb, dmuxb, dmuyb = self.get_init_ats(beam)
        self.init_left[beam].mux = dmuxb
        self.init_left[beam].muy = dmuyb
        self.init_right[beam] = initb

    def set_init_right(self, beam):
        if self.parent.is_ats():
            return self.set_init_ats(beam)
        else:
            LHCIR.set_init_right(self, beam)

    def twiss_ats_init(self, beam):
        line = self.model.sequence[beam]
        tw = line.twiss(
            init=self.init_left[beam],
            start=self.init_left[beam].element_name,
            end="ip1",
        )
        return tw

    def twiss_ats_ip(self, beam):
        line = self.model.sequence[beam]
        ir1 = self.parent.ir1
        rx = self.parent.params["rx_ip1"]
        ry = self.parent.params["ry_ip1"]
        tw = line.twiss(
            start=f"s.ds.l8.b{beam}",
            end="ip1",
            init_at="ip1",
            betx=ir1.params[f"betxip1b{beam}"] / rx,
            bety=ir1.params[f"betyip1b{beam}"] / ry,
        )
        return tw
