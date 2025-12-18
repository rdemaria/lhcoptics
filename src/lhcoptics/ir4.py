from .irs import LHCIR


class LHCIR4(LHCIR):
    name = "ir4"

    def check_ats(self, beam):
        twa = self.twiss_ats_ip(beam)
        twb = self.twiss_ats_init(beam)
        print(f"beam {beam} ats check")
        for kk in "betx bety".split():
            vva = twa[kk, "ip5"]
            vvb = twb[kk, "ip5"]
            print(f"{kk:5} at ip5: {vva:9.6f} {vvb:9.6f}")
        for kk in "mux muy".split():
            vva = twa[kk, "ip5"] - twa[kk][0]
            vvb = twb[kk, "ip5"] - twb[kk][0]
            print(f"{kk:5} at ip1: {vva:9.6f} {vvb:9.6f}")

    def get_mux_ats(self, beam):
        line = self.model.sequence[beam]
        ir5 = self.parent.ir5
        ds = f"e.ds.r4.b{beam}"
        tw = line.twiss(
            start=ds,
            end="ip5",
            init_at="ip5",
            betx=ir5.params[f"betxip5b{beam}"],
            bety=ir5.params[f"betyip5b{beam}"],
        )
        return -tw["mux", ds], -tw["muy", ds]

    def get_init_ats(self, beam):
        rx = self.parent.params["rx_ip5"]
        ry = self.parent.params["ry_ip5"]
        line = self.model.sequence[beam]
        ir5 = self.parent.ir5
        ds = f"e.ds.r4.b{beam}"
        tw = line.twiss(
            start=ds,
            end="ip5",
            init_at="ip5",
            betx=ir5.params[f"betxip5b{beam}"] / rx,
            bety=ir5.params[f"betyip5b{beam}"] / ry,
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
        ir5 = self.parent.ir5
        rx = self.parent.params["rx_ip5"]
        ry = self.parent.params["ry_ip5"]
        tw = line.twiss(
            start=f"s.ds.l4.b{beam}",
            end="ip5",
            init_at="ip5",
            betx=ir5.params[f"betxip5b{beam}"] / rx,
            bety=ir5.params[f"betyip5b{beam}"] / ry,
        )
        return tw
