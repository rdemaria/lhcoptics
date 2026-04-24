from .irs import (
    LHCIR,
    gen_acb_full_names,
    gen_acbx_names,
    gen_d12_names,
    gen_qq,
    gen_qt,
    gen_qtl,
    gen_ttlhc,
)
from .section import gen_acb_alt_names

class LHCIR8(LHCIR):
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
    name = "ir8"

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

    def gen_acb_names(self):
        out = []
        out.extend(gen_acbx_names(self.irn))
        out.extend(gen_acb_full_names("y", "s4", self.irn))
        out.extend(gen_acb_alt_names("y", [4], 1, "lr", self.irn))
        out.extend(f"acbc{hv}s5.l8b{beam}" for hv in "hv" for beam in "12")
        out.extend(f"acby{hv}s5.r8b{beam}" for hv in "hv" for beam in "12")
        out.extend(["acbch5.l8b1", "acbcv5.l8b2", "acbyv5.r8b1", "acbyh5.r8b2"])
        out.extend(gen_acb_alt_names("c", range(6, 11), 1, "lr", self.irn))
        out.extend(gen_acb_alt_names("", range(11, 14), 1, "lr", self.irn))
        return out

    def gen_bend_names(self):
        return gen_d12_names(self.irn, self.variant)

    def gen_experiment_names(self):
        return ["abxws.l8","abxwh.l8","ablw.r8","abxws.r8"]

    def gen_knob_names(self):
        out = []
        if self.variant.startswith("hl"):
            out.extend(
                f"on_{kk}{self.irn}{hv}" for kk in ["a", "o", "sep", "x"] for hv in "hv"
            )
            out.extend(f"on_{xy}ip{self.irn}b{beam}" for xy in "xy" for beam in "12")
        else:
            out.append(f"on_a{self.irn}")
            out.extend(
                f"on_{kk}{self.irn}{hv}" for kk in ["x", "sep"] for hv in "hv"
            )
            out.extend(f"on_o{hv}{self.irn}" for hv in "hv")
            out.extend(f"on_{xy}ip{self.irn}b{beam}" for xy in "xy" for beam in "12")

        #TODO need to support energy dependent knobs
        #out.append(f"on_lhcb")
        return out

    def gen_quad_names(self):
        quads = []
        quads.extend(gen_ttlhc(self.irn))
        quads.extend(gen_qq(range(4, 11), self.irn))
        quads.extend(gen_qtl([11], self.irn))
        quads.extend(gen_qt([12, 13], self.irn))
        return quads

    def gen_strength_names(self):
        out = []
        out.extend(self.gen_quad_names())
        out.extend(self.gen_bend_names())
        out.extend(self.gen_acb_names())
        out.extend(self.gen_experiment_names())
        return out

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
