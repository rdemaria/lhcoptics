from .irs import (
    LHCIR,
    gen_acb_full_names,
    gen_acbx_names,
    gen_d12_names,
    gen_param_names,
    gen_qq,
    gen_qt,
    gen_qtl,
    gen_ttlhc,
)
from .section import gen_acb_alt_names


class LHCIR2(LHCIR):
    """IR2-specific optics model."""

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
    name = "ir2"

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

    def gen_acb_names(self):
        out = []
        out.extend(gen_acbx_names(self.irn))
        out.extend(gen_acb_full_names("y", "s4", self.irn))
        out.extend(gen_acb_alt_names("y", [4], 1, "lr", self.irn))
        out.extend(f"acbc{hv}s5.l2b{beam}" for hv in "hv" for beam in "12")
        out.extend(f"acby{hv}s5.r2b{beam}" for hv in "hv" for beam in "12")
        out.extend(["acbyh5.l2b1", "acbyv5.l2b2", "acbcv5.r2b1", "acbch5.r2b2"])
        out.extend(gen_acb_alt_names("c", range(6, 11), 1, "lr", self.irn))
        out.extend(gen_acb_alt_names("", range(11, 14), 1, "lr", self.irn))
        return out

    def gen_bend_names(self):
        return gen_d12_names(self.irn)

    def gen_experiment_names(self):
        return ["abls","abxwt.l2","abwmd.l2","abaw.r2","abxwt.r2"]

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
        #out.extend([f"on_alice", f"on_sol_alice"])
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
