from .irs import (
    LHCIR,
    gen_acb_full_names,
    gen_acbx_names,
    gen_crab_names,
    gen_param_names,
    gen_qq,
    gen_qt,
    gen_qtl,
    gen_tthl,
    gen_ttlhc,
)
from .section import gen_acb_alt_names

class LHCIR5(LHCIR):
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
    name = "ir5"

    def gen_acb_names(self):
        out = []
        out.extend(gen_acbx_names(self.irn))
        if self.variant.startswith("hl"):
            out.extend(gen_acb_full_names("rd", "4", self.irn))
        out.extend(gen_acb_full_names("y", "s4", self.irn))
        if self.variant.startswith("hl"):
            out.extend(gen_acb_alt_names("y", [4], 1, "lr", self.irn))
            out.extend(gen_acb_alt_names("c", range(5, 10), 0, "lr", self.irn))
            out.extend(gen_acb_alt_names("", range(10, 14), 0, "lr", self.irn))
        else:
            out.extend(gen_acb_alt_names("y", [4], 1, "lr", self.irn))
            out.extend(gen_acb_alt_names("c", range(5, 11), 0, "lr", self.irn))
            out.extend(gen_acb_alt_names("", range(11, 14), 0, "lr", self.irn))
        return out

    def gen_crab_names(self):
        return gen_crab_names(self.irn)

    def gen_param_names(self):
        #TODO add triplet matching conditions
        return gen_param_names(self.irn)

    def gen_quad_names(self):
        quads = []
        if self.variant.startswith("hl"):
            quads.extend(gen_tthl(self.irn))
        else:
            quads.extend(gen_ttlhc(self.irn))
        quads.extend(gen_qq(range(4, 11), self.irn))
        quads.extend(gen_qtl([11], self.irn))
        quads.extend(gen_qt([12, 13], self.irn))
        return quads

    def has_ats_phase(self):
        return self.params["betxip5b1"] <= 2.5



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
