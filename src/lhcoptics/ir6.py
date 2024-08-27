import numpy as np

import xtrack as xt

from .irs import LHCIR


def beta_dump(bet, alf):
    al_dump = 761
    return bet - 2 * alf * al_dump + al_dump**2 * (1 + alf**2) / bet


def tcdq_mingap(bet, dx):
    nsig = 10.1
    dpoverp = 2e-4
    emitx = 2.5e-6 / (7000 / 0.9382720814)
    maxorbitdrift = 0.6e-3
    return (
        nsig * np.sqrt(emitx * bet) - 3e-4 - abs(dx * dpoverp) - maxorbitdrift
    )


def tcdq_gap(bet):
    nsigma = 10.1
    emitx = 2.5e-6 / (7000 / 0.9382720814)
    return nsigma * np.sqrt(emitx * bet)


def dmuxkickb1(tw1):
    muxmkdb1 = tw1["mux", "mkd.h5l6.b1"]
    dmuxb1 = (
        np.array([tw1["mux", f"tcdqa.{abc}4r6.b1"] for abc in "abc"])
        - muxmkdb1
    )
    return abs(dmuxb1 - 0.25).max()


def dmuxkickb2(tw2):
    muxmkdb2 = tw2["mux", "mkd.h5r6.b2"]
    dmuxb2 = muxmkdb2 - np.array(
        [tw2["mux", f"tcdqa.{abc}4l6.b2"] for abc in "abc"]
    )
    return abs(dmuxb2 - 0.25).max()


def dmuxkickb1_bds(tw1):
    return tw1["mux", "mkd.h5l6.b1"] - tw1["mux", "s.ds.l6.b1"]


def dmuxkickb2_bds(tw2):
    return tw2["mux", "mkd.h5r6.b2"] - tw2["mux", "s.ds.l6.b2"]


def dmuxkickb1_eds(tw1):
    return tw1["mux", "e.ds.r6.b1"] - tw1["mux", "mkd.h5l6.b1"]


def dmuxkickb2_eds(tw2):
    return tw2["mux", "e.ds.r6.b2"] - tw2["mux", "mkd.h5r6.b2"]


def betxdump(tw):
    return beta_dump(tw["betx", "ip6"], tw["alfx", "ip6"])


def betydump(tw):
    return beta_dump(tw["bety", "ip6"], tw["alfy", "ip6"])


def betxydump(tw):
    return np.sqrt(betxdump(tw) * betydump(tw))


class LHCIR6(LHCIR):
    name = "ir6"

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
        ir5 = self.parent.ir5
        ds = f"s.ds.l6.b{beam}"
        tw = line.twiss(
            start="ip5",
            end=ds,
            betx=ir5.params[f"betxip5b{beam}"],
            bety=ir5.params[f"betyip5b{beam}"],
        )
        return tw["mux", ds], tw["muy", ds]

    def get_init_ats(self, beam):
        rx = self.parent.params["rx_ip5"]
        ry = self.parent.params["ry_ip5"]
        line = self.model.sequence[beam]
        ir5 = self.parent.ir5
        ds = f"s.ds.l6.b{beam}"
        tw = line.twiss(
            start="ip5",
            end=ds,
            betx=ir5.params[f"betxip5b{beam}"] / rx,
            bety=ir5.params[f"betyip5b{beam}"] / ry,
        )
        init = tw.get_twiss_init(ds)
        mux, muy = self.get_mux_ats(beam)
        init.mux -= mux
        init.muy -= muy
        return init

    def get_params_from_twiss(self, tw1, tw2):
        params = LHCIR.get_params_from_twiss(self, tw1, tw2)

        muxmkdb1 = tw1["mux", "mkd.h5l6.b1"]
        params["dmuxkickb1_tcsg"] = tw1["mux", "tcsp.a4r6.b1"] - muxmkdb1
        dmuxb1 = (
            np.array([tw1["mux", f"tcdqa.{abc}4r6.b1"] for abc in "abc"])
            - muxmkdb1
        )
        for i, abc in enumerate("abc"):
            params[f"dmuxkickb1_tcdq{abc}"] = dmuxb1[i]
        params["dmuxkickb1"] = abs(dmuxb1 - 0.25).max()

        params["dxq5l6b1"] = tw1["dx", "mqy.5l6.b1"]
        params["dxq4r6b1"] = tw1["dx", "mqy.4r6.b1"]
        params["dxtcdqb1"] = tw1["dx", "tcdqa.a4r6.b1"]
        params["betxtcdqb1"] = tw1["betx", "tcdqa.a4r6.b1"]
        params["betytcdqb1"] = tw1["bety", "tcdqa.a4r6.b1"]
        params["betxtcdsb1"] = tw1["betx", "tcdsa.4l6.b1"]
        params["betytcdsb1"] = tw1["bety", "tcdsa.4l6.b1"]
        params["betxtcsgb1"] = tw1["betx", "tcsp.a4r6.b1"]
        params["betytcsgb1"] = tw1["bety", "tcsp.a4r6.b1"]
        params["betxmkdb1"] = tw1["betx", "mkd.h5l6.b1"]
        params["betymkdb1"] = tw1["bety", "mkd.h5l6.b1"]
        params["bxdumpb1"] = beta_dump(
            params["betxip6b1"], params["alfxip6b1"]
        )
        params["bydumpb1"] = beta_dump(
            params["betyip6b1"], params["alfyip6b1"]
        )

        params["dmuxkickb1_bds"] = tw1["mux", "mkd.o5l6.b1"]
        params["dmuxkickb1_bdsa"] = tw1["mux", "mkd.a5l6.b1"]
        params["dmuxkickb1_eds"] = (
            tw1["mux", "e.ds.r6.b1"] - tw1["mux", "mkd.o5l6.b1"]
        )
        params["tcdqmingapb1"] = tcdq_mingap(
            params["betxtcdqb1"], params["dxtcdqb1"]
        )
        params["tcdqgapb1"] = tcdq_gap(params["betxtcdqb1"])

        muxmkdb2 = tw2["mux", "mkd.h5r6.b2"]
        params["dmuxkickb2_tcsg"] = tw2["mux", "tcsp.a4l6.b2"] - muxmkdb2
        dmuxb2 = muxmkdb2 - np.array(
            [tw2["mux", f"tcdqa.{abc}4l6.b2"] for abc in "abc"]
        )
        for i, abc in enumerate("abc"):
            params[f"dmuxkickb2_tcdq{abc}"] = dmuxb2[i]
        params["dmuxkickb2"] = abs(dmuxb2 - 0.25).max()

        params["dxq5r6b2"] = tw2["dx", "mqy.5r6.b2"]
        params["dxq4l6b2"] = tw2["dx", "mqy.4l6.b2"]
        params["dxtcdqb2"] = tw2["dx", "tcdqa.a4l6.b2"]
        params["betxtcdqb2"] = tw2["betx", "tcdqa.a4l6.b2"]
        params["betytcdqb2"] = tw2["bety", "tcdqa.a4l6.b2"]
        params["betxtcdsb2"] = tw2["betx", "tcdsa.4r6.b2"]
        params["betytcdsb2"] = tw2["bety", "tcdsa.4r6.b2"]
        params["betxtcsgb2"] = tw2["betx", "tcsp.a4l6.b2"]
        params["betytcsgb2"] = tw2["bety", "tcsp.a4l6.b2"]
        params["betxmkdb2"] = tw2["betx", "mkd.h5r6.b2"]
        params["betymkdb2"] = tw2["bety", "mkd.h5r6.b2"]
        params["bxdumpb2"] = beta_dump(
            params["betxip6b2"], params["alfxip6b2"]
        )
        params["bydumpb2"] = beta_dump(
            params["betyip6b2"], params["alfyip6b2"]
        )

        params["dmuxkickb2_bds"] = tw2["mux", "mkd.o5r6.b2"]
        params["dmuxkickb2_bdsa"] = tw2["mux", "mkd.a5r6.b2"]
        params["dmuxkickb2_eds"] = (
            tw2["mux", "e.ds.r6.b2"] - tw2["mux", "mkd.o5r6.b2"]
        )
        params["tcdqmingapb2"] = tcdq_mingap(
            params["betxtcdqb2"], params["dxtcdqb2"]
        )
        params["tcdqgapb2"] = tcdq_gap(params["betxtcdqb2"])
        return params

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
            start="ip5",
            end=self.init_right[beam].element_name,
        )
        return tw

    def twiss_ats_ip(self, beam):
        line = self.model.sequence[beam]
        ir5 = self.parent.ir5
        rx = self.parent.params["rx_ip5"]
        ry = self.parent.params["ry_ip5"]
        tw = line.twiss(
            start="ip5",
            end=f"e.ds.r6.b{beam}",
            init_at="ip5",
            betx=ir5.params[f"betxip5b{beam}"] / rx,
            bety=ir5.params[f"betyip5b{beam}"] / ry,
        )
        return tw

    def matchb1(self, dkmin=0.01, dkmax=0.01, extra=True):
        lhc = self.parent.model.multiline

        targets = LHCIR.get_match_targets(
            self,
            b1=True,
            b2=False,
        )
        targets.extend(
            [
                xt.Target(
                    line="b1",
                    tar=dmuxkickb1,
                    value=0,
                    tol=0.012,  # 4.2 degrees
                    tag="ex_dmux_kickb1",
                ),
                xt.Target(
                    line="b1",
                    tar=betxdump,
                    value=xt.GreaterThan(3200),
                    tol=1,
                    tag="ex_betx_dump",
                ),
                xt.Target(
                    line="b1",
                    tar=betydump,
                    value=xt.GreaterThan(3200),  # 4000
                    tol=1,
                    tag="ex_bety_dump",
                ),
                xt.Target(
                    line="b1",
                    tar=betxydump,
                    value=xt.GreaterThan(4400),  # 4400
                    tol=1,
                    tag="ex_bet_dump",
                ),
                xt.Target(
                    line="b1",
                    tar="betx",
                    at="tcdqa.a4r6.b1",
                    value=490,
                    tol=20,
                    tag="ex_betx_tcdqa",
                ),
                xt.Target(
                    line="b1",
                    tar="bety",
                    at="tcdqa.a4r6.b1",
                    value=xt.GreaterThan(145),
                    tol=1,
                    tag="ex_bety_tcdqa",
                ),
                xt.Target(
                    line="b1",
                    tar="dx",
                    at="tcdqa.a4r6.b1",
                    value=0,
                    tol=0.7,
                    tag="ex_dx_tcdqa",
                ),
                xt.Target(
                    line="b1",
                    tar="bety",
                    at="tcdsa.4l6.b1",
                    value=xt.GreaterThan(145),  # 200
                    tol=10,
                    tag="ex_bety_tcdsa",
                ),
                xt.Target(
                    line="b1",
                    tar=dmuxkickb1_bds,
                    value=0,
                    tol=100,
                    tag="ex_dmuxkickb1_bds",
                ),
            ],
        )

        vary = LHCIR.get_match_vary(
            self,
            b1=True,
            b2=False,
            dkmin=dkmin,
            dkmax=dkmax,
        )

        match = lhc.match(
            solve=False,
            default_tol={None: 5e-8},
            solver_options=dict(max_rel_penalty_increase=2.0),
            lines=["b1"],
            start=[self.startb1],
            end=[self.endb1],
            init=[self.init_left[1]],
            targets=targets,
            vary=vary,
            check_limits=False,
            strengths=False,
        )
        if extra:
            match.disable(target="ip_.*")
        else:
            match.disable(target="ex_.*")

        match.disable(vary_name="kq4.l6b1")
        return match

    def matchb2(self, dkmin=0.01, dkmax=0.01, extra=True):
        lhc = self.parent.model.multiline

        targets = LHCIR.get_match_targets(
            self,
            b1=False,
            b2=True,
        )
        targets.extend(
            [
                xt.Target(
                    line="b2",
                    tar=dmuxkickb2,
                    value=0,
                    tol=0.0123,  # 4.4 degrees
                    tag="ex_dmux_kickb2",
                ),
                xt.Target(
                    line="b2",
                    tar=betxdump,
                    value=xt.GreaterThan(3200),
                    tol=1,
                    tag="ex_betx_dump",
                ),
                xt.Target(
                    line="b2",
                    tar=betydump,
                    value=xt.GreaterThan(3200),  # 4000
                    tol=1,
                    tag="ex_bety_dump",
                ),
                xt.Target(
                    line="b2",
                    tar=betxydump,
                    value=xt.GreaterThan(4400),  # 4400
                    tol=1,
                    tag="ex_bet_dump",
                ),
                xt.Target(
                    line="b2",
                    tar="betx",
                    at="tcdqa.a4l6.b2",
                    value=490,
                    tol=20,
                    tag="ex_betx_tcdqa",
                ),
                xt.Target(
                    line="b2",
                    tar="bety",
                    at="tcdqa.a4l6.b2",
                    value=xt.GreaterThan(145),
                    tol=1,
                    tag="ex_bety_tcdqa",
                ),
                xt.Target(
                    line="b2",
                    tar="dx",
                    at="tcdqa.a4l6.b2",
                    value=0,
                    tol=0.7,
                    tag="ex_dx_tcdqa",
                ),
                xt.Target(
                    line="b2",
                    tar="bety",
                    at="tcdsa.4r6.b2",
                    value=xt.GreaterThan(145),  # 200
                    tol=10,
                    tag="ex_bety_tcdsa",
                ),
                xt.Target(
                    line="b2",
                    tar=dmuxkickb2_bds,
                    value=0,
                    tol=100,
                    tag="ex_dmuxkickb2_bds",
                ),
            ]
        )

        vary = LHCIR.get_match_vary(
            self,
            b1=False,
            b2=True,
            dkmin=dkmin,
            dkmax=dkmax,
        )

        match = lhc.match(
            solve=False,
            default_tol={None: 5e-8},
            solver_options=dict(max_rel_penalty_increase=2.0),
            lines=["b2"],
            start=[self.startb2],
            end=[self.endb2],
            init=[self.init_left[2]],
            targets=targets,
            vary=vary,
            check_limits=False,
            strengths=False,
        )

        if extra:
            match.disable(target="ip_.*")
        else:
            match.disable(target="ex_.*")

        match.disable(vary_name="kq4.r6b2")

        return match
