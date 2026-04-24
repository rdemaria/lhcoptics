from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import xdeps as xd

from .knob import Knob
from .utils import print_diff_dict_float

def get_ap_limit(section, ap, verbose=True):
    energy = ap.energy
    if energy < 500e9:
        limit = 12.6
    else:
        limit = 14.6
    n1 = ap.n1.copy()
    n1[ap.name == "tcdqm.b4l6.b1:1"] *= 1.2
    n1[ap.name == "tcdqm.a4l6.b1:1"] *= 1.2
    n1[ap.name == "tcdqm.a4r6.b1:1"] *= 1.2
    n1[ap.name == "tcdqm.b4r6.b1:1"] *= 1.2
    n1[ap.name == "tcdqm.b4l6.b2:1"] *= 1.1
    n1[ap.name == "tcdqm.a4l6.b2:1"] *= 1.1
    n1[ap.name == "tcdqm.a4r6.b2:1"] *= 1.1
    n1[ap.name == "tcdqm.b4r6.b2:1"] *= 1.1
    minap = np.argmin(n1)
    n1min = n1[minap]
    smin = ap.s[minap]
    namemin = ap.name[minap]
    name2min = namemin.lower()[:-2]
    if verbose:
        reset = "\033[0m"
        if n1min < limit:
            color = "\033[91m"  # red
        else:
            color = "\033[92m"  # green
        print(
            f"{section}: limit {name2min:20} {color}{n1min:9.3f}{reset} σ at s={smin:.1f} m"
        )

class LHCMadxModel:
    extra_defs = """
kd1.lr1       :=  ad1.lr1/l.mbxw;
kd2.l1        :=  ad2.l1/l.mbrc ;
kd2.r1        :=  ad2.r1/l.mbrc ;
kd1.l2        :=  ad1.l2/l.mbx  ;
kd1.r2        :=  ad1.r2/l.mbx  ;
kd2.l2        :=  ad2.l2/l.mbrc ;
kd2.r2        :=  ad2.r2/l.mbrc ;
kd3.lr3       :=  ad3.lr3/l.mbw ;
kd4.lr3       :=  ad4.lr3/l.mbw ;
kd3.l4        :=  ad3.l4/l.mbrs ;
kd3.r4        :=  ad3.r4/l.mbrs ;
kd4.l4        :=  ad4.l4/l.mbrb ;
kd4.r4        :=  ad4.r4/l.mbrb ;
kd34.lr3      :=  ad3.lr3/l.mbw ;
kd34.lr7      :=  ad3.lr7/l.mbw ;
kd1.lr5       :=  ad1.lr5/l.mbxw;
kd2.l5        :=  ad2.l5/l.mbrc ;
kd2.r5        :=  ad2.r5/l.mbrc ;
kd3.lr7       :=  ad3.lr7/l.mbw ;
kd4.lr7       :=  ad4.lr7/l.mbw ;
kd1.l8        :=  ad1.l8/l.mbx  ;
kd1.r8        :=  ad1.r8/l.mbx  ;
kd2.l8        :=  ad2.l8/l.mbrc ;
kd2.r8        :=  ad2.r8/l.mbrc ;
ksumd2.l1b2   :=  kd2.l1        ;
ksumd2.l2b2   :=  kd2.l2        ;
ksumd2.l5b2   :=  kd2.l5        ;
ksumd2.l8b2   :=  kd2.l8        ;
ksumd2.r1b2   :=  kd2.l1        ;
ksumd2.r2b2   :=  kd2.l2        ;
ksumd2.r5b2   :=  kd2.l5        ;
ksumd2.r8b2   :=  kd2.l8        ;

kb.a12        :=  ab.a12/l.mb   ;
kb.a23        :=  ab.a23/l.mb   ;
kb.a34        :=  ab.a34/l.mb   ;
kb.a45        :=  ab.a45/l.mb   ;
kb.a56        :=  ab.a56/l.mb   ;
kb.a67        :=  ab.a67/l.mb   ;
kb.a78        :=  ab.a78/l.mb   ;
kb.a81        :=  ab.a81/l.mb   ;

abas:= 12.00/ 6.0*clight/(7E12)*on_sol_atlas;
abls:= 6.05/12.1*clight/(7E12)*on_sol_alice ;
abcs:= 52.00/13.0*clight/(7E12)*on_sol_cms  ;
abxwt.l2      := -0.0000772587268993839836*on_alice ;
abwmd.l2      := +0.0001472587268993839840*on_alice ;
abaw.r2       := -0.0001335474860334838000*on_alice ;
abxwt.r2      := +0.0000635474860334838004*on_alice ;
abxws.l8      := -0.000045681598453109894*on_lhcb   ;
abxwh.l8      := +0.000180681598453109894*on_lhcb   ;
ablw.r8       := -0.000180681598453109894*on_lhcb   ;
abxws.r8      := +0.000045681598453109894*on_lhcb   ;
    """
    extra_defs_hllhc = """
! Solenoids and spectrometers
abas               := 12.00/6.0*clight/(7e12)*on_sol_atlas ;
abls               := 6.05/12.1*clight/(7e12)*on_sol_alice ;
abcs               := 52.00/13.0*clight/(7e12)*on_sol_cms ;
abxwt.l2           := -0.0000772587268993839836*on_alice ;
abwmd.l2           := +0.0001472587268993839840*on_alice ;
abaw.r2            := -0.0001335474860334838000*on_alice ;
abxwt.r2           := +0.0000635474860334838004*on_alice ;
abxws.l8           := -0.000045681598453109894*on_lhcb ;
abxwh.l8           := +0.000180681598453109894*on_lhcb ;
ablw.r8            := -0.000180681598453109894*on_lhcb ;
abxws.r8           := +0.000045681598453109894*on_lhcb ;

! Triplet helper
kqx.l1             := kqx2a.l1           ;
ktqx1.l1           := kqx1.l1-kqx2a.l1   ;
ktqx3.l1           := kqx3.l1-kqx2a.l1   ;
kqx.r1             := kqx2a.r1           ;
ktqx1.r1           := kqx1.r1-kqx2a.r1   ;
ktqx3.r1           := kqx3.r1-kqx2a.r1   ;

kqx.l5             := kqx2a.l5           ;
ktqx1.l5           := kqx1.l5-kqx2a.l5   ;
ktqx3.l5           := kqx3.l5-kqx2a.l5   ;
kqx.r5             := kqx2a.r5           ;
ktqx1.r5           := kqx1.r5-kqx2a.r5   ;
ktqx3.r5           := kqx3.r5-kqx2a.r5   ;

! Main dipole strength
kb.a12             := ab.a12/l.mb        ;
kb.a23             := ab.a23/l.mb        ;
kb.a34             := ab.a34/l.mb        ;
kb.a45             := ab.a45/l.mb        ;
kb.a56             := ab.a56/l.mb        ;
kb.a67             := ab.a67/l.mb        ;
kb.a78             := ab.a78/l.mb        ;
kb.a81             := ab.a81/l.mb        ;

! Nominal D1-4 strengths
kd1.l1             := ad1.l1/l.mbxf      ;
kd1.r1             := ad1.r1/l.mbxf      ;
kd2.l1             := ad2.l1/l.mbrd      ;
kd2.r1             := ad2.r1/l.mbrd      ;
kd1.l2             := ad1.l2/l.mbx       ;
kd1.r2             := ad1.r2/l.mbx       ;
kd2.l2             := ad2.l2/l.mbrc      ;
kd2.r2             := ad2.r2/l.mbrc      ;
kd3.lr3            := ad3.lr3/l.mbw      ;
kd4.lr3            := ad4.lr3/l.mbw      ;
kd3.l4             := ad3.l4/l.mbrs      ;
kd3.r4             := ad3.r4/l.mbrs      ;
kd4.l4             := ad4.l4/l.mbrb      ;
kd4.r4             := ad4.r4/l.mbrb      ;
kd34.lr3           := ad3.lr3/l.mbw      ;
kd1.l5             := ad1.l5/l.mbxf      ;
kd1.r5             := ad1.r5/l.mbxf      ;
kd2.l5             := ad2.l5/l.mbrd      ;
kd2.r5             := ad2.r5/l.mbrd      ;
kd3.lr7            := ad3.lr7/l.mbw      ;
kd4.lr7            := ad4.lr7/l.mbw      ;
kd34.lr7           := ad3.lr7/l.mbw      ;
kd1.l8             := ad1.l8/l.mbx       ;
kd1.r8             := ad1.r8/l.mbx       ;
kd2.l8             := ad2.l8/l.mbrc      ;
kd2.r8             := ad2.r8/l.mbrc      ;
ksumd2.l1b2        := kd2.l1             ;
ksumd2.l2b2        := kd2.l2             ;
ksumd2.l5b2        := kd2.l5             ;
ksumd2.l8b2        := kd2.l8             ;
ksumd2.r1b2        := kd2.l1             ;
ksumd2.r2b2        := kd2.l2             ;
ksumd2.r5b2        := kd2.l5             ;
ksumd2.r8b2        := kd2.l8             ;



! Optimal D1 and D2 strengths
! D1 and D2 angles [rad]
ad1.l1 = 1.5009443796e-03;
ad1.l2 = 1.5325070050e-03;
ad1.l5 = 1.5009443796e-03;
ad1.l8 = 1.5325070050e-03;
ad1.r1 = 1.5009443796e-03;
ad1.r2 = 1.5325070050e-03;
ad1.r5 = 1.5009443796e-03;
ad1.r8 = 1.5325070050e-03;
ad2.l1 = 1.5009443796e-03;
ad2.l2 = 1.5325070050e-03;
ad2.l5 = 1.5009443796e-03;
ad2.l8 = 1.5325070050e-03;
ad2.r1 = 1.5009443796e-03;
ad2.r2 = 1.5325070050e-03;
ad2.r5 = 1.5009443796e-03;
ad2.r8 = 1.5325070050e-03;

! D1 mid-separation [m]
sep_mid_d1.l1 = 2.3527297632e-03;
sep_mid_d1.l2 = 3.6205469140e-03;
sep_mid_d1.l5 = 2.3527297632e-03;
sep_mid_d1.l8 = 3.6205469140e-03;
sep_mid_d1.r1 = 2.3527297632e-03;
sep_mid_d1.r2 = 3.6205469140e-03;
sep_mid_d1.r5 = 2.3527297632e-03;
sep_mid_d1.r8 = 3.6205469140e-03;

! D2 shifts [m]
shift_d2.l1 = 1.5407071706e-03;
shift_d2.l2 = 1.1897265430e-03;
shift_d2.l5 = 1.5407071706e-03;
shift_d2.l8 = 1.1897265430e-03;
shift_d2.r1 = 1.5407071706e-03;
shift_d2.r2 = 1.1897265430e-03;
shift_d2.r5 = 1.5407071706e-03;
shift_d2.r8 = 1.1897265430e-03;! Angles for RBends in IR3
abw.a6l3 = -1.8872909366e-04;
abw.a6r3 = -1.8872909366e-04;
abw.b6l3 = -1.8872910038e-04;
abw.b6r3 = -1.8872910038e-04;
abw.c6l3 = -1.8872911382e-04;
abw.c6r3 = -1.8872911382e-04;
abw.d6l3 = 1.8872911382e-04;
abw.d6r3 = 1.8872911382e-04;
abw.e6l3 = 1.8872910038e-04;
abw.e6r3 = 1.8872910038e-04;
abw.f6l3 = 1.8872909366e-04;
abw.f6r3 = 1.8872909366e-04;

! In-out angle differences for RBends in IR3
adiff.bw.a6l3 = 1.8872909366e-04;
adiff.bw.a6r3 = -1.8872909366e-04;
adiff.bw.b6l3 = 5.6618728769e-04;
adiff.bw.b6r3 = -5.6618728769e-04;
adiff.bw.c6l3 = 9.4364550190e-04;
adiff.bw.c6r3 = -9.4364550190e-04;
adiff.bw.d6l3 = 9.4364550190e-04;
adiff.bw.d6r3 = -9.4364550190e-04;
adiff.bw.e6l3 = 5.6618728769e-04;
adiff.bw.e6r3 = -5.6618728769e-04;
adiff.bw.f6l3 = 1.8872909366e-04;
adiff.bw.f6r3 = -1.8872909366e-04;

! Shifts for RBends in IR3
shift.bw.a6l3 = 1.6919790135e-02;
shift.bw.a6r3 = 1.6919790135e-02;
shift.bw.b6l3 = 1.6120522415e-02;
shift.bw.b6r3 = 1.6120522415e-02;
shift.bw.c6l3 = 1.4521986906e-02;
shift.bw.c6r3 = 1.4521986906e-02;
shift.bw.d6l3 = 4.4780130939e-03;
shift.bw.d6r3 = 4.4780130939e-03;
shift.bw.e6l3 = 2.8794775851e-03;
shift.bw.e6r3 = 2.8794775851e-03;
shift.bw.f6l3 = 2.0802098651e-03;
shift.bw.f6r3 = 2.0802098651e-03;! IR4 RBend angles
ad3.l4 = 1.5716058651e-03;
ad3.r4 = 1.5716058651e-03;
ad4.l4 = 1.5716046865e-03;
ad4.r4 = 1.5716046865e-03;

! IR4 RBend shifts
shift.mbrb.l4 = 1.8564631277e-03;
shift.mbrb.r4 = 1.8564631277e-03;
shift.mbrs.l4 = 1.1435410493e-03;
shift.mbrs.r4 = 1.1435410493e-03;! Angles for RBends in IR7
abw.a6l7 = 1.8872909041e-04;
abw.a6r7 = 1.8872909041e-04;
abw.b6l7 = 1.8872909713e-04;
abw.b6r7 = 1.8872909713e-04;
abw.c6l7 = -1.8872909713e-04;
abw.c6r7 = -1.8872909713e-04;
abw.d6l7 = -1.8872909041e-04;
abw.d6r7 = -1.8872909041e-04;

! In-out angle differences for RBends in IR7
adiff.bw.a6l7 = -1.8872909041e-04;
adiff.bw.a6r7 = 1.8872909041e-04;
adiff.bw.b6l7 = -5.6618727795e-04;
adiff.bw.b6r7 = 5.6618727795e-04;
adiff.bw.c6l7 = -5.6618727795e-04;
adiff.bw.c6r7 = 5.6618727795e-04;
adiff.bw.d6l7 = -1.8872909041e-04;
adiff.bw.d6r7 = 1.8872909041e-04;

! Shifts for RBends in IR7
shift.bw.a6l7 = 1.6919790136e-02;
shift.bw.a6r7 = 1.6919790136e-02;
shift.bw.b6l7 = 1.5988412068e-02;
shift.bw.b6r7 = 1.5988412068e-02;
shift.bw.c6l7 = 2.8794781954e-03;
shift.bw.c6r7 = 2.8794781954e-03;
shift.bw.d6l7 = 2.0802104882e-03;
shift.bw.d6r7 = 2.0802104882e-03;

! Sep dipole strength corrections
kd1.l1 = 0.00023938497863369427;
kd2.l1 = 0.0001929729771192161;
kd1.r1 = 0.00023938497863369422;
kd2.r1 = 0.00019297297711921607;
kd1.l2 = 0.00016216998996044585;
kd2.l2 = 0.00016216998996044585;
kd1.r2 = 0.00016216998996044585;
kd2.r2 = 0.00016216998996044585;
kd1.l5 = 0.00023938497863369427;
kd2.l5 = 0.0001929729771192161;
kd1.r5 = 0.00023938497863369422;
kd2.r5 = 0.00019297297711921607;
kd1.l8 = 0.00016216998996044585;
kd2.l8 = 0.00016216998996044585;
kd1.r8 = 0.00016216998996044585;
kd2.r8 = 0.00016216998996044585;
kd34.lr3 = 5.5508556629e-05;
e_kd3.l3 = -2.3380076269952575e-15;
e_kd4.l3 = -2.3379970162010004e-15;
e_kd3.r3 = -2.338008321555881e-15;
e_kd4.r3 = -2.3380189323500575e-15;
kd34.lr7 = 5.5508555673e-05;
e_kd3.l7 = -2.6314720697379465e-15;
e_kd4.l7 = -2.6314680732113866e-15;
e_kd3.r7 = -2.6314486486963916e-15;
e_kd4.r7 = -2.631452645222881e-15;
kd3.l4 = 0.0001663074304915815;
kd4.l4 = 0.00016630730577215885;
kd3.r4 = 0.00016630743049158152;
kd4.r4 = 0.00016630730577215882;


! Main dipole strength corrections
kb.a12             := ab.a12/l.mb*(1+6.9e-13)      ;
kb.a23             := ab.a23/l.mb*(1+6.9e-13)      ;
kb.a34             := ab.a34/l.mb*(1+6.9e-13)      ;
kb.a45             := ab.a45/l.mb*(1+6.9e-13)      ;
kb.a56             := ab.a56/l.mb*(1+6.9e-13)      ;
kb.a67             := ab.a67/l.mb*(1+6.9e-13)      ;
kb.a78             := ab.a78/l.mb*(1+6.9e-13)      ;
kb.a81             := ab.a81/l.mb*(1+6.9e-13)      ;
"""
    extra_madx = (
        extra_defs
        + """
beam, particle=proton, energy=6.8e12, sequence=lhcb1;
beam, particle=proton, energy=6.8e12, sequence=lhcb2, bv=-1;
use, sequence=lhcb1;
use, sequence=lhcb2;
    """
    )

    @staticmethod
    def knobs_to_expr(knobs, strengths=None):
        """
        Print knobs as madx expressions.

        knobs: list of knobs
        strengths: dictionary of strengths to be used as base values
        """
        if strengths is None:
            strengths = {}
        weights = {}
        for knob in knobs:
            for st, val in knob.weights.items():
                weights.setdefault(st, []).append(f"{val:+.15g} * {knob.name}")
        for st in weights:
            basevalue = strengths.get(st)
            if basevalue is not None:
                weights[st].insert(0, f"{basevalue:+.15g}")
        out = []
        for knob in knobs:
            out.append(f"{knob.name} = {knob.value:.15g};")
        for st in weights:
            rhs = "\n  ".join(weights[st])
            out.append(f"{st} :=\n  {rhs};")
        return out

    def __init__(self, madx):
        self.madx = madx
        self.b1 = MADSequence(self.madx, "lhcb1")
        self.b2 = MADSequence(self.madx, "lhcb2")

    @classmethod
    def from_madx_scripts(cls, *madxfiles, extra=True, stdout=False, basedir=None):
        from cpymad.madx import Madx

        madx = Madx(stdout=stdout)
        if basedir is not None:
            madx.chdir(str(basedir))
        madx.options(echo=False, warn=False, info=False)
        for madxfile in madxfiles:
            madx.call(str(madxfile))
        if extra:
            madx.input(cls.extra_madx)
        return cls(madx)

    def __contains__(self, key):
        return key in self.madx.globals

    def __getitem__(self, key):
        return self.madx.globals[key]

    def __setitem__(self, key, value):
        self.madx.globals[key] = value

    def call(self, filename):
        self.madx.call(filename)
        return self

    def diff(self, other):
        selfvar = {k: v for k, v in self.madx.globals.items() if isinstance(v, float)}
        othervar = {k: v for k, v in other.madx.globals.items() if isinstance(v, float)}
        print_diff_dict_float(selfvar, othervar)

    def filter(self, pattern):
        return [k for k in self.madx.globals if re.match(pattern, k)]

    def get_ap_arc(self, arc, beam):
        Path("temp").mkdir(exist_ok=True)
        self.madx.exec(f"mk_aparc({arc},b{beam})")
        tab = xd.Table(self.madx.table.aperture)
        tab._data.update(self.madx.table.aperture.summary)
        return tab

    def get_ap_ir(self, irn, beam, verbose=True):
        Path("temp").mkdir(exist_ok=True)
        self.madx.exec(f"mk_apir({irn},b{beam})")
        tab = xd.Table(self.madx.table.aperture)
        tab._data.update(self.madx.table.aperture.summary)
        if verbose:
            get_ap_limit(f"IR{irn} B{beam}", tab, verbose=True)
        return tab

    def get_ap_irs(self, verbose=True):
        out = {}
        for ir in range(1, 9):
            for beam in [1, 2]:
                tab = self.get_ap_ir(ir, beam)
                out[f"ir{ir}b{beam}"] = tab

        if verbose:
            for kk, tab in out.items():
                get_ap_limit(kk, tab, verbose=verbose)
        return out

    def get_p0c(self):
        return self.madx.sequence.lhcb1.beam.pc

    def get_variant(self):
        vv = self.madx.globals
        if "acbrdh4.l1b1" in vv:
            variant = "hl"
        elif vv["kqx.l1"] > 0:
            variant = "lhc2024"
        elif vv["kqx.l5"] > 0:
            variant = "lhc2025"
        else:
            variant = "lhc"
        return variant

    def is_full(self):
        return "lhcb1" in self.madx.sequence and "lhcb2" in self.madx.sequence

    def mad_find_and_set0_knobs(madx, strengths, variant=None):
        """
        Find knobs that are used to set the strengths and set knob value to 0
        """
        defs = {}
        for st in strengths:
            if expr := madx.globals.cmdpar[st].expr:
                for knob in re.findall("on[A-z_0-9.]*", expr):
                    if len(knob) < 12:
                        defs.setdefault(knob, []).append(st)
        knobs = {}
        for knob, strengths in defs.items():
            value = madx.globals[knob]
            madx.globals[knob] = 0
            base = {st: madx.globals[st] for st in strengths}
            weights = {}
            madx.globals[knob] = 1
            for s in strengths:
                dvalue = madx.globals[s] - base[s]
                weights[s] = dvalue
            madx.globals[knob] = 0
            knobs[knob] = Knob(knob, value, weights, variant=variant).specialize()
        return knobs

    def mad_mkknob(self, name, variant=None):
        madx = self.madx
        base = dict(madx.globals)
        value = madx.globals.get(name, 0)
        dvalue = value + 1
        weights = {}
        for s in base:
            dvalue = madx.globals[s] - base[s]
            if dvalue != 0:
                weights[s] = dvalue
        madx.globals[name] = value
        return Knob(name, value, weights, variant=variant).specialize()

    def make_and_set0_knobs(self, knob_names, variant=None):
        """
        for each knob name in knob_names
           look for all expressions containing the knob name
           compute the weights by setting the knob to 1 and 0
           set the knob to 0
           return knob object builit with name, knob value and weights
        """
        madx = self.madx
        defs = {}
        for knob in knob_names:
            for st in madx.globals:
                if expr := madx.globals.cmdpar[st].expr:
                    if knob in expr:
                        defs.setdefault(knob, []).append(st)
        knobs = {}
        for knob, strengths in defs.items():
            value = madx.globals[knob]
            madx.globals[knob] = 0
            base = dict(madx.globals)
            weights = {}
            madx.globals[knob] = 1
            for s in strengths:
                dvalue = madx.globals[s] - base[s]
                weights[s] = dvalue
            madx.globals[knob] = 0
            knobs[knob] = Knob(knob, value, weights, variant=variant).specialize()
        return knobs

    def plot(self, beam=None, left="betx bety", right="dx"):
        ylabels = {
            "betx": r"$\beta$ [m]",
            "bety": r"$\beta$ [m]",
            "mux": r"$\mu$ [2π]",
            "muy": r"$\mu$ [2π]",
            "x": r"$x$ [m]",
            "y": r"$y$ [m]",
            "dx": r"$D$ [m]",
            "dy": r"$D$ [m]",
        }
        if beam is None:
            self.plot(beam=1, left=left, right=right)
            self.plot(beam=2, left=left, right=right)
        else:
            tw = self.twiss(sequence=beam)
            fig, ax = plt.subplots(num=f"MAD lhcb{beam}", clear=True)
            lines = []
            for col in left.split():
                (ln,) = ax.plot(tw.s, getattr(tw, col), label=col)
                lines.append(ln)
                ax.set_ylabel(ylabels.get(col, col))
            ax2 = ax.twinx()
            for col in right.split():
                (ln,) = ax2.plot(
                    tw.s, getattr(tw, col), label=col, color=f"C{len(lines)}"
                )
                lines.append(ln)
                ax2.set_ylabel(ylabels.get(col, col))
            ax.legend(handles=lines)
            ax.set_xlabel("s [m]")
            ax.set_title(f"MAD LHC Optics - Beam {beam}")

    def plot_ap_ir(self, irn, beam):
        apb = self.get_ap_ir(irn, beam)
        fig, ax = plt.subplots(num=f"MAD IR{irn} B{beam} aperture", clear=True)
        n1 = apb.n1
        n1[(n1 > 30) | (n1 == 0)] = 30
        ax.plot(apb.s, n1, "b", label="n1")
        ax.set_xlabel("s [m]")
        ax.set_ylabel("$n_1$ [σ]")
        ax.legend()
        idx = np.argmin(apb.n1)
        ax.axvline(apb.s[idx], color="k", ls="--")

    def plot_ap_orbit(self, irn):
        apb1 = self.get_ap_ir(irn, 1)
        apb2 = self.get_ap_ir(irn, 2)
        fig, ax = plt.subplots(2, 1, num=f"MAD IR{irn} orbit", clear=True)
        ax[0].plot(apb1.s, apb1.x, "b", label="x B1")
        ax[0].plot(apb2.s, apb2.x, "r", label="x B2")
        ax[0].set_xlabel("s [m]")
        ax[0].set_ylabel("x [m]")
        ax[0].legend()
        idx = np.argmin(apb1.n1)
        ax[0].axvline(apb1.s[idx], color="k", ls="--")
        ax[1].plot(apb1.s, apb1.y, "b", label="y B1")
        ax[1].plot(apb2.s, apb2.y, "r", label="y B2")
        ax[1].set_xlabel("s [m]")
        ax[1].set_ylabel("y [m]")
        ax[1].legend()
        idx = np.argmin(apb2.n1)
        ax[1].axvline(apb2.s[idx], color="k", ls="--")

    def plot_beta(self):
        self.plot(left="betx bety", right="dx")

    def plot_orbit(self):
        self.plot(left="x y", right="dx dy")

    def print_ip(self):
        tw1, tw2 = self.twiss()
        print(f"{'':7}", end="")
        for ir in [1, 2, 5, 8]:
            print(f"    IR{ir}B1     IR{ir}B2", end="")
        print()
        for cc, scale in zip("betx bety x y px py".split(), [1, 1, 1e3, 1e3, 1e6, 1e6]):
            print(f"{cc:7}", end="")
            for ir in [1, 2, 5, 8]:
                ipn = f"ip{ir}:1"
                print(
                    f"{tw1[cc, ipn] * scale:9.3f} {tw2[cc, ipn] * scale:9.3f}", end=""
                )
            print()

    def twiss(self, beam=None, chrom=True, **kwargs):
        if beam is None:
            return (self.b1.twiss(chrom=chrom, **kwargs), self.b2.twiss(chrom=chrom, **kwargs))
        else:
            return getattr(self, f"b{beam}").twiss(chrom=chrom, **kwargs)

    def update(self, src, knobs_check=True):
        if hasattr(src, "strengths"):
            self.update_vars(src.strengths)
        else:
            self.update_vars(src)
        if hasattr(src, "knobs"):
            self.update_knobs(src.knobs, knobs_check=knobs_check)

    def update_knobs(self, knobs, verbose=False, knobs_off=False, knobs_check=False):
        for k, knob in knobs.items():
            if not knobs_off:
                if verbose:
                    print(f"{k:20} {self[k]:15.6g} -> {knob.value:15.6g}")
                self[k] = knob.value
            for wn, value in knob.weights.items():
                name = f"{wn}_from_{k}"
                self[name] = value
                if expr := self.madx.globals.cmdpar[wn].expr:
                    if expr is not None and k in expr and name not in expr:
                        raise ValueError(f"{wn} depends on {k} but not on {name}")
                if expr is None:
                    wnvalue = self[wn]
                    self.madx.input(f"{wn} := {wnvalue} + ({name} * {k});")
                elif expr is not None and k not in expr:
                    self.madx.input(f"{wn} := {expr} + ({name} * {k});")

    def update_vars(self, strengths, verbose=False):
        for k, v in strengths.items():
            if verbose:
                print(f"{k:20} {self[k]:15.6g} -> {v:15.6g}")
            self[k] = v

class MADSequence:
    def __init__(self, madx, sequence_name):
        self.madx = madx
        self.sequence = sequence_name

    @property
    def end(self):
        self.madx_sequence.elements[-2].name

    @property
    def madx_sequence(self):
        return self.madx.sequence[self.sequence]

    @property
    def start(self):
        self.madx_sequence.elements[1].name

    def rmatrix(self, start=None, end=None):
        if start is None:
            start = self.start
        if end is None:
            end = self.end
        self.madx.use(sequence=self.sequence, range=f"{start}/{end}")
        tw = self.madx.twiss(betx=1, bety=1, rmatrix=True)
        return RMatrix(tw.getmat("re", -1, 6, 6))

    def twiss(
        self,
        start=None,
        end=None,
        betx=None,
        bety=None,
        alfx=None,
        alfy=None,
        dx=None,
        dpx=None,
        dy=None,
        dpy=None,
        init=None,
        s=0,
        at=None,
        full=True,
        chrom=True,
    ):
        """
        Parameters:
        - full: if False consider only the portion from start to end
        - at: location to which initial conditions are to be set
        - init, betx, .... : initial conditions if all are None, then they are taken from the periodic solution at `at`
        - start, end: start and end of the twiss output
        - chrom: if True, compute chromatic properties
        """
        if set([betx, bety, alfx, alfy, dx, dpx, dy, dpy, init]) == {
            None
        }:  # is periodic
            self.madx.use(sequence=self.sequence, range=f"{start}/{end}")
            tw = self.madx.twiss(sequence=self.sequence)
        if full is False:  # consider only the machine from start to end
            if start is None:
                start = self.madx.sequence
            if end is None:
                end = self.madx.globals[self.sequence].sequence.end
        else:
            tw = self.madx.twiss(
                betx=betx,
                bety=bety,
                alfx=alfx,
                alfy=alfy,
                sequence=self.sequence,
            )
        tab=xd.Table(tw)
        return tab

    def twiss_line(self, betx=1, bety=1, alfx=0, alfy=0):
        tw = self.madx.twiss(
            betx=betx, bety=bety, alfx=alfx, alfy=alfy, sequence=self.sequence
        )
        return xd.Table(tw)

    def twiss_line_back(self, start, end, betx=1, bety=1, alfx=0, alfy=0, dx=0, dpx=0):
        self.madx.use(sequence=self.sequence, range=f"{start}/{end}")
        tw = self.madx.twiss(betx=1, bety=1, rmatrix=True)
        rmat = tw.getmat("re", -1, 6, 6)
        r11 = rmat[0, 0]
        r12 = rmat[0, 1]
        r21 = rmat[1, 0]
        r22 = rmat[1, 1]
        r33 = rmat[2, 2]
        r34 = rmat[2, 3]
        r43 = rmat[3, 2]
        r44 = rmat[3, 3]
        r16 = rmat[0, 5]
        r26 = rmat[1, 5]
        bx2 = betx
        by2 = bety
        ax2 = alfx
        ay2 = alfy
        dx2 = dx
        dpx2 = dpx
        gx2 = (1 + ax2**2) / bx2
        gy2 = (1 + ay2**2) / by2
        bx1 = r22**2 * bx2 + 2 * r22 * r12 * ax2 + r12**2 * gx2
        ax1 = r22 * r21 * bx2 + (r11 * r22 + r12 * r21) * ax2 + r12 * r11 * gx2
        by1 = r44**2 * by2 + 2 * r44 * r34 * ay2 + r34**2 * gy2
        ay1 = r44 * r43 * by2 + (r33 * r44 + r34 * r43) * ay2 + r34 * r33 * gy2
        dx1 = r22 * (dx2 - r16) - r12 * (dpx2 - r26)
        dpx1 = -r21 * (dx2 - r16) + r11 * (dpx2 - r26)
        tw = self.madx.twiss(betx=bx1, bety=by1, alfx=ax1, alfy=ay1, dx=dx1, dpx=dpx1)
        return tw

    def twiss_periodic(self, use=False):
        if use:
            self.madx.use(sequence=self.sequence)
        tw = self.madx.twiss(sequence=self.sequence)
        return xd.Table(tw)

class RMatrix:
    def __init__(self, matrix):
        self.matrix = matrix

    @classmethod
    def from_twiss(cls, tw):
        pass

class TwissInit:
    def __init__(
        self,
        betx=1,
        alfx=1,
        mux=0,
        bety=1,
        alfy=0,
        muy=0,
        x=0,
        px=0,
        y=0,
        py=0,
        t=0,
        pt=0,
        dx=0,
        dpx=0,
        dy=0,
        dpy=0,
        wx=0,
        phix=0,
        dmux=0,
        wy=0,
        phiy=0,
        dmuy=0,
        ddx=0,
        ddpx=0,
        ddy=0,
        ddpy=0,
        r11=0,
        r12=0,
        r21=0,
        r22=0,
    ):
        self.betx = betx
        self.alfx = alfx
        self.mux = mux
        self.bety = bety
        self.alfy = alfy
        self.muy = muy
        self.x = x
        self.px = px
        self.y = y
        self.py = py
        self.t = t
        self.pt = pt
        self.dx = dx
        self.dpx = dpx
        self.dy = dy
        self.dpy = dpy
        self.wx = wx
        self.phix = phix
        self.dmux = dmux
        self.wy = wy
        self.phiy = phiy
        self.dmuy = dmuy
        self.ddx = ddx
        self.ddpx = ddpx
        self.ddy = ddy
        self.ddpy = ddpy
        self.r11 = r11
        self.r12 = r12
        self.r21 = r21
        self.r22 = r22
