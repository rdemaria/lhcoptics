"""Xsuite-backed LHC optics model helpers."""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xdeps
import xtrack as xt

from .knob import Knob
from .model_madx import LHCMadxModel
from .utils import match_compare_log

rbend_data = {
    # D1 and D2 angles [rad]
    "ad1.l1": 1.5009443796e-03,
    "ad1.l2": 1.5325070050e-03,
    "ad1.l5": 1.5009443796e-03,
    "ad1.l8": 1.5325070050e-03,
    "ad1.r1": 1.5009443796e-03,
    "ad1.r2": 1.5325070050e-03,
    "ad1.r5": 1.5009443796e-03,
    "ad1.r8": 1.5325070050e-03,
    "ad2.l1": 1.5009443796e-03,
    "ad2.l2": 1.5325070050e-03,
    "ad2.l5": 1.5009443796e-03,
    "ad2.l8": 1.5325070050e-03,
    "ad2.r1": 1.5009443796e-03,
    "ad2.r2": 1.5325070050e-03,
    "ad2.r5": 1.5009443796e-03,
    "ad2.r8": 1.5325070050e-03,
    # D1 mid-separation [m]
    "sep_mid_d1.l1": 2.3527297632e-03,
    "sep_mid_d1.l2": 3.6205469140e-03,
    "sep_mid_d1.l5": 2.3527297632e-03,
    "sep_mid_d1.l8": 3.6205469140e-03,
    "sep_mid_d1.r1": 2.3527297632e-03,
    "sep_mid_d1.r2": 3.6205469140e-03,
    "sep_mid_d1.r5": 2.3527297632e-03,
    "sep_mid_d1.r8": 3.6205469140e-03,
    # D2 shifts [m]
    "shift_d2.l1": 1.5407071706e-03,
    "shift_d2.l2": 1.1897265430e-03,
    "shift_d2.l5": 1.5407071706e-03,
    "shift_d2.l8": 1.1897265430e-03,
    "shift_d2.r1": 1.5407071706e-03,
    "shift_d2.r2": 1.1897265430e-03,
    "shift_d2.r5": 1.5407071706e-03,
    "shift_d2.r8": 1.1897265430e-03,
    # Angles for RBends in IR3
    "abw.a6l3": -1.8872909366e-04,
    "abw.a6r3": -1.8872909366e-04,
    "abw.b6l3": -1.8872910038e-04,
    "abw.b6r3": -1.8872910038e-04,
    "abw.c6l3": -1.8872911382e-04,
    "abw.c6r3": -1.8872911382e-04,
    "abw.d6l3": 1.8872911382e-04,
    "abw.d6r3": 1.8872911382e-04,
    "abw.e6l3": 1.8872910038e-04,
    "abw.e6r3": 1.8872910038e-04,
    "abw.f6l3": 1.8872909366e-04,
    "abw.f6r3": 1.8872909366e-04,
    # In-out angle differences for RBends in IR3
    "adiff.bw.a6l3": 1.8872909366e-04,
    "adiff.bw.a6r3": -1.8872909366e-04,
    "adiff.bw.b6l3": 5.6618728769e-04,
    "adiff.bw.b6r3": -5.6618728769e-04,
    "adiff.bw.c6l3": 9.4364550190e-04,
    "adiff.bw.c6r3": -9.4364550190e-04,
    "adiff.bw.d6l3": 9.4364550190e-04,
    "adiff.bw.d6r3": -9.4364550190e-04,
    "adiff.bw.e6l3": 5.6618728769e-04,
    "adiff.bw.e6r3": -5.6618728769e-04,
    "adiff.bw.f6l3": 1.8872909366e-04,
    "adiff.bw.f6r3": -1.8872909366e-04,
    # Shifts for RBends in IR3
    "shift.bw.a6l3": 1.6919790135e-02,
    "shift.bw.a6r3": 1.6919790135e-02,
    "shift.bw.b6l3": 1.6120522415e-02,
    "shift.bw.b6r3": 1.6120522415e-02,
    "shift.bw.c6l3": 1.4521986906e-02,
    "shift.bw.c6r3": 1.4521986906e-02,
    "shift.bw.d6l3": 4.4780130939e-03,
    "shift.bw.d6r3": 4.4780130939e-03,
    "shift.bw.e6l3": 2.8794775851e-03,
    "shift.bw.e6r3": 2.8794775851e-03,
    "shift.bw.f6l3": 2.0802098651e-03,
    "shift.bw.f6r3": 2.0802098651e-03,
    # IR4 RBend angles
    "ad3.l4": 1.5716058651e-03,
    "ad3.r4": 1.5716058651e-03,
    "ad4.l4": 1.5716046865e-03,
    "ad4.r4": 1.5716046865e-03,
    # IR4 RBend shifts
    "shift.mbrb.l4": 1.8564631277e-03,
    "shift.mbrb.r4": 1.8564631277e-03,
    "shift.mbrs.l4": 1.1435410493e-03,
    "shift.mbrs.r4": 1.1435410493e-03,
    # Angles for RBends in IR7
    "abw.a6l7": 1.8872909041e-04,
    "abw.a6r7": 1.8872909041e-04,
    "abw.b6l7": 1.8872909713e-04,
    "abw.b6r7": 1.8872909713e-04,
    "abw.c6l7": -1.8872909713e-04,
    "abw.c6r7": -1.8872909713e-04,
    "abw.d6l7": -1.8872909041e-04,
    "abw.d6r7": -1.8872909041e-04,
    # In-out angle differences for RBends in IR7
    "adiff.bw.a6l7": -1.8872909041e-04,
    "adiff.bw.a6r7": 1.8872909041e-04,
    "adiff.bw.b6l7": -5.6618727795e-04,
    "adiff.bw.b6r7": 5.6618727795e-04,
    "adiff.bw.c6l7": -5.6618727795e-04,
    "adiff.bw.c6r7": 5.6618727795e-04,
    "adiff.bw.d6l7": -1.8872909041e-04,
    "adiff.bw.d6r7": 1.8872909041e-04,
    # Shifts for RBends in IR7
    "shift.bw.a6l7": 1.6919790136e-02,
    "shift.bw.a6r7": 1.6919790136e-02,
    "shift.bw.b6l7": 1.5988412068e-02,
    "shift.bw.b6r7": 1.5988412068e-02,
    "shift.bw.c6l7": 2.8794781954e-03,
    "shift.bw.c6r7": 2.8794781954e-03,
    "shift.bw.d6l7": 2.0802104882e-03,
    "shift.bw.d6r7": 2.0802104882e-03,
    # D1 and D2 strengths [1/m]'kd1.l1': 0.00023938497863369427,
    "kd2.l1": 0.0001929729771192161,
    "kd1.r1": 0.00023938497863369422,
    "kd2.r1": 0.00019297297711921607,
    "kd1.l2": 0.00016216998996044585,
    "kd2.l2": 0.00016216998996044585,
    "kd1.r2": 0.00016216998996044585,
    "kd2.r2": 0.00016216998996044585,
    "kd1.l5": 0.00023938497863369427,
    "kd2.l5": 0.0001929729771192161,
    "kd1.r5": 0.00023938497863369422,
    "kd2.r5": 0.00019297297711921607,
    "kd1.l8": 0.00016216998996044585,
    "kd2.l8": 0.00016216998996044585,
    "kd1.r8": 0.00016216998996044585,
    "kd2.r8": 0.00016216998996044585,
    "kd34.lr3": 5.5508556629e-05,
    "e_kd3.l3": -2.3380076269952575e-15,
    "e_kd4.l3": -2.3379970162010004e-15,
    "e_kd3.r3": -2.338008321555881e-15,
    "e_kd4.r3": -2.3380189323500575e-15,
    "kd34.lr7": 5.5508555673e-05,
    "e_kd3.l7": -2.6314720697379465e-15,
    "e_kd4.l7": -2.6314680732113866e-15,
    "e_kd3.r7": -2.6314486486963916e-15,
    "e_kd4.r7": -2.631452645222881e-15,
    "kd3.l4": 0.0001663074304915815,
    "kd4.l4": 0.00016630730577215885,
    "kd3.r4": 0.00016630743049158152,
    "kd4.r4": 0.00016630730577215882,
}

### MB correction is  0.00035664252265824644 vs nominal of 0.00035664252265800027
### kd12 = (1+6.9e-13) *


def configure_rbend_helper(lhc, element_name, angle, angle_diff, k0, rbend_shift):
    element = lhc[element_name]
    element.rbend_model = "straight-body"
    element.rbend_compensate_sagitta = False
    element.angle = angle
    element.rbend_angle_diff = angle_diff
    element.k0 = k0
    element.rbend_shift = rbend_shift


def config_rbend_ir15(lhc):
    beam_angle_diff_signs = (("b1", -1), ("b2", 1))

    for location, angle_sign, k0_sign, shift_sign in (
        ("r5", -1, -1, -1),
        ("r1", -1, -1, -1),
        ("l5", 1, 1, 1),
        ("l1", 1, 1, 1),
    ):
        angle_ref = lhc.ref[f"ad1.{location}"]
        k0_ref = lhc.ref[f"kd1.{location}"]
        shift_ref = lhc.ref[f"sep_mid_d1.{location}"] / 2
        for beam, angle_diff_sign in beam_angle_diff_signs:
            configure_rbend_helper(
                lhc,
                f"mbxf.4{location}/{beam}",
                angle_sign * angle_ref,
                angle_diff_sign * angle_ref,
                k0_sign * k0_ref,
                shift_sign * shift_ref,
            )

    for location, angle_sign, k0_sign, shift_sign in (
        ("r5", 1, 1, -1),
        ("r1", 1, 1, -1),
        ("l5", -1, -1, 1),
        ("l1", -1, -1, 1),
    ):
        angle_ref = lhc.ref[f"ad2.{location}"]
        k0_ref = lhc.ref[f"kd2.{location}"]
        shift_ref = lhc.ref[f"shift_d2.{location}"]
        for beam, angle_diff_sign in beam_angle_diff_signs:
            configure_rbend_helper(
                lhc,
                f"mbrd.4{location}.{beam}",
                angle_sign * angle_ref,
                angle_diff_sign * angle_ref,
                k0_sign * k0_ref,
                shift_sign * shift_ref,
            )


def config_rbend_ir28(lhc):
    beam_angle_diff_signs = (("b1", 1), ("b2", -1))

    for location, angle_sign, k0_sign, shift_sign in (
        ("r2", 1, 1, 1),
        ("l2", -1, -1, -1),
        ("r8", 1, 1, 1),
        ("l8", -1, -1, -1),
    ):
        angle_ref = lhc.ref[f"ad1.{location}"]
        k0_ref = lhc.ref[f"kd1.{location}"]
        shift_ref = lhc.ref[f"sep_mid_d1.{location}"] / 2
        for beam, angle_diff_sign in beam_angle_diff_signs:
            configure_rbend_helper(
                lhc,
                f"mbx.4{location}/{beam}",
                angle_sign * angle_ref,
                angle_diff_sign * angle_ref,
                k0_sign * k0_ref,
                shift_sign * shift_ref,
            )

    for location, angle_sign, k0_sign, shift_sign in (
        ("r2", -1, -1, 1),
        ("r8", -1, -1, 1),
        ("l2", 1, 1, -1),
        ("l8", 1, 1, -1),
    ):
        angle_ref = lhc.ref[f"ad2.{location}"]
        k0_ref = lhc.ref[f"kd2.{location}"]
        shift_ref = lhc.ref[f"shift_d2.{location}"]
        for beam, angle_diff_sign in beam_angle_diff_signs:
            configure_rbend_helper(
                lhc,
                f"mbrc.4{location}.{beam}",
                angle_sign * angle_ref,
                angle_diff_sign * angle_ref,
                k0_sign * k0_ref,
                shift_sign * shift_ref,
            )


def config_rbend_ir7(lhc):
    for side in ("r7", "l7"):
        for section in "abcd":
            if section in "ab":
                k0 = lhc.ref[f"kd34.lr7"] + lhc.ref[f"e_kd3.{side}"]
            else:
                k0 = -lhc.ref[f"kd34.lr7"] - lhc.ref[f"e_kd4.{side}"]
            prefix = f"{section}6{side}"
            angle_ref = lhc.ref[f"abw.{prefix}"]
            angle_diff_ref = lhc.ref[f"adiff.bw.{prefix}"]
            shift = -lhc.ref[f"shift.bw.{prefix}"]
            for beam, angle_diff_sign in (("b1", 1), ("b2", -1)):
                configure_rbend_helper(
                    lhc,
                    f"mbw.{prefix}.{beam}",
                    angle_ref,
                    angle_diff_sign * angle_diff_ref,
                    k0,
                    shift,
                )


def config_rbend_ir3(lhc):
    for side in ("r3", "l3"):
        for section in "abcdef":
            if section in "abc":
                k0 = -lhc.ref[f"kd34.lr3"] - lhc.ref[f"e_kd3.{side}"]
            else:
                k0 = lhc.ref[f"kd34.lr3"] + lhc.ref[f"e_kd4.{side}"]
            prefix = f"{section}6{side}"
            angle_ref = lhc.ref[f"abw.{prefix}"]
            angle_diff_ref = lhc.ref[f"adiff.bw.{prefix}"]
            shift = lhc.ref[f"shift.bw.{prefix}"]
            for beam, angle_diff_sign in (("b1", 1), ("b2", -1)):
                configure_rbend_helper(
                    lhc,
                    f"mbw.{prefix}.{beam}",
                    angle_ref,
                    angle_diff_sign * angle_diff_ref,
                    k0,
                    shift,
                )


def config_rbend_ir4(lhc):
    for side, beam_b1_angle_diff_sign in (("r4", -1), ("l4", 1)):
        for magnet, element_prefix, angle_key, angle_sign, k0_key, k0_sign in (
            ("mbrs", "mbrs", f"ad3.{side}", -1, f"kd3.{side}", -1),
            ("mbrb", "mbrb", f"ad4.{side}", 1, f"kd4.{side}", 1),
        ):
            angle_ref = lhc[angle_key]
            k0_ref = lhc.ref[k0_key]
            shift = lhc[f"shift.{magnet}.{side}"]
            for beam, beam_sign in (
                ("b1", beam_b1_angle_diff_sign),
                ("b2", -beam_b1_angle_diff_sign),
            ):
                configure_rbend_helper(
                    lhc,
                    f"{element_prefix}.5{side}.{beam}",
                    angle_sign * angle_ref,
                    beam_sign * angle_ref,
                    k0_sign * k0_ref,
                    shift,
                )


def delete_term(ex, var):
    terms = []
    for term in termlist(ex):
        if isinstance(term, xdeps.refs.MulExpr) and term._rhs == var:
            continue
        terms.append(term)
    return sum(terms)


def pprint(expr):
    return "\n + ".join(str(term) for term in termlist(expr))


def termlist(ex, lst=None):
    if lst is None:
        lst = []
    if isinstance(ex, xdeps.refs.AddExpr):
        return lst + termlist(ex._lhs) + termlist(ex._rhs)
    if isinstance(ex, xdeps.refs.SubExpr):
        if isinstance(ex._rhs, xdeps.refs.MulExpr):
            ex = ex._lhs + (-1 * ex._rhs._lhs) * ex._rhs._rhs
        else:
            ex = ex._lhs + (-1) * ex._rhs
        return lst + termlist(ex._lhs) + termlist(ex._rhs)
    return [ex]


def test_chroma_knobs(collider):
    # Test the knobs

    # Correct to zero
    collider.b1.match(
        vary=xt.VaryList(["dqpx.b1_op", "dqpy.b1_op"], step=1e-4),
        targets=xt.TargetSet(dqx=0.0, dqy=0.0, tol=1e-4),
    )
    collider.b2.match(
        vary=xt.VaryList(["dqpx.b2_op", "dqpy.b2_op"], step=1e-4),
        targets=xt.TargetSet(dqx=0.0, dqy=0.0, tol=1e-4),
    )

    # Apply deltas
    collider.vars["dqpx.b1_op"] += 2
    collider.vars["dqpy.b1_op"] += 4
    collider.vars["dqpx.b2_op"] += 3
    collider.vars["dqpy.b2_op"] += 5

    twtest = collider.twiss()

    return twtest.dqx, twtest.dqy


def test_coupling_knobs(collider):
    line = collider.b1
    # Check orthogonality
    line.vars["cmrs.b1_op"] = 1e-3
    line.vars["cmis.b1_op"] = 1e-3
    assert np.isclose(
        collider.b1.twiss().c_minus / np.sqrt(2), 1e-3, rtol=0, atol=1.5e-5
    )

    line.vars["cmrs.b2_op"] = 1e-3
    line.vars["cmis.b2_op"] = 1e-3
    assert np.isclose(
        collider.b2.twiss().c_minus / np.sqrt(2), 1e-3, rtol=0, atol=1.5e-5
    )


class LHCXsuiteModel:
    """Model wrapper around an Xsuite environment for LHC optics workflows."""

    _xt = xt

    slicing_recipe = [
        {"element_type": "Cavity", "num_kicks": 1},
        {"element_type": "CrabCavity", "num_kicks": 1},
        {"element_type": "Sextupole", "num_kicks": 1},
        {"element_type": "Octupole", "num_kicks": 1},
        {
            "element_type": "Multipole",
            "num_kicks": 1,
        },  # in case correctors are thick        multipoles
        {"name": r"mb\..*", "num_kicks": 2},
        {"name": r"mq\..*", "num_kicks": 2},
        {"name": r"mqxa\..*", "num_kicks": 16},  # old triplet
        {"name": r"mqxb\..*", "num_kicks": 16},  # old triplet
        {"name": r"mqxc\..*", "num_kicks": 16},  # new mqxa (q1,q3)
        {"name": r"mqxd\..*", "num_kicks": 16},  # new mqxb (q2a,q2b)
        {"name": r"mqxfa\..*", "num_kicks": 16},  # new (q1,q3 v1.1)
        {"name": r"mqxfb\..*", "num_kicks": 16},  # new (q2a,q2b v1.1)
        {"name": r"mbxa\..*", "num_kicks": 4},  # new d1
        {"name": r"mbxf\..*", "num_kicks": 4},  # new d1 (v1.1)
        {"name": r"mbrd\..*", "num_kicks": 4},  # new d2 (if needed)
        {"name": r"mqyy\..*", "num_kicks": 4},  # new q4
        {"name": r"mqyl\..*", "num_kicks": 4},  # new q5
        {"name": r"mbw\..*", "num_kicks": 4},
        {"name": r"mbx\..*", "num_kicks": 4},
        {"name": r"mbrb\..*", "num_kicks": 4},
        {"name": r"mbrc\..*", "num_kicks": 4},
        {"name": r"mbrs\..*", "num_kicks": 4},
        {"name": r"mqwa\..*", "num_kicks": 4},
        {"name": r"mqwb\..*", "num_kicks": 4},
        {"name": r"mqy\..*", "num_kicks": 4},
        {"name": r"mqm\..*", "num_kicks": 4},
        {"name": r"mqmc\..*", "num_kicks": 4},
        {"name": r"mqml\..*", "num_kicks": 4},
        {"name": r"mqtlh\..*", "num_kicks": 2},
        {"name": r"mqtli\..*", "num_kicks": 2},
        {"name": r"mqt\..*", "num_kicks": 2},
        {"name": r"mqs.*", "num_kicks": 2},
    ]

    def __init__(
        self,
        env=None,
        optics=None,
        settings=None,
        jsonfile=None,
        madxfile=None,
    ):
        self.env = env
        self.optics = optics
        self.settings = settings
        self.jsonfile = jsonfile
        self._var_values = env._xdeps_vref._owner
        self.ref = env._xdeps_vref
        self.eref = env._xdeps_eref
        self.mgr = env._xdeps_manager
        self.madxfile = madxfile
        if hasattr(env, "b1") and hasattr(env, "b2"):
            self.sequence = {1: env.b1, 2: env.b2}
        else:
            self.sequence = None
        self._aperture = None

    @classmethod
    def from_cpymad(cls, madx, sliced=False, madxfile=None):
        if not madx.sequence.lhcb1.has_beam:
            madx.use(sequence="lhcb1")
            madx.use(sequence="lhcb2")

        lines = xt.Environment.from_madx(
            madx=madx, enable_layout_data=True, return_lines=True
        )
        lines["b1"] = lines.pop("lhcb1")
        lines["b2"] = lines.pop("lhcb2")
        lines["b1"].particle_ref = xt.Particles(mass0=xt.PROTON_MASS_EV, p0c=450e9)
        lines["b2"].particle_ref = xt.Particles(mass0=xt.PROTON_MASS_EV, p0c=450e9)

        lhc = xt.Environment(lines=lines)
        for line in lhc.lines.values():
            line.twiss_default["method"] = "4d"
            line.twiss_default["co_search_at"] = "ip7"
            line.twiss_default["strengths"] = True
            line.twiss_default["compute_chromatic_properties"] = False
            if "b2" in line.name:
                line.twiss_default["reverse"] = True
            line.metadata = lines[line.name].metadata

        if sliced:
            for line_name, line in list(lines.items()):
                sliced_line = line.copy()
                sliced_line.slice_thick_elements(
                    slicing_strategies=[
                        xt.Strategy(slicing=None),
                        xt.Strategy(slicing=xt.Uniform(8, mode="thick"), name="mb.*"),
                        xt.Strategy(slicing=xt.Uniform(8, mode="thick"), name="mq.*"),
                    ]
                )
                lhc.lines[f"{line_name}s"] = sliced_line

        return cls(env=lhc, madxfile=madxfile)

    @classmethod
    def from_json(cls, jsonfile):
        jsonfile = str(jsonfile)
        lhc = xt.Environment.from_json(jsonfile)
        return cls(lhc, jsonfile=jsonfile)

    @classmethod
    def from_madx_optics(cls, madxfile):
        env = xt.load(madxfile)
        return cls(env=env, madxfile=madxfile)

    @classmethod
    def from_madx_sequence(cls, filename):
        """Create an `LHCXsuiteModel` from a MAD-X sequence file."""
        seq_text = open(filename).read().lower()
        # Recover expressions in at arguments of the sequences
        assert " at=" in seq_text
        assert ",at=" not in seq_text
        assert "at =" not in seq_text
        seq_text = seq_text.replace(" at=", "at:=")

        # Rename sequences
        assert "LHCB1" not in seq_text
        assert "LHCB2" not in seq_text
        seq_text = seq_text.replace("lhcb1", "b1")
        seq_text = seq_text.replace("lhcb2", "b2")

        lhc = xt.load(
            string=seq_text, format="madx", reverse_lines=["b2"], _rbend_correct_k0=True
        )
        print("Configuring sequence in Xsuite...")
        for nn in list(lhc.elements.keys()):
            if hasattr(lhc.elements[nn], "k0_from_h"):
                lhc.elements[nn].k0_from_h = False

        lhc.vars.update(rbend_data)
        config_rbend_ir15(lhc)
        config_rbend_ir28(lhc)
        config_rbend_ir3(lhc)
        config_rbend_ir4(lhc)
        config_rbend_ir7(lhc)
        lhc["p0c"] = 450e9
        lhc.new_particle("particle_ref_b1", p0c="p0c")
        lhc.new_particle("particle_ref_b2", p0c="p0c")
        lhc.b1.set_particle_ref("particle_ref_b1")
        lhc.b2.set_particle_ref("particle_ref_b2")

        for ll in lhc.lines.values():
            ll.twiss_default["method"] = "4d"
            ll.twiss_default["co_search_at"] = "ip7"
            ll.twiss_default["strengths"] = True
            if "b2" in ll.name:
                ll.twiss_default["reverse"] = True

        lhc["lagrf400.b1"] = 0.5
        lhc["vrf400"] = 6.5
        return cls(env=lhc)

    @classmethod
    def from_madxfile(cls, madxfile, sliced=False):
        """Create an `LHCXsuiteModel` from a MAD-X input file."""

        madmodel = LHCMadxModel.from_madx_scripts(madxfile)
        model = cls.from_cpymad(madmodel.madx, sliced=sliced, madxfile=madxfile)
        model.madxfile = madxfile
        return model

    @classmethod
    def make_json_from_sequence(cls, sequencefile, jsonfile, knobfile):
        print("Loading sequence into Xsuite...")
        model = cls.from_madx_sequence(sequencefile)
        lhc = model.env
        print(f"Saving JSON model to {jsonfile!r}...")
        lhc.to_json(jsonfile)

        jsonfile = Path(jsonfile)
        thin_jsonfile = jsonfile.with_name(jsonfile.stem + "_thin.json")
        print(f"Creating and saving thin JSON model to {thin_jsonfile!r}...")
        cls.make_thin_env(lhc).to_json(thin_jsonfile)

    @classmethod
    def make_thin_env(cls, env):
        slicing_strategies = [xt.Strategy(None)]  # By default leave untouched
        for config in cls.slicing_recipe:
            if "element_type" in config:
                cls = getattr(xt, config["element_type"])
                num_kicks = config["num_kicks"]
                slicing_strategies.append(
                    xt.Strategy(element_type=cls, slicing=xt.Teapot(num_kicks))
                )
            elif "name" in config:
                name = config["name"]
                num_kicks = config["num_kicks"]
                slicing_strategies.append(
                    xt.Strategy(name=name, slicing=xt.Teapot(num_kicks))
                )
        for line in env.lines.values():
            line.slice_thick_elements(slicing_strategies=slicing_strategies)
        return env

    @classmethod
    def set_line_model_drift_kick(cls, line):
        tt = line.get_table()
        tt_no_drift = tt.rows.match_not(
            element_type="Drift"
        )  # need to skip condemned magnets that are drifts
        for config in cls.slicing_recipe:
            if "element_type" in config:
                element_type = config["element_type"]
                num_kicks = config["num_kicks"]
                line.set(
                    tt_no_drift.rows.match(element_type=element_type),
                    model="drift-kick-drift-expanded",
                    integrator="teapot",
                    num_multipole_kicks=num_kicks,
                )
            elif "name" in config:
                name = config["name"]
                num_kicks = config["num_kicks"]
                line.set(
                    tt_no_drift.rows.match(name=name),
                    model="drift-kick-drift-expanded",
                    integrator="teapot",
                    num_multipole_kicks=num_kicks,
                )

    @classmethod
    def set_line_model_thick(cls, line):
        tt = line.get_table()
        tt_no_drift = tt.rows.match_not(
            element_type="Drift"
        )  # need to skip condemned magnets that are drifts
        for config in cls.slicing_recipe:
            if "element_type" in config:
                element_type = config["element_type"]
                line.set(
                    tt_no_drift.rows.match(element_type=element_type),
                    model="adaptive",
                    integrator="adaptive",
                    num_multipole_kicks=0,
                )
            elif "name" in config:
                name = config["name"]
                line.set(
                    tt_no_drift.rows.match(name=name),
                    model="adaptive",
                    integrator="adaptive",
                    num_multipole_kicks=0,
                )

    def __contains__(self, key):
        return key in self._var_values

    def __getitem__(self, key):
        return self._var_values[key]

    def __repr__(self):
        if self.madxfile is not None:
            return f"<LHCXsuiteModel {self.madxfile!r}>"
        if self.jsonfile is not None:
            return f"<LHCXsuiteModel {self.jsonfile!r}>"
        return f"<LHCXsuiteModel {id(self)}>"

    def __setitem__(self, key, value):
        self.ref[key] = value

    @property
    def b1(self):
        return self.env.b1

    @property
    def b1s(self):
        return self.env.b1s

    @property
    def b2(self):
        return self.env.b2

    @property
    def b2s(self):
        return self.env.b2s

    @property
    def p0c(self):
        return self.env.b1.particle_ref.p0c[0]

    #Now done in xsuite with new json files, but leaving here for reference in case we need to do it manually again
    #@p0c.setter
    #def p0c(self, value):
    #    self.env.b1.particle_ref.p0c = value
    #    self.nv.b2.particle_ref.p0c = value

    def _match_d12r(self, ipn):
        lhc = self.env
        ipr = f"ip{ipn}"
        sds = f"s.ds.r{ipn}.b1"
        print(f"match {ipr} right from {ipr} to {sds}")
        lhc.b1.match(
            vary=xt.VaryList([f"kd1.r{ipn}", f"kd2.r{ipn}"], step=1e-12),
            targets=xt.TargetSet(at=sds, px=0, x=0, tol=1e-16),
            betx=1,
            bety=1,
            start=ipr,
            end=sds,
        )

    def _match_d12l(self, ipn):
        lhc = self.env
        ipl = "ip1.l1" if ipn == 1 else f"ip{ipn}"
        eds = f"e.ds.l{ipn}.b1"
        print(f"match {ipl} left from {eds} to {ipl}")
        lhc.b1.match(
            vary=xt.VaryList([f"kd1.l{ipn}", f"kd2.l{ipn}"], step=1e-12),
            targets=xt.TargetSet(at=ipl, px=0, x=0, tol=1e-16),
            betx=1,
            bety=1,
            start=eds,
            end=ipl,
        )

    def _match_d34l(self, ipn):
        lhc = self.env
        ipl = f"ip{ipn}"
        eds = f"e.ds.l{ipn}.b1"
        print(f"match {ipl} left from {eds} to {ipl}")

        lhc.b1.match(
            vary=xt.VaryList([f"e_kd3.l{ipn}", f"e_kd4.l{ipn}"], step=1e-12),
            targets=xt.TargetSet(at=ipl, px=0, x=0, tol=1e-16),
            betx=1,
            bety=1,
            start=eds,
            end=ipl,
        )

    def _match_d34r(self, ipn):
        lhc = self.env
        ipr = f"ip{ipn}"
        sds = f"s.ds.r{ipn}.b1"
        print(f"match {ipr} right from {ipr} to {sds}")
        lhc.b1.match(
            vary=xt.VaryList([f"e_kd3.r{ipn}", f"e_kd4.r{ipn}"], step=1e-12),
            targets=xt.TargetSet(at=sds, px=0, x=0, tol=1e-16),
            betx=1,
            bety=1,
            start=ipr,
            end=sds,
        )

    def _match_d34l4(self):
        lhc = self.env
        ipl = "ip4"
        eds = "e.ds.l4.b1"
        print(f"match {ipl} left from {eds} to {ipl}")
        lhc.b1.match(
            vary=xt.VaryList([f"kd3.l4", f"kd4.l4"], step=1e-12),
            targets=xt.TargetSet(at=ipl, px=0, x=0, tol=1e-16),
            betx=1,
            bety=1,
            start=eds,
            end=ipl,
        )

    def _match_d34r4(self):
        lhc = self.env
        ipr = "ip4"
        sds = "s.ds.r4.b1"
        print(f"match {ipr} right from {ipr} to {sds}")
        lhc.b1.match(
            vary=xt.VaryList([f"kd3.r4", f"kd4.r4"], step=1e-12),
            targets=xt.TargetSet(at=sds, px=0, x=0, tol=1e-16),
            betx=1,
            bety=1,
            start=ipr,
            end=sds,
        )

    def aperture(
        self,
        beam,
        emit=2.5e-6,
        p0c=None,
        bbeat=1.1,
        delta_err=2e-4,
        ndisp_err=0.1,
        co_error=2e-3,
        nsigma=12,
    ):
        if self._aperture is None:
            self._aperture = self.make_aperture()
        tw = self.sequence[beam].twiss(strengths=False)
        ap = self._aperture.apertures[beam - 1]
        if p0c is None:
            p0c = tw.particle_on_co.p0c
        bsx = bbeat * np.sqrt(tw.betx * emit / p0c * 0.938e9 + (tw.dx * delta_err) ** 2)
        xap = (
            ap.offset[:, 0] + tw.x
        )  # position of the beam with respect to the aperture
        ap_xmarg = ap.bbox[:, 0] - abs(xap) - co_error
        ap_x = ap_xmarg / bsx
        tw["ap_bsx"] = bsx
        tw["ap_x"] = ap_x
        tw["ap_xmarg"] = ap_xmarg
        return tw.rows[ap.profile != -1]

    def copy(self):
        return self.__class__(
            env=self.env.copy(),
            settings=self.settings,
            jsonfile=self.jsonfile,
            madxfile=self.madxfile,
        )

    def create_knob(self, knob, verbose=False, set_value=True):
        """
        Create the knob in the model, deleting any previous definition of this knob
        """
        self.delete_knob(knob.name, verbose=verbose, dry_run=False)
        knobname = knob.name
        for wtarget, value in knob.weights.items():
            wname = f"{wtarget}_from_{knobname}"
            if verbose:
                print(f"Setting weight {wname} = {value:15.6g}")
                print(f"Creating expression {wtarget} += {wname} * {knobname}")
            self[wname] = value
            self.ref[wtarget] += self.ref[wname] * self.ref[knobname]
        if set_value:
            if verbose:
                print(f" Setting knob {knobname} = {knob.value:15.6g}")
            self[knobname] = knob.value

    def create_knobs(self, knobs, verbose=False, set_value=True):
        for _, knob in knobs.items():
            self.create_knob(knob, verbose=verbose, set_value=set_value)

    def cycle(self, element):
        self.b1.cycle(element, inplace=True)
        self.b2.cycle(element, inplace=True)

    def delete_knob(self, knobname, verbose=False, dry_run=False):
        direct_deps = list(self.mgr.rdeps.get(self.ref[knobname], {}))
        for dependency in direct_deps:
            if verbose:
                print(f"Deleting dependency {dependency}")
            oldexpr = dependency._expr
            newexpr = delete_term(oldexpr, self.ref[knobname])
            if verbose:
                print(f" Old expr: {oldexpr}")
                print(f" New expr: {newexpr}")
            if not dry_run:
                self.mgr.set_value(dependency, newexpr)
        direct_deps = list(self.mgr.rdeps.get(self.ref[knobname], {}))
        if len(direct_deps) > 0:
            print(f"After deletion, knob {knobname} still has dependencies:")
            for dependency in direct_deps:
                print(f" - {dependency}")
            raise ValueError(f"Knob {knobname} still has dependencies after deletion")

    def diff(self, other):
        all_keys = set(self._var_values.keys()) | set(other._var_values.keys())
        for key in all_keys:
            if key not in self._var_values:
                print(f"{key:20} {other._var_values[key]:15.6g} only in other")
            elif key not in other._var_values:
                print(f"{key:20} {self._var_values[key]:15.6g} only in self")
            elif self._var_values[key] != other._var_values[key]:
                print(
                    f"{key:20} {self._var_values[key]:15.6g} {other._var_values[key]:15.6g}"
                )

    def filter(self, pattern):
        var_values = list(self._var_values.keys())
        return list(filter(lambda item: re.match(pattern, item), var_values))

    def get_acb_names(self, pattern="mcb", debug=False):
        out = {}
        for k, v in self.search(pattern).items():
            if hasattr(v, "knl"):
                if "h" in k:
                    exp = self.eref[k].knl[0]._expr
                else:
                    exp = self.eref[k].ksl[0]._expr
                if exp is not None:
                    out[k] = exp._get_dependencies().pop()._key
                elif debug:
                    print(f"Warning: no expression found for {k}")
        return out

    def get_kq_names(self, pattern="mq[^s]", debug=False):
        out = {}
        for k, v in self.search(pattern).items():
            if hasattr(v, "k1"):
                exp = self.eref[k].k1._expr
                if exp is not None:
                    out[k] = exp._get_dependencies().pop()._key
                elif debug:
                    print(f"Warning: no expression found for {k}")
        return out

    def get(self, key, default=None):
        return self._var_values.get(key, default)

    def get_cmin(self, beam=None, pos="ip1"):
        """Compute the c-minus at a given position."""
        if beam is None:
            return [
                self.get_cmin(beam=1, pos=pos),
                self.get_cmin(beam=2, pos=pos),
            ]
        line = self.sequence[beam]
        if line.element_names[0] != pos:
            line.cycle(pos, inplace=True)
        tw = line.twiss(compute_chromatic_properties=False, strengths=True)
        k1sl = tw["k1sl"]
        pi2 = 2 * np.pi
        j2pi = 1j * pi2
        cmin = (
            np.sum(k1sl * np.sqrt(tw.betx * tw.bety) * np.exp(j2pi * (tw.mux - tw.muy)))
            / pi2
        )
        if line.element_names[0] != "ip1":
            line.cycle("ip1", inplace=True)

    def get_knob(self, knob):
        value = self._var_values[knob.name]
        weights = {}
        for wname in knob.weights:
            weights[wname] = self._var_values[f"{wname}_from_{knob.name}"]
        return Knob(knob.name, value, weights, variant=knob.variant).specialize()

    def get_knob_by_probing(self, name, variant=None, verbose=False):
        if verbose:
            print(f"Getting knob {name} by probing with variant={variant}")
        weights = {}
        oldvars = self._var_values.copy()
        oldvalue = self._var_values[name]
        self[name] = oldvalue + 1
        for key in self._var_values:
            vnew = self._var_values[key]
            if hasattr(vnew, "__sub__"):
                dvar = self._var_values[key] - oldvars[key]
                if dvar != 0:
                    weights[key] = dvar
                    if verbose:
                        print(f"Weight {key} = {dvar:15.6g}")
        del weights[name]
        self[name] = oldvalue
        return Knob(name, oldvalue, weights, variant=variant).specialize()

    def get_knob_by_weight_names(self, name, variant=None, verbose=False):
        if verbose:
            print(f"Getting knob {name} by weight names with variant={variant}")
        weights = {}
        value = self._var_values[name]
        weight_suffix = f"_from_{name}"
        for key in self._var_values:
            if key.endswith(weight_suffix):
                weights[key.split(weight_suffix)[0]] = self._var_values[key]
                if verbose:
                    print(
                        f"Weight {key.split(weight_suffix)[0]} = {self._var_values[key]:15.6g}"
                    )
        return Knob(name, value, weights, variant=variant).specialize()

    def get_knob_by_xdeps(self, name, variant=None, verbose=False):
        if verbose:
            print(f"Getting knob {name} by xdeps with variant={variant}")
        mgr = self.env.ref_manager
        if name not in self.env:
            return Knob(name, value=0.0, weights={}, variant=variant).specialize()
        ref = self.env.ref[name]
        weight_names = [item._key for item in mgr.rdeps[ref]]
        var_values = self._var_values
        weights = {}
        tasks = mgr.tasks
        for wname in weight_names:
            expr = tasks[self.env.ref[wname]].expr
            if verbose:
                print(f"Weight {wname}")
            for term in termlist(expr):
                if isinstance(term, xdeps.refs.MulExpr) and term._rhs == ref:
                    if np.isscalar(term._lhs):
                        value = float(term._lhs)
                    else:
                        value = term._lhs._get_value()
                        if verbose:
                            print(f"   Term: {term}")
                            print(f"   Weight: {value}")
                    weights[wname] = value
        value = self[name]
        return Knob(name, value, weights, variant=variant).specialize()

    def get_mo_rdt(self, qx=62.270, qy=60.295, i_mo=40, beam=None, verbose=True):
        if beam is None:
            tw1, rdt1 = self.get_mo_rdt(qx=qx, qy=qy, i_mo=i_mo, beam=1, verbose=False)
            tw2, rdt2 = self.get_mo_rdt(qx=qx, qy=qy, i_mo=i_mo, beam=2, verbose=False)
            if verbose:
                print(f"Avg. RDTs {'Beam 1':>15} {'Beam 2':>15}")
                for rdt_name in ["f4000", "f0004", "f2002"]:
                    avg1 = np.mean(np.abs(rdt1[rdt_name])) / 1e4
                    avg2 = np.mean(np.abs(rdt2[rdt_name])) / 1e4
                    print(f"{rdt_name}/1e4 {avg1:15.6g} {avg2:15.6g}")
            return (tw1, rdt1), (tw2, rdt2)

        lhc = self.env
        lhc[f"dqx.b{beam}"] = qx - 62.31
        lhc[f"dqy.b{beam}"] = qy - 60.32
        line = lhc.b1 if beam == 1 else lhc.b2
        tw = line.twiss(method="4d", reverse=False)
        if verbose:
            print(f"Setting dqx.b{beam} dqy.b{beam} for tunes {tw.qx:.6f} {tw.qy:.6f}")
            print("Setting MO knobs for i_mo and on_mo")
        motf = lhc["kmax_mo"] / lhc["imax_mo"] / lhc.b1.particle_ref.rigidity0[0]
        lhc[f"i_mo.b{beam}"] = 0
        lhc[f"on_mo.b{beam}"] = 0
        for knob_name in lhc.vars.get_table().rows[f"ko[fd].a..b{beam}"].name:
            lhc[knob_name] = f"i_mo.b{beam}*{motf} - 6*on_mo.b{beam}"
        lhc[f"i_mo.b{beam}"] = i_mo
        rdts = ["f4000", "f0004", "f2002"]
        strengths = line.get_table(attr=True)
        rdt = xt.rdt_first_order_perturbation(rdt=rdts, twiss=tw, strengths=strengths)
        if verbose:
            print("Setting MO to 0")
        for knob_name in lhc.vars.get_table().rows[f"ko[fd].a..b{beam}"].name:
            lhc[knob_name] = 0
        lhc[f"dqx.b{beam}"] = 0
        lhc[f"dqy.b{beam}"] = 0
        return tw, rdt

    def get_p0c(self):
        return self.env.b1.particle_ref.p0c[0]

    def get_survey(self, beam=None):
        if beam is None:
            return [self.get_survey(beam=1), self.get_survey(beam=2)]
        if beam == 1:
            return self.sequence[beam].survey()
        return self.sequence[beam].survey(reverse=False).reverse()

    def get_survey_flat(self, beam=None):
        if beam is None:
            return [self.get_survey_flat(beam=1), self.get_survey_flat(beam=2)]
        line = self.sequence[beam].copy()
        for name, element in line.element_dict.items():
            if name.startswith("mb."):
                element.h = 0
        survey = line.survey(reverse=False)  # needs to force it because not supported
        if beam == 2:
            survey = survey.reverse()
        return survey

    def get_triplet_chrom(self, beam=1, verbose=False):
        line = self.sequence[beam]
        tw = line.twiss(strengths=False, compute_chromatic_properties=False)
        betxip1 = tw["betx", "ip1"]
        betyip1 = tw["bety", "ip1"]
        betxip5 = tw["betx", "ip5"]
        betyip5 = tw["bety", "ip5"]
        ip1left = line.twiss(
            init_at="ip1.l1",
            betx=betxip1,
            bety=betyip1,
            start=f"ms.10l1.b{beam}",
            end="ip1.l1",
            compute_chromatic_properties=True,
        )
        wxip1l = ip1left.wx_chrom[1]
        wyip1l = ip1left.wy_chrom[1]
        ip1right = line.twiss(
            init_at="ip1",
            betx=betxip1,
            bety=betyip1,
            start="ip1",
            end=f"ms.10r1.b{beam}",
            compute_chromatic_properties=True,
        )
        wxip1r = ip1right.wx_chrom[-2]
        wyip1r = ip1right.wy_chrom[-2]
        ip5left = line.twiss(
            init_at="ip5",
            betx=betxip5,
            bety=betyip5,
            start=f"ms.10l5.b{beam}",
            end="ip5",
            compute_chromatic_properties=True,
        )
        wxip5l = ip5left.wx_chrom[1]
        wyip5l = ip5left.wy_chrom[1]
        ip5right = line.twiss(
            init_at="ip5",
            betx=betxip5,
            bety=betyip5,
            start="ip5",
            end=f"ms.10r5.b{beam}",
            compute_chromatic_properties=True,
        )
        wxip5r = ip5right.wx_chrom[-2]
        wyip5r = ip5right.wy_chrom[-2]
        if verbose:
            print(f"ip1: left wx={wxip1l} wy={wyip1l} right wx={wxip1r} wy={wyip1r}")
            print(f"ip5: left wx={wxip5l} wy={wyip5l} right wx={wxip5r} wy={wyip5r}")
        dwx1 = (wxip1r - wxip1l) / 2
        dwy1 = (wyip1r - wyip1l) / 2
        dwx5 = (wxip5r - wxip5l) / 2
        dwy5 = (wyip5r - wyip5l) / 2
        return dwx1, dwy1, dwx5, dwy5

    def get_variant(self):
        vv = self.env.vars
        if "acbrdh4.l1b1" in vv:
            variant = "hl"
        elif vv["kqx.l1"] > 0:
            variant = "lhc2024"
        elif vv["kqx.l5"] > 0:
            variant = "lhc2025"
        else:
            variant = "lhc"
        return variant

    def info(self, key):
        return self.env.vars[key]._info(limit=None)

    def is_full(self):
        return self.b1 is not None and self.b2 is not None

    def keys(self):
        return self._var_values.keys()

    def knob_check(self, knob, verbose=False):
        """
        Return True has the expeceted structure
        Return False has a different structure
        """
        knobname = knob.name
        deps = self.mgr.rdeps.get(self.ref[knobname], {})
        depnames = {dep._key for dep in deps}
        if verbose:
            print(f"Check knob {knobname}")
            for dep in deps:
                print(f"- {dep._key} {dep._expr}")
        if depnames != set(knob.weights.keys()):
            if verbose:
                knob_weights = set(knob.weights.keys())
                print(f"Model weights `{depnames}` != knob weights `{knob_weights}`")
            return False
        return True

    def knobs_delete_all(self):
        for variable in self._var_values:
            self[variable] = self[variable]

    def load(self, madxfile):
        self.env.vars.load(madxfile)
        return self

    def make_and_set0_knobs(self, knob_names, variant=None):
        knobs = {}
        for knob_name in knob_names:
            knob = self.get_knob_by_xdeps(knob_name, variant=variant)
            knobs[knob_name] = knob
            self[knob_name] = 0
        return knobs

    def make_thin_model(self):
        self.__class__.make_thin_env(self.env)
        return self

    def make_aperture(self):
        from .aperture import LHCAperture

        return LHCAperture.from_xsuite_model(self)

    def match(self, *args, **kwargs):
        return self.env.match(*args, **kwargs)

    def match_chroma(
        self, beam=None, dqx=0, dqy=0, arcs="weak", solve=True, verbose=True
    ):
        """
        Match the chromaticity of the optics.

        NB: breaks knobs and restore them
        """
        if beam is None:
            for beam_number in [1, 2]:
                self.match_chroma(
                    beam=beam_number,
                    dqx=dqx,
                    dqy=dqy,
                    arcs=arcs,
                    solve=solve,
                    verbose=verbose,
                )
            return None

        model = self
        beam_name = f"b{beam}"
        line = getattr(model, beam_name)
        for fd in "fd":
            for strength_name in self.search(f"ks{fd}[12]\\.a..{beam_name}$"):
                model.ref[strength_name] = model[
                    strength_name
                ]  # reset otherwise error in knobs
                if arcs == "weak":
                    if (
                        "a81" in strength_name
                        or "a12" in strength_name
                        or "a45" in strength_name
                        or "a56" in strength_name
                    ):
                        continue
                if arcs == "strong":
                    if (
                        "a23" in strength_name
                        or "a34" in strength_name
                        or "a67" in strength_name
                        or "a78" in strength_name
                    ):
                        continue
                tmp = f"ks{fd}_{beam_name}"
                model[tmp] = model[strength_name]
                if verbose:
                    print(f"Set {tmp} from {strength_name} to {model[tmp]}")
                model.ref[strength_name] = model.ref[tmp]
        mtc = line.match(
            solve=False,
            vary=[xt.VaryList([f"ksf_{beam_name}", f"ksd_{beam_name}"], step=1e-9)],
            targets=[xt.TargetSet(dqx=dqx, dqy=dqy, tol=1e-6)],
            strengths=False,
            compute_chromatic_properties=True,
            n_steps_max=50,
            verbose=False,
        )
        if not verbose:
            mtc._err.show_call_counter = False
        if solve:
            mtc.solve(broyden=True)
        if verbose:
            match_compare_log(mtc)
        return mtc

    def match_knob(self, *args, **kwargs):
        return self.env.match_knob(*args, **kwargs)

    def match_dipoles(self, check=False):
        for ipn in [1, 2, 5, 8]:
            self._match_d12l(ipn)
            self._match_d12r(ipn)
        for ipn in [3, 7]:
            self._match_d34l(ipn)
            self._match_d34r(ipn)
        self._match_d34l4()
        self._match_d34r4()
        if check:
            tw = self.b1.twiss(strengths=False)
            print(f"max orbit is {max(abs(tw.x))} m in b1")
            tw = self.b2.twiss(strengths=False)
            print(f"max orbit is {max(abs(tw.x))} m in b2")

        lhc = self.env
        var_names = []
        for arc in [12, 23, 34, 45, 56, 67, 78, 81]:
            lhc[f"kb.a{arc}"] = f"ab.a{arc}/l.mb*(1+6.9e-13)"

        for ipn in [1, 2, 5, 8]:
            var_names += [f"kd1.l{ipn}", f"kd2.l{ipn}", f"kd1.r{ipn}", f"kd2.r{ipn}"]
        for ipn in [3, 7]:
            var_names += [f"kd34.lr{ipn}"]
            var_names += [
                f"e_kd3.l{ipn}",
                f"e_kd4.l{ipn}",
                f"e_kd3.r{ipn}",
                f"e_kd4.r{ipn}",
            ]
        var_names += [f"kd3.l4", f"kd4.l4", f"kd3.r4", f"kd4.r4"]
        return {var_name: float(lhc[var_name]) for var_name in var_names}

    def match_w(self, beam=None, target="triplet", k2max=0.42, verbose=True):
        """
        Docstring for match_w

        K2max=1.280/0.017^2/23348.89927*2*600/550; !=0.4138703096
        """

        if beam is None:
            return [
                self.match_w(beam=1, target=target, k2max=k2max, verbose=verbose),
                self.match_w(beam=2, target=target, k2max=k2max, verbose=verbose),
            ]

        line = getattr(self, f"b{beam}")
        if beam == 1:
            strong_f = [f"ksf1.a{aa}b{beam}" for aa in [81, 12, 45, 56]]
            strong_d = [f"ksd2.a{aa}b{beam}" for aa in [81, 12, 45, 56]]
            weak_f = [f"ksf2.a{aa}b{beam}" for aa in [81, 12, 45, 56]]
            weak_d = [f"ksd1.a{aa}b{beam}" for aa in [81, 12, 45, 56]]
        else:
            strong_f = [f"ksf2.a{aa}b{beam}" for aa in [81, 12, 45, 56]]
            strong_d = [f"ksd1.a{aa}b{beam}" for aa in [81, 12, 45, 56]]
            weak_f = [f"ksf1.a{aa}b{beam}" for aa in [81, 12, 45, 56]]
            weak_d = [f"ksd2.a{aa}b{beam}" for aa in [81, 12, 45, 56]]

        vary = xt.VaryList(strong_f + strong_d, step=1e-3, limits=(-k2max, k2max))
        for knob_name in weak_f:
            line.vars[knob_name] = 0.06
        for knob_name in weak_d:
            line.vars[knob_name] = -0.099

        if target == "triplet":
            ix1, iy1, ix5, iy5 = self.get_triplet_chrom(beam=beam)
            targets = [
                xt.TargetSet(
                    bx_chrom=0, ax_chrom=0, by_chrom=0, ay_chrom=0, tol=1e-6, at="ip3"
                ),
                xt.TargetSet(
                    bx_chrom=0, ax_chrom=0, by_chrom=0, ay_chrom=0, tol=1e-6, at="ip7"
                ),
                xt.TargetSet(
                    bx_chrom=0,
                    ax_chrom=-ix1,
                    by_chrom=0,
                    ay_chrom=-iy1,
                    tol=1e-6,
                    at="ip1",
                ),
                xt.TargetSet(
                    bx_chrom=0,
                    ax_chrom=-ix5,
                    by_chrom=0,
                    ay_chrom=-iy5,
                    tol=1e-6,
                    at="ip5",
                ),
            ]
        elif target == "max_chrom":
            targets = [
                xt.TargetSet(wx_chrom=0, wy_chrom=0, at="ip3"),
                xt.TargetSet(wx_chrom=0, wy_chrom=0, at="ip7"),
                xt.Target(lambda tw: tw.wx_chrom.max(), value=700),
                xt.Target(lambda tw: tw.wy_chrom.max(), value=700),
            ]
        else:
            raise ValueError(
                f"target can be only 'triplet' or 'max_chrom', not {target!r}"
            )

        mtc = line.match(
            solve=False,
            vary=vary,
            targets=targets,
            strengths=False,
            compute_chromatic_properties=True,
            verbose=False,
        )
        if not verbose:
            mtc._err.show_call_counter = False
        if target == "triplet":
            mtc.run_jacobian(n_steps=10)
        elif target == "max_chrom":
            mtc.run_simplex(n_steps=50)
        if verbose:
            match_compare_log(mtc)
        return mtc

    def plot_aperture(self, beam=None):
        if beam is None:
            return [self.plot_aperture(beam=1), self.plot_aperture(beam=2)]
        su = self.get_survey_flat(beam)
        line = self.sequence[beam]
        fig, (ax1, ax2) = plt.subplots(2, 1, num=f"aperture{beam}", clear=True)
        print(su, line, fig, ax1, ax2)
        raise NotImplementedError("Not implemented")

    def plot_beamsize(
        self,
        beam=None,
        emit_n=2.5e-6,
        p0c=None,
        bbeat=1.1,
        co_error=2e-3,
        ndisp_err=0.1,
        delta_err=2e-4,
        nsigma=12,
        survey=False,
    ):
        if beam is None:
            return [
                self.plot_beamsize(
                    beam=1,
                    emit_n=emit_n,
                    p0c=p0c,
                    bbeat=bbeat,
                    co_error=co_error,
                    ndisp_err=ndisp_err,
                    delta_err=delta_err,
                    nsigma=nsigma,
                    survey=survey,
                ),
                self.plot_beamsize(
                    beam=2,
                    emit_n=emit_n,
                    p0c=p0c,
                    bbeat=bbeat,
                    co_error=co_error,
                    ndisp_err=ndisp_err,
                    delta_err=delta_err,
                    nsigma=nsigma,
                    survey=survey,
                ),
            ]
        line = self.sequence[beam]
        tw = line.twiss(strengths=False)
        if p0c is None:
            p0c = self.p0c
        ex = emit_n / p0c * 0.9382720813e9
        ey = ex
        dx_err = 2.0 * np.sqrt(tw.betx / 170) * ndisp_err
        dy_err = 2.0 * np.sqrt(tw.bety / 170) * ndisp_err
        dx = tw.dx + dx_err
        dy = tw.dy + dy_err
        sx = nsigma * bbeat * np.sqrt(tw.betx * ex) + abs(dx) * delta_err + co_error
        sy = nsigma * bbeat * np.sqrt(tw.bety * ey) + abs(dy) * delta_err + co_error
        x = tw.x
        y = tw.y
        if survey:
            survey_data = self.get_survey_flat(beam)
            x += survey_data.X
            y += survey_data.Y
        xp = x + sx
        xm = x - sx
        yp = y + sy
        ym = y - sy
        if not survey:
            fig, (ax1, ax2) = plt.subplots(2, 1, num=f"aperture{beam}", clear=True)
        else:
            if beam == 1:
                fig, (ax1, ax2) = plt.subplots(
                    2, 1, num=f"aperture{beam}", figsize=(12, 6)
                )
            else:
                ax1, ax2 = plt.gcf().get_axes()
        color = "b" if beam == 1 else "r"
        ax1.plot(tw.s, x, label="x", color=color)
        ax2.plot(tw.s, y, label="y", color=color)
        ax1.set_ylabel("x [m]")
        ax2.set_ylabel("y [m]")
        ax1.fill_between(tw.s, xp, xm, alpha=0.5, color=color, label=f"{nsigma} sigma")
        ax2.fill_between(tw.s, yp, ym, alpha=0.5, color=color, label=f"{nsigma} sigma")
        return ax1, ax2

    def plot_survey(self, beam=None):
        if beam is None:
            return [self.plot_survey(beam=1), self.plot_survey(beam=2)]
        survey = self.get_survey(beam)

        plt.figure("LHC Survey", figsize=(6, 6))
        ax = plt.subplot(111)
        color = "b" if beam == 1 else "r"
        ax.plot(survey.Z, survey.X, label=f"Beam {beam}", color=color)
        ax.set_xlabel("Z [m]")
        ax.set_ylabel("X [m]")
        survey_ips = survey.rows["ip[1-8]"].cols["name Z X"]
        for name, x_coord, y_coord in survey_ips.rows:
            plt.text(x_coord, y_coord, name.upper(), color="black")
        return self

    def plot_survey_flat(self, figsize=(12, 3)):
        survey_b1 = self.get_survey_flat(beam=1)
        survey_b2 = self.get_survey_flat(beam=2)
        plt.figure("LHC Survey Flat", figsize=figsize)
        plt.plot(survey_b1.s, survey_b1.X, label="Beam 1", color="blue")
        plt.plot(survey_b2.s, survey_b2.X, label="Beam 2", color="red")
        plt.xlabel("S [m]")
        plt.ylabel("X [m]")
        survey_ips = survey_b1.rows["ip[1-8]"].cols["name s X"]
        for name, x_coord, _ in survey_ips.rows:
            plt.text(x_coord, 0, name.upper(), color="black")
        plt.tight_layout()
        return self

    def print_deps(self, varname):
        ref = self.env.ref[varname]
        direct_deps = self.mgr.rdeps.get(ref, {})
        print(f"Direct dependencies of {varname!r}:")
        for dependency in direct_deps:
            print(f" - {dependency} {dependency._expr}")
        print(f"Indirect dependencies of {varname!r}:")
        indirect_deps = set(self.mgr.find_deps([ref])) - set(direct_deps) - {ref}
        for dependency in indirect_deps:
            print(f" - {dependency} {dependency._expr}")

    def search(self, pattern):
        out = {}
        for key in self._var_values.keys():
            if re.match(pattern, key):
                out[key] = self.env[key]
        for key, value in self.env.elements.items():
            if re.match(pattern, key):
                out[key] = value
        return out

    def set_model_driftkick(self):
        for line in self.env.lines.values():
            self.__class__.set_line_model_driftkick(line)

    def set_model_thick(self):
        for line in self.env.lines.values():
            self.__class__.set_line_model_thick(line)

    def show_knob(self, knobname):
        print(f"Knob: {knobname} = {self[knobname]:15.6g}")
        for dependency in self.mgr.rdeps.get(self.ref[knobname], {}):
            print("Target:", dependency._key)
            print("     Expr:", pprint(dependency._expr))
            wname = f"{dependency._key}_from_{knobname}"
            if wname in self:
                print(f"    Weight {wname} = {self[wname]:15.6g}")

    def slice(self, slices=8):
        for beam in [1, 2]:
            line = self.sequence[beam]
            line.slice_thick_elements(
                slicing_strategies=[
                    xt.Strategy(slicing=None),
                    xt.Strategy(slicing=xt.Uniform(slices, mode="thick"), name="mb.*"),
                    xt.Strategy(slicing=xt.Uniform(slices, mode="thick"), name="mq.*"),
                ]
            )

    def to_json(self, jsonfile="lhc.json"):
        self.env.to_json(jsonfile)

    def twiss(
        self,
        start=None,
        end=None,
        init=None,
        init_at=None,
        zero_at=None,
        beam=None,
        full=True,
        chrom=False,
        strengths=True,
    ):
        """
        :param init: None gives periodic solution, else propagate init to "start" and "end"
        :param start: Starting point of the output twiss table
        :param end: Ending point of the output twiss table

        Examples
        - twiss(): periodic solution, full machine, start/end of the line
        - twiss(start="ip8", end="ip2"): as before by data at the start/end of the line
        - twiss(start="ip8", end="ip2", init_at="ip1"): periodic solution, full machine, start/end of the line, s,mux,muy=0 at ip1
        - twiss(start="ip8", end="ip2", init="init"):

        NB: Still fails when full=False and boundaries are reversed w.r.t the line orde
        """
        if beam is None:
            return [
                self.twiss(
                    start,
                    end,
                    init,
                    init_at=init_at,
                    full=full,
                    chrom=chrom,
                    beam=1,
                    strengths=strengths,
                ),
                self.twiss(
                    start,
                    end,
                    init,
                    init_at=init_at,
                    full=full,
                    chrom=chrom,
                    beam=2,
                    strengths=strengths,
                ),
            ]
        if beam == 1:
            line_start = self.sequence[beam].element_names[0]
            line_end = self.sequence[beam].element_names[-1]
        else:
            line_start = self.sequence[beam].element_names[-1]
            line_end = self.sequence[beam].element_names[0]
        if start is None and end is None:
            start = line_start
            end = line_end
        if full:
            boundary_start = line_start
            boundary_end = line_end
        else:
            boundary_start = start
            boundary_end = end
        if init is None:
            if init_at is None:
                init_at = start
            init = self.twiss_init(
                boundary_start,
                boundary_end,
                init_at,
                beam,
                chrom=chrom,
                strengths=strengths,
            )
            init.s = 0
            init.mux = 0
            init.muy = 0
        return self.twiss_open(start, end, init, beam, strengths=strengths, chrom=chrom)

    def twiss_init(self, start, end, init_at, beam, chrom=False, strengths=True):
        init = (
            self.sequence[beam]
            .twiss(
                start=start,
                end=end,
                init="periodic",
                strengths=strengths,
                compute_chromatic_properties=chrom,
            )
            .get_twiss_init(init_at)
        )
        return init

    def twiss_open(self, start, end, init, beam, strengths=True, chrom=False):
        return self.sequence[beam].twiss(
            start=start,
            end=end,
            init=init,
            strengths=strengths,
            compute_chromatic_properties=chrom,
        )

    def update(self, src, knobs_check=True):
        if hasattr(src, "strengths"):
            self.update_vars(src.strengths)
        else:
            self.update_vars(src)
        if hasattr(src, "knobs"):
            self.update_knobs(src.knobs, knobs_check=knobs_check)

    def update_knob(self, knob, verbose=False, set_value=True, knobs_check=True):
        """
        Update the model with the knob weight values

        Check that the knob exists.
        If it exists, check that has the same structure
        else raise an error.
        """
        knobname = knob.name

        if knobs_check:
            check = self.knob_check(knob)
            if check is False:
                self.knob_check(knob, verbose=True)
                raise ValueError(f"Knob {knobname} has different structure in {self}")

        if set_value:
            if verbose and knob.value != self[knobname]:
                print(
                    f"Update {knobname} from {self[knobname]:15.6g} to {knob.value:15.6g}"
                )
                self[knobname] = knob.value
        for wtarget, value in knob.weights.items():
            wname = f"{wtarget}_from_{knobname}"
            if verbose and wname in self and self[wname] != value:
                print(f"Update {wname} from {self[wname]:15.6g} to {value:15.6g}")
            self[wname] = value

    def update_knobs(self, knobs, verbose=False, set_value=True, knobs_check=True):
        for _, knob in knobs.items():
            self.update_knob(
                knob,
                verbose=verbose,
                set_value=set_value,
                knobs_check=knobs_check,
            )

    def update_vars(self, strengths, verbose=False):
        for key, value in strengths.items():
            if verbose:
                if key in self and self[key] != value:
                    print(f"Update {key} from {self[key]:15.6g} to {value:15.6g}")
            self[key] = value

    def update_from_madx_optics(self, filename, knobs=None, verbose=False):
        self.env.vars.load(filename)
        if knobs is not None:
            self.create_knobs(knobs)


class SinglePassDispersion(xdeps.Action):
    """Action computing single-pass dispersion between two elements."""

    def __init__(self, line, ele_start, ele_stop, backtrack=False, delta=1e-3):
        self.line = line
        self.ele_start = ele_start
        self.ele_stop = ele_stop
        self.delta = delta
        self.backtrack = backtrack
        self._pp = line.build_particles(delta=delta)

    def run(self):
        for name in ["x", "px", "y", "py", "zeta", "delta", "at_element"]:
            setattr(self._pp, name, 0)
        self._pp.delta = self.delta
        self.line.track(
            self._pp,
            ele_start=self.ele_start,
            ele_stop=self.ele_stop,
            backtrack=self.backtrack,
        )
        return {
            f"d{name}": getattr(self._pp, name)[0] / self.delta
            for name in ["x", "px", "y", "py"]
        }
