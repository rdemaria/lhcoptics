import re

import xdeps as xd


from .knob import Knob
from .utils import print_diff_dict_float


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


class MADSequence:
    def __init__(self, madx, sequence_name):
        self.madx = madx
        self.sequence = sequence_name

    @property
    def madx_sequence(self):
        return self.madx.sequence[self.sequence]

    @property
    def start(self):
        self.madx_sequence.elements[1].name

    @property
    def end(self):
        self.madx_sequence.elements[-2].name

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
    ):
        """
        Parameters:
        - full: if False consider only the portion from start to end
        - at: location to which initial conditions are to be set
        - init, betx, .... : initial conditions if all are None, then they are taken from the periodic solution at `at`
        - start, end: start and end of the twiss output
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
        return xd.Table(tw)

    def twiss_periodic(self, use=False):
        if use:
            self.madx.use(sequence=self.sequence)
        tw = self.madx.twiss(sequence=self.sequence)
        return xd.Table(tw)

    def twiss_line(self, betx=1, bety=1, alfx=0, alfy=0):
        tw = self.madx.twiss(
            betx=betx, bety=bety, alfx=alfx, alfy=alfy, sequence=self.sequence
        )
        return xd.Table(tw)

    def rmatrix(self, start=None, end=None):
        if start is None:
            start = self.start
        if end is None:
            end = self.end
        self.madx.use(sequence=self.sequence, range=f"{start}/{end}")
        tw = self.madx.twiss(betx=1, bety=1, rmatrix=True)
        return RMatrix(tw.getmat("re", -1, 6, 6))

    def twiss_line_back(
        self, start, end, betx=1, bety=1, alfx=0, alfy=0, dx=0, dpx=0
    ):
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
        tw = self.madx.twiss(
            betx=bx1, bety=by1, alfx=ax1, alfy=ay1, dx=dx1, dpx=dpx1
        )
        return tw


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

    extra_madx = (
        extra_defs
        + """
beam, particle=proton, energy=6.8e12, sequence=lhcb1;
beam, particle=proton, energy=6.8e12, sequence=lhcb2, bv=-1;
use, sequence=lhcb1;
use, sequence=lhcb2;
    """
    )

    def __init__(self, madx):
        self.madx = madx
        self.b1 = MADSequence(self.madx, "lhcb1")
        self.b2 = MADSequence(self.madx, "lhcb2")

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

    @classmethod
    def from_madxfile(cls, madxfile, extra=True):
        from cpymad.madx import Madx

        madx = Madx()
        madx.options(echo=False, warn=False, info=False)
        madx.call(str(madxfile))
        if extra:
            madx.input(cls.extra_madx)
        return cls(madx)

    def twiss(self, sequence=None, **kwargs):
        if sequence is None:
            return (self.b1.twiss(**kwargs), self.b2.twiss(**kwargs))
        else:
            return getattr(self, f"b{sequence}").twiss(**kwargs)

    def call(self, filename):
        self.madx.call(filename)
        return self

    def filter(self, pattern):
        return [k for k in self.madx.globals if re.match(pattern, k)]

    def update_vars(self, strengths, verbose=False):
        for k, v in strengths.items():
            if verbose:
                print(f"{k:20} {self[k]:15.6g} -> {v:15.6g}")
            self[k] = v

    def update_knobs(self, knobs, verbose=False, knobs_off=False):
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
                        raise ValueError(
                            f"{wn} depends on {k} but not on {name}"
                        )
                if expr is None:
                    wnvalue = self[wn]
                    self.madx.input(f"{wn} := {wnvalue} + ({name} * {k});")
                elif expr is not None and k not in expr:
                    self.madx.input(f"{wn} := {expr} + ({name} * {k});")

    def update(self, src):
        if hasattr(src, "strengths"):
            self.update_vars(src.strengths)
        else:
            self.update_vars(src)
        if hasattr(src, "knobs"):
            self.update_knobs(src.knobs)

    def mad_mkknob(self, name):
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
        return Knob(name, value, weights)

    def mad_find_and_set0_knobs(madx, strengths):
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
            knobs[knob] = Knob(knob, value, weights)
        return knobs

    def make_and_set0_knobs(self, knob_names):
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
            knobs[knob] = Knob(knob, value, weights)
        return knobs

    def diff(self, other):
        selfvar = {
            k: v for k, v in self.madx.globals.items() if isinstance(v,float)
        }
        othervar = {
            k: v for k, v in other.madx.globals.items() if isinstance(v,float)
        }
        print_diff_dict_float(selfvar, othervar)

    def __getitem__(self, key):
        return self.madx.globals[key]

    def __setitem__(self, key, value):
        self.madx.globals[key] = value

    def aperture(self, irn, beam):
        self.madx.exec(f"mk_apir({irn},b{beam})")
        return xd.Table(self.madx.table.aperture)
