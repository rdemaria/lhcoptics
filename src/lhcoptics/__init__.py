"""
LHCModel -> Compute twiss, aperture, match
LHCOptics -> Contains complete optics
LHCIR(s) -> Contains strengths, constraints, knobs
LHCArc -> Contains strengths, constraints, knobs
"""

__version__ = "0.0.3"
from .arcs import LHCArc
from .circuits import LHCCircuits
from .lsa_util import get_lsa
from .nxcals_util import get_spark, get_nxcals, NXCals
from .ir1 import LHCIR1
from .ir2 import LHCIR2
from .ir3 import LHCIR3
from .ir4 import LHCIR4
from .ir5 import LHCIR5
from .ir6 import LHCIR6
from .ir7 import LHCIR7
from .ir8 import LHCIR8
from .knob import Knob
from .model_madx import LHCMadxModel
from .model_xsuite import LHCXsuiteModel
from .optics import LHCOptics
from .opttable import LHCOpticsTable, LHCIRTable, LHCArcTable
from .rdmsignal import poly_fit, poly_val
from .repo import LHC
from .aperture import LHCAperture

__all__ = [
    get_lsa,
    get_spark,
    get_nxcals,
    Knob,
    LHC,
    LHCAperture,
    LHCArc,
    LHCCircuits,
    LHCIR1,
    LHCIR2,
    LHCIR3,
    LHCIR4,
    LHCIR5,
    LHCIR6,
    LHCIR7,
    LHCIR8,
    LHCIRTable,
    LHCArcTable,
    LHCMadxModel,
    LHCOptics,
    LHCOpticsTable,
    LHCXsuiteModel,
    poly_fit,
    poly_val,
]
