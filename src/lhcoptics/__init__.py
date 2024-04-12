"""
LHCModel -> Compute twiss, aperture, match
LHCOptics -> Contains complete optics
LHCIROptics -> Contains strengths, constraints, knobs
LHCArcOptics -> Contains strengths, constraints, knobs
"""

__version__ = "0.0.0"
from .optics import LHCOptics
from .model_xsuite import LHCXsuiteModel, LHCMadModel
from .circuits import LHCCircuits
