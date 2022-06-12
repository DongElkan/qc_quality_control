import dataclasses
import collections
import numpy as np

from typing import Optional


# XIC
XIC = collections.namedtuple("XIC", ["mz", "tol", "rt", "intensity"])


@dataclasses.dataclass
class MassSpectrum:
    """
    Mass Spectrum object
    """
    scan: Optional[int] = None
    mz: Optional[np.ndarray] = None
    intensity: Optional[np.ndarray] = None
    rt: Optional[float] = None
    id: Optional[str] = None
    precursormz: Optional[float] = None
    charge: Optional[int] = None
    collision: Optional[str] = None
    energy: Optional[float] = None
    ms_level: Optional[int] = None
    peak_num: Optional[int] = None
