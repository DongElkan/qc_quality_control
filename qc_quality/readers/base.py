import dataclasses
import collections
import numpy as np


# XIC
XIC = collections.namedtuple("XIC", ["mz", "tol", "xic"])


@dataclasses.dataclass
class MassSpectrum:
    """
    Mass Spectrum object
    """
    scan: int = None
    spectrum: np.ndarray = None
    rt: float = None
    id: str = None
    precursormz: float = None
    charge: int = None
    collision: str = None
    energy: float = None
    ms_level: int = None

    def get(self, attr):
        return self.__dict__.get(attr)
