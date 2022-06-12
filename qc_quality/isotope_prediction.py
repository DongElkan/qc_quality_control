"""
This module provides a class for prediction of isotopic distribution.

"""
import re
import collections
import numpy as np

from numpy import ndarray
from typing import Dict, Tuple, Optional

from .constants import ELEMENTS, PROTON, Isotope
from .utilities import (multinomial_coef_log,
                        element_isotope_dist2,
                        element_isotope_distn,
                        element_dist_combine,
                        centroid_mass,
                        dot)
from .multipermute import partitions, permutation


class IsotopeDistribution:
    """
    This class predicts isotopic distribution and other methods.

    """
    def __init__(self, resolving_power=1e6):
        self.rp = resolving_power

        self._elements: Dict[str, Isotope] = dict()
        self._init_elements()

        self.elem_compositions: Optional[Dict[str, int]] = None

    def predict(self, formula: str, charge: Optional[int] = None)\
            -> Tuple[ndarray, ndarray]:
        """
        Predicts isotopic distribution according to the formula.

        Args:
            formula: Formula of the molecule.
            charge: Charge of ions. If not have, leave it empty.

        Returns:
            Array: m/z of the distribution.
            Array: Intensity of the distribution.

        """
        iso_mass, iso_dist = self._predict(formula)

        if charge is None:
            return iso_mass, iso_dist

        # convert mass to charge
        iso_mz = iso_mass / charge + PROTON

        return iso_mz, iso_dist

    def monoisotopic_mass(self) -> float:
        """
        Calculates monoisotopic mass of the molecule.

        Returns:
            float: Monoisotopic mass.

        """
        if self.elem_compositions is None:
            raise ValueError("Elemental compositions are not obtained. "
                             "Call IsotopeDistribution.predict first.")

        m: float = 0.
        for elem, n in self.elem_compositions.keys():
            iso = self._elements[elem]
            i = iso.prob.argmax()
            m += iso.mass[i] * n

        return m

    def average_mass(self) -> float:
        """
        Calculates average mass of the molecule.

        Returns:
            float: Average mass.

        """
        if self.elem_compositions is None:
            raise ValueError("Elemental compositions are not obtained. "
                             "Call IsotopeDistribution.predict first.")

        m: float = 0.
        for elem, n in self.elem_compositions.keys():
            iso = self._elements[elem]
            avg_m = dot(iso.mass, iso.prob) / iso.prob.sum()
            m += avg_m * n

        return m

    def _init_elements(self):
        """
        Initializes elements' isotope arrays.

        """
        for elem, isos in ELEMENTS.items():
            mass = np.fromiter([iso.mass for iso in isos], np.float64)
            probs = np.fromiter([iso.prob for iso in isos], np.float64)
            if len(isos) == 1:
                self._elements[elem] = Isotope(mass, probs)
            else:
                # make sure is sorted
                ix = mass.argsort()
                self._elements[elem] = Isotope(mass[ix], probs[ix])

    def _predict(self, formula: str) -> Tuple[ndarray, ndarray]:
        """
        Predicts isotopic distribution

        Args:
            formula: Formula.

        Returns:
            Array: m/z of the distribution.
            Array: Intensity of the distribution.

        """
        # elemental compositions
        self._parse_formula(formula)

        # elemental distributions
        tmp_mass = []
        tmp_int = []
        for elem, n in self.elem_compositions.items():
            iso = self._elements[elem]
            print(elem, iso)
            if iso.mass.size == 2:
                m_dist, i_dist = element_isotope_dist2(iso.mass, iso.prob, n)
            elif iso.mass.size == 1:
                m_dist = np.fromiter([n * iso.mass], np.float64)
                i_dist = np.fromiter([1.], np.float64)
            else:
                m_dist, i_dist = self._calculate_multi_iso_dist(
                    iso.mass, iso.prob, n)
            tmp_mass.append(m_dist)
            tmp_int.append(i_dist)

        # combine elemental distributions
        mass0 = tmp_mass[0]
        int0 = tmp_int[0]
        for m, t in zip(tmp_mass[1:], tmp_int[1:]):
            mass0, int0 = element_dist_combine(mass0, int0, m, t)

        # group the mass
        return centroid_mass(mass0, int0)

    def _parse_formula(self, formula: str):
        """
        Parses formula to get number of elements.

        Args:
            formula: Formula.

        Returns:
            dict: Elements with number

        Raises:
            ValueError

        """
        # non-number or character
        inval_char = re.findall(r"[^0-9a-zA-Z]", formula)
        if inval_char:
            raise ValueError(
                f"Unrecognized characters: {', '.join(inval_char)}. Formula "
                "should only contain characters and numbers."
            )

        fsplits = re.findall(r'(\w+?)(\d+)', formula)

        # get element compositions
        elem_composite: Dict[str, int] = collections.defaultdict(int)
        inval_elem = []
        for s, sn in fsplits:
            if s in ELEMENTS:
                elem_composite[s] += int(sn)
                continue

            if len(s) == 1:
                inval_elem.append(s)
            else:
                ss = []
                sj = []
                for sk in s:
                    if sk.islower():
                        sj.append(sk)
                    else:
                        if sj:
                            ss.append("".join(sj))
                            sj.clear()
                        sj.append(sk)

                ss.append("".join(sj))
                for i, sk in enumerate(ss):
                    if sk in ELEMENTS:
                        elem_composite[sk] += int(sn) if i == len(ss)-1 else 1
                    else:
                        inval_elem.append(sk)

        # raise if exist invalid elements
        if inval_elem:
            raise ValueError(f"Unrecognized elements: {','.join(inval_elem)}.")

        self.elem_compositions = elem_composite

    @staticmethod
    def _calculate_multi_iso_dist(mass: ndarray, prob: ndarray, n: int)\
            -> Tuple[ndarray, ndarray]:
        """
        Calculates isotopic distribution for elements with more than 2
        isotopes.

        Args:
            mass: Mass of isotopes.
            prob: Probability of isotopes.
            n: Number of atoms.

        Returns:
            Array: Mass
            Array: Isotopic intensity

        """
        k = mass.size
        # partitions
        parts = partitions(n, k)

        # convert the lists to arrays
        parts2 = np.zeros((len(parts), k), dtype=np.int64)
        for i, p in enumerate(parts):
            parts2[i] = np.fromiter(p, np.int64)

        # calculate multinomial coefficients, log transformed
        mcoefs = multinomial_coef_log(parts2, n)

        # permute the mass and probability arrays for distribution
        perms = permutation(np.fromiter(range(k), np.int64))
        mass_perms = np.zeros((len(perms), k), dtype=np.float64)
        prob_perms = np.zeros((len(perms), k), dtype=np.float64)
        for i, ix in enumerate(perms):
            mass_perms[i] = mass[ix]
            prob_perms[i] = prob[ix]

        # calculate distributions
        return element_isotope_distn(mcoefs, mass_perms, prob_perms, parts2)


# if __name__ == "__main__":
#     form = "SC112H165CN27O36"
#     t = time.time()
#     iso_predictor = IsotopeDistribution()
#     print(f"{'%.8f' % (time.time() - t)} secs.")
    # profile.run("_ = iso_predictor.predict(form)")
    # mx, dx = iso_predictor.predict(form)
    # for ii in range(mx.size):
    #     print(f"{'%.6f' % mx[ii]} {'%.6f' % dx[ii]}")
