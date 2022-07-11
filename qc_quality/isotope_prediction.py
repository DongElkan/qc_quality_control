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

    Args:
        resolving_power: Resolving power for generating the
            distribution. This is defined at 10% valley.

    """
    def __init__(self, resolving_power=1e6):
        self.rp = resolving_power

        self._elements: Dict[str, Isotope] = dict()
        self._init_elements()

        self.elem_compositions: Optional[Dict[str, int]] = None

    def predict(self,
                formula: str,
                charge: Optional[int] = None,
                ms_mode: str = "centroid") -> Tuple[ndarray, ndarray]:
        """
        Predicts isotopic distribution according to the formula.

        Args:
            formula: Formula of the molecule.
            charge: Charge of ions. If not have, leave it empty.
            ms_mode: Mass spectrometry mode, {`centroid`, `profile`}.
                `centroid`: Centroid mode.
                `profile`: Profile mode.

        Returns:
            Array: m/z of the distribution.
            Array: Intensity of the distribution.

        """
        if ms_mode not in ("centroid", "profile"):
            raise ValueError("Unrecognized ms_mod argument. Expect `centroid`"
                             f" and `profile`, got {ms_mode}.")

        iso_mass, iso_dist = self._predict(formula)

        if charge is None or charge == 0:
            if ms_mode == "centroid":
                return iso_mass, iso_dist
            return self._profile(iso_mass, iso_dist)

        # convert mass to charge
        iso_mz = (iso_mass + (charge * PROTON)) / abs(charge)

        if ms_mode == "centroid":
            return iso_mz, iso_dist
        return self._profile(iso_mz, iso_dist)

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
        for elem, n in self.elem_compositions.items():
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
        for elem, n in self.elem_compositions.items():
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

        fsplits = re.findall(r'[A-Z]|[a-z]|\d+', formula)

        # get element compositions
        elem_composite: Dict[str, int] = collections.defaultdict(int)
        inval_elem = []
        i = 0
        while i < len(fsplits):
            elem = ""
            n = 1
            j = i
            for s in fsplits[i:]:
                if s.isupper() and elem:
                    # has been processed
                    break

                if s.isupper() or s.islower():
                    elem += s
                    j += 1
                elif s.isdigit():
                    n = int(s)
                    j += 1
                    break

            if elem in ELEMENTS:
                elem_composite[elem] += n
            else:
                inval_elem.append(elem)

            i = j

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

    def _profile(self, iso_mass, iso_dist):
        """ Converts to profile mode for output. """
        # for gaussian peak
        u = iso_mass
        s = u / (self.rp * np.sqrt(2 * np.log(10)))
        # lowest intensity is set to 10^-8 of the highest peak intensity
        delta_ms = s * np.sqrt(2 * 8 * np.log(10))

        # isotopic distribution
        if delta_ms.max() > 0.45:
            # this is low resolution
            i0 = iso_mass.argmin()
            i1 = iso_mass.argmax()
            ms_arr = np.linspace(iso_mass[i0] - delta_ms[i0],
                                 iso_mass[i1] + delta_ms[i1], num=2000)
            iso_dist_profiles = np.zeros(ms_arr.size)
            for m, t, sk, dm in zip(iso_mass, iso_dist, s, delta_ms):
                msk = ms_arr - m
                gau_peak = (np.exp(- msk * msk / (2 * sk * sk))
                            / (sk * np.sqrt(2 * np.pi)))
                iso_dist_profiles += gau_peak / gau_peak.max() * t
            return ms_arr, iso_dist_profiles

        iso_dist_profiles = []
        iso_mass_profile = []
        for m, t, sk, dm in zip(iso_mass, iso_dist, s, delta_ms):
            ms_arr = np.linspace(m - dm, m + dm, num=200)
            msk = ms_arr - m
            gau_peak = (np.exp(- msk * msk / (2 * sk * sk))
                        / (sk * np.sqrt(2 * np.pi)))
            iso_dist_profiles.append(gau_peak / gau_peak.max() * t)
            iso_mass_profile.append(ms_arr)

        return (np.concatenate(iso_mass_profile, axis=0),
                np.concatenate(iso_dist_profiles, axis=0))


# if __name__ == "__main__":
#     form = "SC112H165CN27O36"
#     t = time.time()
#     iso_predictor = IsotopeDistribution()
#     print(f"{'%.8f' % (time.time() - t)} secs.")
    # profile.run("_ = iso_predictor.predict(form)")
    # mx, dx = iso_predictor.predict(form)
    # for ii in range(mx.size):
    #     print(f"{'%.6f' % mx[ii]} {'%.6f' % dx[ii]}")
