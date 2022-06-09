import base64
import zlib
import struct
import collections
import numpy as np

from .base import MassSpectrum, XIC
from .utilities import match_array, match_index

from typing import Callable, Optional, Union, List, Dict, Any

try:
    import xml.etree.cElementTree as et
except ImportError:
    import xml.etree.ElementTree as et


NAME_PAIR = [
    ("scan", "index"),
    ("mz", "mz"),
    ("intensity", "intensity"),
    ("id", "id"),
    ("rt", "scan start time"),
    ("ms_level", "ms level"),
    ("precursormz", "selected ion m/z"),
    ("charge", "charge state"),
    ("collision", "collision"),
    ("energy", "energy")
]


def _decode_binary(string: str,
                   array_length: int,
                   precision: int = 64,
                   decompress: Optional[Callable] = zlib.decompress):
    """
    Decode binary string to float points.
    """
    decoded = base64.b64decode(string)
    if decompress is not None:
        decoded = decompress(decoded)

    int_type = "d" if precision == 64 else "L"

    return struct.unpack(f"<{array_length}{int_type}", decoded)


class _ParseSpectrumElement:
    """
    Parse spectrum tree to get all records for easy access
    """
    def __init__(self, element):
        self._element = element
        self._info: Dict[str, Any] = dict()

        self._get_collision()
        self._get_spectrum()

        self._update()

    def _update(self):
        """ Updates the attributes. """
        self._info["id"] = self._element.get("id")
        self._info["index"] = int(self._element.get("index"))
        self.__dict__.update(self._info)

    def _get_spectrum(self):
        """ Decodes spectrum. """
        array_len = int(self._element.get("defaultArrayLength"))

        spec = {}
        for array in self._element.iter(tag="binaryDataArray"):
            params = self._get_param(array)
            for m, _ in params:
                if m.endswith("array"):
                    array_name = m
                elif m.endswith("compression"):
                    compress = m
                elif m.endswith("float"):
                    bit = m
            bit = 64 if bit.startswith("64") else 32

            binary = array.find("binary")
            if compress.startswith("no "):
                array_data = _decode_binary(
                    binary.text, array_len, precision=bit, decompress=None)
            else:
                array_data = _decode_binary(
                    binary.text, array_len, precision=bit)
            spec[array_name] = array_data

        if spec:
            intens = np.fromiter(spec["intensity array"], np.float64)
            ix = intens > 0
            mz = np.fromiter(spec["m/z array"], np.float64)[ix]
            intens = intens[ix]
            # make sure m/z in ascending order
            ix = mz.argsort()
            self._info["mz"] = mz[ix]
            self._info["intensity"] = intens[ix]
        else:
            self._info["mz"] = []
            self._info["intensity"] = []

    def _get_collision(self):
        """ Gets the collision information """
        org_params = []
        for activation in self._element.iter(tag="activation"):
            org_params += self._get_param(activation)

        if not org_params:
            return

        params = []
        for name, val in org_params:
            if "energy" in name:
                params.append(("energy", val))
            elif not val:
                params.append(("collision", name))

        self._info.update(params)

    @staticmethod
    def _get_param(subelement):
        """ Get parameter from the element """
        params = []
        for param in subelement.iter(tag="cvParam"):
            name = param.get("name")
            val = param.get("value")
            if val is not None:
                try:
                    val = float(val)
                    if val.is_integer():
                        val = int(val)
                except ValueError:
                    pass
            params.append((name, val))
        return params


class mzMLReader:
    """
    This class reads mzML file.

    Args:

    """
    def __init__(self, mzmlfile: str):
        self.mzml_file = mzmlfile
        if self._check_offset():
            self._offsets = self._get_scan_offsets()
        else:
            self._offsets = self._get_offsets_from_file()

        self._ms_levels: Optional[np.ndarray] = None
        self._ms1_index: Optional[np.ndarray] = None
        self._spectra: Optional[List[MassSpectrum]] = None

        self._read_spectrum()
        self._get_ms_level()

    def get_mass_spectrum(self,
                          index: Optional[Union[np.ndarray, list]] = None,
                          mslevel: Optional[int] = None) -> List[MassSpectrum]:
        """
        Gets tandem mass spectra according to the input scans.
        If the no scan is specified, all tandem mass spectra
        are output.

        Args:
            index: Index of spectra extracted. If is set to None,
                all mass spectra are output.
            mslevel: Mass spectrum level in the experiment. If is
                set to None, all mass spectra specified by index
                are output, regardless of the level.
        """
        if mslevel is not None:
            idx = match_index(self._ms_levels, mslevel)
            if idx.size == 0:
                raise ValueError("The ms_level does not exist.")

            if index is None:
                return [self._spectra[i] for i in idx]

            excepts = set(index).difference(idx)
            if excepts:
                raise ValueError("The indices "
                                 f"{', '.join(str(i) for i in excepts)}"
                                 f" do not match the ms_level {mslevel}")
            return [self._spectra[i] for i in sorted(index)]
        return [self._spectra[i] for i in sorted(index)]

    def get_xic(self, mz: float, tol: float = 0.1, unit: str = "Da") -> XIC:
        """
        Gets MS1 according to the input scans.

        Args:
            mz: m/z value for extracting XIC.
            tol: Tolerance for extracting XIC.
            unit: Unit of tolerance. {`Da`, `ppm`}. Defaults to "Da".

        Returns:
            XIC

        """
        t = tol
        if unit == "ppm":
            t = mz / 1e6 * tol
        return self._xic(mz, t)

    def _get_ms_level(self):
        """ Gets mass spectrum levels. """
        self._ms_levels = np.fromiter([s.ms_level for s in self._spectra],
                                      np.int64)
        ms1_idx = match_index(self._ms_levels, 1)
        if ms1_idx.size > 0:
            self._ms1_index = ms1_idx

    def _get_scan_offsets(self, block_size=10240, fromstring=et.fromstring):
        """
        Gets offsets. Use file.seek and file.read to get the offset
        block directly.
        """
        offsets = collections.defaultdict()

        # read the index block from the end of the file
        f = open(self.mzml_file, "rb")
        locate = file_size = f.seek(0, 2)
        nblks = int(file_size / block_size)
        for i in range(1, nblks):
            locate -= block_size
            f.seek(locate)
            line = f.read(block_size)
            if b'<index name="spectrum"' in line:
                break

        # get the offsets in this block
        f.seek(locate)
        index = -1
        for line in f:
            if line.lstrip().startswith(b"<offset "):
                el = fromstring(line)
                index += 1
                offsets[index] = int(el.text)

        return offsets

    def _get_offsets_from_file(self, fromstring=et.fromstring):
        """ Gets offsets from the file by reading the spectrum. """
        # get the number of spectra in the mzML file
        with open(self.mzml_file, "rb") as f:
            c = f.tell()
            for line in iter(f.readline, b""):
                if line.lstrip().startswith(b"<spectrumList "):
                    s = line.rstrip()
                    if not s.endswith(b"/>"):
                        s = s[:-1].decode('ascii')
                    el = fromstring("".join([s, "/>"]))
                    n = int(el.get("count")) - 1
                    break
                c = f.tell()

        # get the offsets
        offsets, index = collections.defaultdict(), None
        with open(self.mzml_file, "rb") as f:
            c = f.seek(c)
            for line in iter(f.readline, b""):
                if line.lstrip().startswith(b"<spectrum "):
                    el = fromstring(
                        "".join([line.decode('ascii'), "</spectrum>"])
                    )
                    index = int(el.get("index"))
                    offsets[index] = c
                elif index == n and line.startswith(b"</spectrumList>"):
                    break
                c = f.tell()
        offsets[index + 1] = c

        return offsets

    def _read_spectrum(self, fromstring=et.fromstring):
        """ Gets mass spectra. """
        offsets = self._offsets
        indices = sorted(offsets.keys())

        f = open(self.mzml_file, "rb")

        # get spectra
        spectra = []
        for i0, i1 in zip(indices[:-1], indices[1:]):
            start, end = offsets.get(i0), offsets.get(i1)

            # get spectrum element
            f.seek(start, 0)
            text = f.read(end - start)
            # correct offsets to make sure spectrum bounds
            text_split = text.splitlines()
            # stop if the block does not start from spectrum
            if text_split[0].lstrip().startswith(b"<spectrum "):
                if not text_split[-1].endswith(b"</spectrum>"):
                    i = len(text_split)
                    while i > 0:
                        i -= 1
                        if text_split[i].endswith(b"</spectrum>"):
                            break
                    text = b"\n".join(text_split[:i + 1])
                blk = fromstring(text)

                # parse to get spectrum or XIC
                spectrum = self._parse_spectrum(blk)
                if spectrum is not None:
                    spectra.append(spectrum)

        f.close()

        self._spectra = spectra
        return self

    @staticmethod
    def _parse_spectrum(spectrum_tree):
        """ Get spectrum and XIC """
        spec = _ParseSpectrumElement(spectrum_tree)
        return MassSpectrum(
            **{fd: getattr(spec, m, None) for fd, m in NAME_PAIR})

    def _xic(self, mz, tol):
        """ Generates XIC according to matched peaks. """
        if self._ms1_index is None:
            raise ValueError("Can't find MS1.")

        rt = []
        intens = []
        for i in self._ms1_index:
            spec = self._spectra[i]
            if spec.mz:
                i0, i1 = match_array(spec.mz, mz, tol)
                intens.append(0. if i0 == i1
                              else spec.intensity[i0:i1].max())
            else:
                intens.append(0.)
            rt.append(spec.rt)
        return XIC(mz=mz,
                   tol=tol,
                   rt=np.fromiter(rt, np.float64),
                   intensity=np.fromiter(intens, np.float64))

    def _check_offset(self, block_size=10240):
        """ Checks whether there exists offset for fast file access.
        """
        f = open(self.mzml_file, "rb")
        locate = f.seek(0, 2)
        # number of blocks
        nblks = int(locate / block_size) + 1
        # get the existence of the tag
        t = False
        for i in range(1, nblks):
            locate -= block_size
            f.seek(locate)
            line = f.read(block_size)

            # if the offset tag is found, return True
            if b"<offset " in line:
                t = True
                break
            elif b"</spectrumList>" in line:
                # if the block goes to the spectrum list part,
                # return False
                break

        # get the tail
        if not t and locate <= block_size:
            f.seek(0)
            line = f.read(locate)
            if b"<offset " in line:
                t = True
        f.close()

        return t
