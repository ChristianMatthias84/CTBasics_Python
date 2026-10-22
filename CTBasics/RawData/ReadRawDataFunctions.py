import os
import struct
import xml.etree.ElementTree as ElementTree
from pathlib import Path

import numpy as np

_TABLES_ENV = "CTBASICS_RAWDATA_TABLES"

# IRSTBL layout: 104-byte header of int32 fields, then the payload.
_IRSTBL_HEADER_FIELDS = 26
_IRSTBL_DATA_TYPES = {15: np.float32, 16: np.float64}


class _Struct(dict):
    """MATLAB xml2struct result: attribute access over nested dictionaries."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(name) from error


def _ToStruct(element):
    """Converts an XML element the way MATLAB's xml2struct does.

    Attributes land in 'Attributes', the trimmed text in 'Text', and repeated children
    become a list under their tag name.
    """
    node = _Struct()

    if element.attrib:
        node["Attributes"] = _Struct(element.attrib)

    text = (element.text or "").strip()
    if text:
        node["Text"] = text

    for child in element:
        converted = _ToStruct(child)
        if child.tag in node:
            existing = node[child.tag]
            if isinstance(existing, list):
                existing.append(converted)
            else:
                node[child.tag] = [existing, converted]
        else:
            node[child.tag] = converted

    return node


class ReadRawDataFunctions:
    """Access to the scanner tables and the measured raw-data files."""

    _MODE_TYPES = {
        0: "Unknown",
        1: "P45A",
        2: "P46B",
        3: "P2408B",
        4: "P46C",
        5: "P2408C",
        6: "P46A",
        7: "P47A",
        8: "P58A",
        9: "P2408E",
        10: "P46BS",
        11: "P47AS",
        13: "P58A",
        20: "P81A",
        21: "P81B",
        22: "P82A",
        23: "P82B",
        24: "P83A",
        25: "P83B",
        26: "P60A",
        27: "P60B",
        28: "P70A",
        30: "P63A",
        31: "P64A",
    }

    @staticmethod
    def GetModeType(SCI_Model_ID):
        """Converts the numeric SCI model ID into the scanner name."""
        if SCI_Model_ID not in ReadRawDataFunctions._MODE_TYPES:
            raise ValueError("Unknown SCI Model ID")
        return ReadRawDataFunctions._MODE_TYPES[SCI_Model_ID]

    @staticmethod
    def GetNoOfScans(DataDescriptor):
        """Number of scan blocks stored in the file.

        MATLAB reads only the single character in front of '</NoOfScans>', which caps the
        count at nine; the full tag value is read here instead.
        """
        from CTBasics.RawData.RawDataScanDescriptor import RawDataScanDescriptor

        TotalScans = RawDataScanDescriptor.Load(DataDescriptor, "NoOfScans")
        if TotalScans is None:
            raise ValueError("NoOfScans tag not found")
        return int(float(TotalScans))

    @staticmethod
    def GetTablePath(ModelType):
        """Directory of the scanner tables; CTBASICS_RAWDATA_TABLES overrides the default."""
        base = os.environ.get(_TABLES_ENV)
        FileBasePath = Path(base) if base else Path(__file__).parent / "Tables"
        return FileBasePath / ModelType

    @staticmethod
    def LoadBetaTable(ModelType):
        """Fan angles of one scanner model, read from Beta_A.IRSTBL."""
        BetaFileName = ReadRawDataFunctions.GetTablePath(ModelType) / "Beta_A.IRSTBL"
        return ReadRawDataFunctions.readTable(BetaFileName)

    @staticmethod
    def LoadSystemGeometryXML(ModelType):
        """Parses Geometry.xml of one scanner model into an xml2struct-like structure."""
        filename = ReadRawDataFunctions.GetTablePath(ModelType) / "Geometry.xml"
        if not filename.is_file():
            raise FileNotFoundError(f"No scanner geometry for model type '{ModelType}'")

        root = ElementTree.parse(filename).getroot()
        return _Struct({root.tag: _ToStruct(root)})

    @staticmethod
    def readTable(filename):
        """Reads a Siemens IRSTBL table file.

        The 104-byte header stores the payload offset and size as int32, the element type
        code at word 8 and the per-dimension sizes from word 10 on.
        """
        raw = Path(filename).read_bytes()
        if len(raw) < 4 * _IRSTBL_HEADER_FIELDS:
            raise ValueError(f"'{filename}' is too short for an IRSTBL header")

        header = struct.unpack_from(f"<{_IRSTBL_HEADER_FIELDS}i", raw, 0)
        offset, nbytes, typecode = header[4], header[5], header[8]

        if typecode not in _IRSTBL_DATA_TYPES:
            raise ValueError(f"Unsupported IRSTBL data type code {typecode}")

        dtype = _IRSTBL_DATA_TYPES[typecode]
        data = np.frombuffer(raw, dtype=dtype, count=nbytes // np.dtype(dtype).itemsize,
                             offset=offset).astype(float)

        dims = [d for d in header[10:17] if d > 1] or [data.size]

        Table = _Struct()
        Table["data"] = data.reshape(dims) if len(dims) > 1 else data
        Table["dims"] = dims
        Table["type"] = typecode
        return Table

    @staticmethod
    def ReadRawDataScanBlock(FileName, ScanNo):
        raise NotImplementedError(
            "Reading measured scan blocks requires the external readXACB raw-data toolchain"
        )

    @staticmethod
    def ReadRawData(FileName):
        raise NotImplementedError(
            "Reading measured raw-data files requires the external readXACB raw-data toolchain"
        )
