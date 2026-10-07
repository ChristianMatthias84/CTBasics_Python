import numpy as np


class RawDataScanDescriptor:
    """Fills a ScanDescriptor from the descriptors stored inside a raw-data file."""

    @staticmethod
    def LoadFromFile(filename):
        """Returns (ScanDescriptor, Readings) for a measured raw-data file."""
        from .ReadRawDataFunctions import ReadRawDataFunctions

        Readings, _, DataDescriptor = ReadRawDataFunctions.ReadRawData(filename)

        ScanDescriptor = RawDataScanDescriptor.Init(DataDescriptor)
        ScanDescriptor.Filename = filename

        return ScanDescriptor, Readings

    @staticmethod
    def Load(DataDescriptor_in, tag):
        """Reads one tag from the mode parameter XML.

        MATLAB scans line by line and returns the text between the first '>' and the
        first '</' of the line that closes the tag, so a tag must not span lines.
        """
        lines = str(DataDescriptor_in.ModeParXML.String).splitlines()

        for line in lines:
            if f"</{tag}>" in line:
                return line[line.index(">") + 1:line.index("</")]

        return None

    @staticmethod
    def Flip(Readings):
        """Flips every projection along the detector row direction."""
        Readings = np.asarray(Readings, dtype=float).copy()
        Readings[:, :, :] = Readings[:, :, ::-1]
        return Readings

    @staticmethod
    def Init(DataDescriptor_in):
        """Creates the matching fan-beam ScanDescriptor and loads it from the raw data."""
        from ..Descriptors.ScanDescriptor import ScanDescriptor
        from ..Descriptors.ScannerGeometryTypes import ScannerGeometryTypes
        from ..Descriptors.ScanTypes import ScanTypes

        Scans = len(DataDescriptor_in) if isinstance(DataDescriptor_in, (list, tuple)) else 1
        DataDescriptor = DataDescriptor_in[0] if Scans > 1 else DataDescriptor_in

        ScanKind = RawDataScanDescriptor.Load(DataDescriptor, "Kind")

        ScanType = ScanTypes.Spiral if "Spiral" in str(ScanKind) else ScanTypes.Sequence

        ScanDescriptor_out = ScanDescriptor.Create(ScannerGeometryTypes.Fan, ScanType)
        ScanDescriptor_out.Load(DataDescriptor_in)

        return ScanDescriptor_out
