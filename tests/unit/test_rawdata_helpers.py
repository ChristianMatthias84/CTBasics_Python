"""Scanner table access and raw-data descriptor helpers.

The IRSTBL and Geometry.xml readers are checked against the tables shipped with the
package, so the binary layout and the XML structure are exercised with real data.
"""
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from CTBasics.RawData.RawDataHelperFunctions import RawDataHelperFunctions
from CTBasics.RawData.RawDataScanDescriptor import RawDataScanDescriptor
from CTBasics.RawData.ReadRawDataFunctions import ReadRawDataFunctions


def test_get_mode_type_maps_known_ids():
    assert ReadRawDataFunctions.GetModeType(24) == "P83A"
    assert ReadRawDataFunctions.GetModeType(31) == "P64A"
    assert ReadRawDataFunctions.GetModeType(0) == "Unknown"


def test_get_mode_type_rejects_unknown_ids():
    with pytest.raises(ValueError):
        ReadRawDataFunctions.GetModeType(99)


def test_get_table_path_can_be_redirected(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("CTBASICS_RAWDATA_TABLES", str(tmp_path))
    assert ReadRawDataFunctions.GetTablePath("P83A") == tmp_path / "P83A"


def test_get_table_path_defaults_to_the_shipped_tables():
    assert (ReadRawDataFunctions.GetTablePath("P60A") / "Beta_A.IRSTBL").is_file()


@pytest.mark.parametrize("ModelType, MFan, betaMax", [
    ("P60A", 840, 0.387398),
    ("P63A", 2752, 0.426177),
    ("P83B", 920, 0.424346),
])
def test_beta_table_holds_a_symmetric_monotone_fan(ModelType, MFan, betaMax):
    Table = ReadRawDataFunctions.LoadBetaTable(ModelType)
    beta = Table.data

    assert beta.shape == (MFan,)
    assert np.all(np.diff(beta) > 0)
    assert beta[0] == pytest.approx(-betaMax, abs=1e-6)
    assert beta[-1] == pytest.approx(betaMax, abs=1e-6)
    assert np.allclose(beta, -beta[::-1], atol=1e-7)


@pytest.mark.parametrize("ModelType, ModuleWidth, NModules", [
    ("P60A", 20, 42),
    ("P63A", 64, 43),
    ("P83B", 20, 46),
])
def test_beta_table_shows_the_detector_module_gaps(ModelType, ModuleWidth, NModules):
    """The channel spacing widens at every detector module border."""
    beta = ReadRawDataFunctions.LoadBetaTable(ModelType).data
    spacing = np.diff(beta)

    widest = spacing.max()
    borders = np.where(spacing > 0.5 * (widest + np.median(spacing)))[0]

    assert len(borders) == NModules - 1
    assert np.array_equal(borders, np.arange(ModuleWidth - 1, beta.size - 1, ModuleWidth))
    assert beta.size == NModules * ModuleWidth


def test_equiangular_scanner_has_no_module_gaps():
    spacing = np.diff(ReadRawDataFunctions.LoadBetaTable("P47A").data)

    # float32 table, so the constant spacing repeats to roughly single precision.
    assert np.std(spacing) < 1e-7


def test_system_geometry_xml_exposes_the_measurement_system():
    Geometry = ReadRawDataFunctions.LoadSystemGeometryXML("P60A")
    MeasurementSystem = Geometry.ScannerGeometry.MeasurementSystem
    MeasurementSystem = MeasurementSystem[0] if isinstance(MeasurementSystem, list) else MeasurementSystem

    RF = float(MeasurementSystem.RadiusFocusPath.Text)
    RFD = float(MeasurementSystem.RadiusFocusDetector.Text)

    assert RF == pytest.approx(610.0)
    assert RFD == pytest.approx(1113.0)
    assert float(MeasurementSystem.AnodeAngle.Text) == pytest.approx(8.0)
    assert Geometry.ScannerGeometry.Scanner.Attributes["Model"] == "P60A"


def test_beta_table_and_geometry_agree_on_the_field_of_measurement():
    """RM derived from the table must stay inside the focus path radius."""
    Geometry = ReadRawDataFunctions.LoadSystemGeometryXML("P60A")
    MeasurementSystem = Geometry.ScannerGeometry.MeasurementSystem
    MeasurementSystem = MeasurementSystem[0] if isinstance(MeasurementSystem, list) else MeasurementSystem

    RF = float(MeasurementSystem.RadiusFocusPath.Text)
    beta = ReadRawDataFunctions.LoadBetaTable("P60A").data

    RM = math.sin(beta[-1]) * RF

    assert 200.0 < RM < RF


def test_unknown_model_has_no_geometry():
    with pytest.raises(FileNotFoundError):
        ReadRawDataFunctions.LoadSystemGeometryXML("P00X")


# ------------------------------------------------------------------ descriptors

def test_raw_data_descriptor_tag_reader():
    descriptor = SimpleNamespace(ModeParXML=SimpleNamespace(
        String="<A>1</A>\n<NoOfScans>3</NoOfScans>\n<B>2</B>"))

    assert RawDataScanDescriptor.Load(descriptor, "NoOfScans") == "3"
    assert RawDataScanDescriptor.Load(descriptor, "Missing") is None


def test_no_of_scans_reads_the_full_tag_value():
    descriptor = SimpleNamespace(ModeParXML=SimpleNamespace(String="<NoOfScans>12</NoOfScans>"))

    assert ReadRawDataFunctions.GetNoOfScans(descriptor) == 12


def test_flip_reverses_the_detector_rows():
    Readings = np.arange(2 * 3 * 4, dtype=float).reshape(2, 3, 4)

    Flipped = RawDataScanDescriptor.Flip(Readings)

    assert np.array_equal(Flipped, Readings[:, :, ::-1])
    assert np.array_equal(Readings, np.arange(2 * 3 * 4).reshape(2, 3, 4))


# --------------------------------------------------------------------- helpers

def test_linearize_readings_runs_the_column_index_fastest():
    Readings = np.arange(2 * 3 * 4, dtype=float).reshape(2, 3, 4)

    Linearized = RawDataHelperFunctions.LinearizeReadings(Readings)

    assert Linearized.shape == (2 * 3 * 4,)
    # First reading, column-major: (c=0,r=0), (c=1,r=0), (c=2,r=0), (c=0,r=1), ...
    assert np.array_equal(Linearized[:3], [Readings[0, 0, 0], Readings[0, 1, 0], Readings[0, 2, 0]])
    assert Linearized[3] == Readings[0, 0, 1]
    assert Linearized[12] == Readings[1, 0, 0]


def test_linearize_readings_rejects_non_3d_input():
    with pytest.raises(ValueError):
        RawDataHelperFunctions.LinearizeReadings(np.zeros((3, 4)))


def test_simple_linearize_array_is_row_major():
    x = np.array([[1.0, 2.0], [3.0, 4.0]])

    assert np.array_equal(RawDataHelperFunctions.SimpleLinearizeArray(x), [1.0, 2.0, 3.0, 4.0])


# ------------------------------------------------------- geometry from raw data

def _ModeParDescriptor(ModelType="P60A", MFan=840, Slices=63):
    return SimpleNamespace(
        ModeParXML=SimpleNamespace(
            Type=SimpleNamespace(ModelType=ModelType),
            ModePar=SimpleNamespace(SliceWidthDMS=600.0, RotTime=0.25, IntegrationTime=250.0),
            String="<FlyingFocalSpot>None</FlyingFocalSpot>",
        ),
        ScanDescr=SimpleNamespace(Type=SimpleNamespace(ChnNum=[MFan], Slices=Slices)),
    )


def test_fan_beam_geometry_loads_from_the_scanner_tables():
    from CTBasics.Descriptors.Geometries.FanBeam import FanBeam

    Geometry = FanBeam().Load(_ModeParDescriptor())

    assert Geometry.RF == pytest.approx(610.0)
    # RD is stored as the focus-detector distance minus the focus path radius.
    assert Geometry.RD == pytest.approx(1113.0 - 610.0)
    assert Geometry.MFan == 840
    assert Geometry.beta.shape == (840,)
    assert Geometry.Ns == 64
    assert Geometry.ds == pytest.approx(0.6)
    assert Geometry.c == pytest.approx(64 * 0.6)
    assert Geometry.RM == pytest.approx(math.sin(Geometry.beta[-1]) * Geometry.RF)
    # 0.25 s rotation at 250 us integration time gives 1000 readings per turn.
    assert Geometry.Sampling.N2pi == 1000
    assert Geometry.FFS.ToString() == "FFS_[1x1]"
