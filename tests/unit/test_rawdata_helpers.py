from types import SimpleNamespace
from pathlib import Path

from ctbasics.rawdata import ReadRawDataFunctions, RawDataScanDescriptor


def test_get_mode_type_maps_known_ids():
    assert ReadRawDataFunctions.get_mode_type(24) == "P83A"
    assert ReadRawDataFunctions.get_mode_type(31) == "P64A"


def test_get_table_path_joins_tables_folder(tmp_path: Path):
    assert ReadRawDataFunctions.get_table_path("P83A", tmp_path) == tmp_path / "Tables" / "P83A"


def test_raw_data_descriptor_tag_reader():
    descriptor = SimpleNamespace(ModeParXML=SimpleNamespace(String="<A>1</A>\n<NoOfScans>3</NoOfScans>\n<B>2</B>"))

    assert RawDataScanDescriptor.load(descriptor, "NoOfScans") == "3"