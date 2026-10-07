from pathlib import Path

import numpy as np
import pytest

from CTBasics.Memory.Precision import Precision
from CTBasics.ReadAndWrite import IOHelperFunctions


@pytest.fixture
def configured(tmp_path, monkeypatch):
    """Creates a CTBasicsConfig.txt plus the Data/Output folders the constructor requires."""
    repo = tmp_path / "repo"
    (repo / "Data").mkdir(parents=True)
    output = tmp_path / "out"
    output.mkdir()

    config = tmp_path / "CTBasicsConfig.txt"
    config.write_text(
        "# global path\n"
        f"{repo}\n"
        "# output path\n"
        f"{output}\n"
        "# tfs path\n"
        "\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("CTBASICS_CONFIG", str(tmp_path))
    return repo, output


def test_constructor_reads_the_matlab_config_lines(configured):
    repo, output = configured

    io = IOHelperFunctions()

    assert io.GlobalPath == repo
    assert io.GlobalDataPath == repo / "Data"
    assert io.GlobalOutputFolderPath == output


def test_constructor_reports_a_missing_output_folder(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    (repo / "Data").mkdir(parents=True)

    config = tmp_path / "CTBasicsConfig.txt"
    config.write_text(f"#\n{repo}\n#\n{tmp_path / 'missing'}\n#\n\n", encoding="utf-8")
    monkeypatch.setenv("CTBASICS_CONFIG", str(tmp_path))

    with pytest.raises(FileNotFoundError, match="output folder"):
        IOHelperFunctions()


def test_save_and_load_round_trip_through_the_output_folder(configured):
    volume = np.arange(12, dtype=float).reshape(3, 4) * 0.5

    IOHelperFunctions.Save(volume, "volume.bin")
    loaded = IOHelperFunctions.Load("volume.bin")

    assert loaded.dtype == np.float64
    assert np.array_equal(loaded, volume.ravel())
    assert Precision.Value() == "double"


def test_save_rejects_a_non_binary_extension(configured):
    with pytest.raises(ValueError, match="Invalid file-extension"):
        IOHelperFunctions.Save(np.zeros(4), "volume.raw")


def test_load_reports_a_missing_file(configured):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        IOHelperFunctions.Load("absent.bin")


def test_get_folder_content_separates_files_and_directories(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    (tmp_path / "b.txt").write_text("b", encoding="utf-8")

    fileNames, dirNames = IOHelperFunctions.get_folder_content(tmp_path)

    assert fileNames == ["a.txt", "b.txt"]
    assert dirNames == ["sub"]


def test_contains_follows_the_matlab_ignore_case_signature():
    assert IOHelperFunctions.contains(["Alpha", "Beta"], "beta", "IgnoreCase", True)
    assert IOHelperFunctions.contains("HelloWorld", "world", "IgnoreCase", True)

    # Without the explicit 'IgnoreCase' marker the comparison stays case sensitive.
    assert not IOHelperFunctions.contains("HelloWorld", "world")
    assert not IOHelperFunctions.contains(["Alpha"], "beta", "IgnoreCase", True)


def test_get_all_dicom_folders_finds_nested_image_folders(tmp_path):
    nested = tmp_path / "patient" / "series"
    nested.mkdir(parents=True)
    (nested / "image1.IMA").write_bytes(b"")
    (tmp_path / "patient" / "notes.txt").write_bytes(b"")

    folders = IOHelperFunctions.get_all_dicom_folders(tmp_path)

    assert [Path(f) for f in folders] == [nested]


def test_create_circular_mask_matches_the_analytic_area():
    radius, resolution = 20, 101
    mask = IOHelperFunctions.create_circular_mask(radius, (51, 51), resolution)

    assert mask[50, 50]
    assert not mask[50, 50 + radius + 1]
    assert abs(mask.sum() - np.pi * radius ** 2) / (np.pi * radius ** 2) < 0.01


def test_center_of_gravity_finds_an_offset_blob():
    image = np.zeros((101, 101))
    image[59:64, 39:44] = 1.0

    x, y = IOHelperFunctions.center_of_gravity(image, (51, 51), 40)

    # create_circular_mask returns [x, y] = [column, row] like MATLAB's WeightedCentroid.
    assert abs(x - 42.0) < 1.0
    assert abs(y - 62.0) < 1.0


def test_get_center_locates_a_centered_object():
    image = np.zeros((128, 128))
    image[40:90, 30:100] = 1.0

    center = IOHelperFunctions.getCenter(image)

    assert abs(int(center[0]) - 65) <= 2
    assert abs(int(center[1]) - int(round(65 * 0.98))) <= 2


def test_encode_private_header_reads_the_diameter_fields():
    header = b"000000.600 000253.749 000242.663 000002.000"

    horiz, vert = IOHelperFunctions.encode_private_header(header)

    assert horiz == pytest.approx(253.749)
    assert vert == pytest.approx(242.663)


def test_read_pfo_info_detects_the_negative_markers():
    assert IOHelperFunctions.readPFOinfo(None, "C:/scans/head_PFO") == "PFO"
    assert IOHelperFunctions.readPFOinfo(None, "C:/scans/head_noPFO") == "noPFO"
    assert IOHelperFunctions.readPFOinfo(None, "C:/scans/head_NOIBHC") == "noPFO"


def test_extract_series_description_needs_a_unique_match():
    tissue, diam = IOHelperFunctions.extractSeriesDescription(
        "C:/scans/Head_Brain_200mm", ["Brain", "Adipose"], ["200mm", "300mm"]
    )
    assert (tissue, diam) == ("Brain", "200mm")

    tissue, diam = IOHelperFunctions.extractSeriesDescription(
        "C:/scans/Brain_Adipose_200mm", ["Brain", "Adipose"], ["200mm"]
    )
    assert np.isnan(tissue) and np.isnan(diam)


def test_save_double_to_file_is_matlab_fwrite_compatible(tmp_path):
    target = tmp_path / "data.bin"
    IOHelperFunctions.SaveDoubleToFile(target, np.arange(5, dtype=np.float32))

    assert np.array_equal(np.fromfile(target, dtype=np.float64), np.arange(5))


def test_export_slices_writes_one_png_per_slice(tmp_path):
    volume = np.linspace(0.0, 1.0, 2 * 3 * 4).reshape(2, 3, 4)

    IOHelperFunctions.ExportSlices(volume, tmp_path)

    assert sorted(p.name for p in tmp_path.glob("*.png")) == [
        "Slice_1.png", "Slice_2.png", "Slice_3.png", "Slice_4.png"
    ]
