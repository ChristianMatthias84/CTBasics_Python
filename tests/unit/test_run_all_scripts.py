"""Unit tests for the top level entry points (RunAllScripts, init_CTBasics)."""
from pathlib import Path

import pytest

import RunAllScripts
from CTBasics.init_CTBasics import init_CTBasics


def _WriteConfig(directory, TfsPath):
    dataPath = directory / "Data"
    dataPath.mkdir()
    outputPath = directory / "Out"
    outputPath.mkdir()

    configFile = directory / "CTBasicsConfig.txt"
    configFile.write_text("\n".join([
        "GlobalPath", str(directory),
        "OutputPath", str(outputPath),
        "TfsPath", str(TfsPath),
    ]), encoding="utf-8")
    return configFile


def test_run_all_scripts_matches_matlab_order():
    # RunAllScripts.m runs these thirteen scripts in exactly this order.
    assert len(RunAllScripts.SCRIPTS) == 13
    assert RunAllScripts.SCRIPTS[0].endswith("TestScript2D")
    assert RunAllScripts.SCRIPTS[6].endswith("TestMAR")
    assert RunAllScripts.SCRIPTS[10].endswith("StaticCT_JP_HeartRateVsNSubsets_Sweep")
    assert RunAllScripts.SCRIPTS[-1].endswith("StaticCT_4_RealTimeImaging_Script_CH")
    # The commented out raw data script must not sneak back in.
    assert not any("TestScriptScannerData3D" in name for name in RunAllScripts.SCRIPTS)


def test_run_all_scripts_imports_every_listed_module():
    import importlib

    for name in RunAllScripts.SCRIPTS:
        module = importlib.import_module(name)
        assert callable(module.main), name


def test_run_all_scripts_stops_on_error():
    with pytest.raises(ModuleNotFoundError):
        RunAllScripts.main(Scripts=["CTBasics.DoesNotExist"])


def test_run_all_scripts_can_continue_on_error():
    results = RunAllScripts.main(Scripts=["CTBasics.DoesNotExist"], StopOnError=False)

    assert isinstance(results["CTBasics.DoesNotExist"], ModuleNotFoundError)


def test_init_ctbasics_resolves_paths(tmp_path):
    tfsPath = tmp_path / "tfs"
    tfsPath.mkdir()
    configFile = _WriteConfig(tmp_path, tfsPath)

    paths = init_CTBasics(str(configFile))

    assert paths["GlobalPath"] == Path(tmp_path)
    assert paths["GlobalDataPath"] == Path(tmp_path) / "Data"
    assert paths["TfsPath"] == tfsPath
    assert len(paths["TfsSubFolders"]) == 5


def test_init_ctbasics_rejects_invalid_tfs_path(tmp_path):
    configFile = _WriteConfig(tmp_path, tmp_path / "missing")

    with pytest.raises(NotADirectoryError, match="Invalid TFS path!"):
        init_CTBasics(str(configFile))


def test_init_ctbasics_skips_tfs_check_without_raw_data_projects(tmp_path):
    configFile = _WriteConfig(tmp_path, tmp_path / "missing")

    paths = init_CTBasics(str(configFile), IncludeRawDataProjects=False)

    assert paths["TfsPath"] is None


def test_init_ctbasics_requires_a_config_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        init_CTBasics(str(tmp_path / "CTBasicsConfig.txt"))
