from pathlib import Path

from ctbasics.io import IOHelperFunctions, load_ctbasics_config


def test_load_ctbasics_config_reads_expected_lines(tmp_path: Path):
    config = tmp_path / "CTBasicsConfig.txt"
    repo_root = tmp_path / "repo"
    output_root = tmp_path / "output"
    repo_root.mkdir()
    output_root.mkdir()
    config.write_text(
        "%global folder of the repository\n"
        f"{repo_root}\n"
        "%Output directory\n"
        f"{output_root}\n"
        "%TFS path\n"
        "C:/TFS\n",
        encoding="utf-8",
    )

    global_path, output_path, tfs_path = load_ctbasics_config(config)

    assert global_path == repo_root
    assert output_path == output_root
    assert tfs_path == Path("C:/TFS")


def test_io_helper_contains_and_folder_content(tmp_path: Path):
    folder = tmp_path / "sample"
    subfolder = folder / "sub"
    folder.mkdir()
    subfolder.mkdir()
    (folder / "a.txt").write_text("x", encoding="utf-8")

    files, dirs = IOHelperFunctions.get_folder_content(folder)

    assert files == ["a.txt"]
    assert dirs == ["sub"]
    assert IOHelperFunctions.contains(["Alpha", "Beta"], "beta", ignore_case=True)
    assert IOHelperFunctions.contains("HelloWorld", "world", ignore_case=True)