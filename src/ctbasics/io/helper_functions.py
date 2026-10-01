from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pydicom


def load_ctbasics_config(config_path: Path) -> tuple[Path, Path, Path | None]:
    lines = config_path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 6:
        raise ValueError("CTBasicsConfig.txt must contain at least 6 lines")

    global_path = Path(lines[1].strip())
    output_path = Path(lines[3].strip())
    tfs_path = Path(lines[5].strip()) if lines[5].strip() else None
    return global_path, output_path, tfs_path


@dataclass
class IOHelperFunctions:
    global_path: Path
    global_data_path: Path
    global_output_folder_path: Path

    def __init__(self, config_path: Path | None = None) -> None:
        if config_path is None:
            config_env = Path.cwd().parent / "CTBasicsConfig.txt"
            config_path = config_env if config_env.is_file() else Path.home() / "Documents" / "MATLAB" / "CTBasicsConfig.txt"

        global_path, output_path, _ = load_ctbasics_config(config_path)
        data_path = global_path / "Data"
        if not data_path.is_dir():
            raise FileNotFoundError(f"Data folder does not exist: {data_path}")
        if not output_path.is_dir():
            raise FileNotFoundError(f"Output folder does not exist: {output_path}")

        self.global_path = global_path
        self.global_data_path = data_path
        self.global_output_folder_path = output_path

    @staticmethod
    def get_folder_content(m_path: str | Path) -> tuple[list[str], list[str]]:
        path = Path(m_path)
        file_names: list[str] = []
        dir_names: list[str] = []
        for entry in path.iterdir():
            if entry.is_dir():
                dir_names.append(entry.name)
            else:
                file_names.append(entry.name)
        return file_names, dir_names

    @staticmethod
    def contains(str1: str | Sequence[str], str2: str, ignore_case: bool = False) -> bool:
        if isinstance(str1, str):
            left = str1.lower() if ignore_case else str1
            right = str2.lower() if ignore_case else str2
            return right in left

        if ignore_case:
            needle = str2.lower()
            return any(needle in item.lower() for item in str1)
        return any(str2 in item for item in str1)

    @staticmethod
    def read_dicom_slice(filename: str | Path):
        info = pydicom.dcmread(str(filename), force=True)
        mapped = info.pixel_array.astype("int16")
        hu = mapped * float(getattr(info, "RescaleSlope", 1.0)) + float(getattr(info, "RescaleIntercept", 0.0))
        return hu, info, mapped