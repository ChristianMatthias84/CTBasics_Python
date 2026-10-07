"""Startup check for the CTBasics configuration.

Port of init_CTBasics.m. MATLAB uses this file mainly to extend the MATLAB search path;
in Python the package is importable after ``pip install -e .``, so only the parts with
real behaviour remain: the presence check for CTBasicsConfig.txt and the validation of
the optional TFS toolbox path used by the raw-data projects.
"""
from pathlib import Path

from .ReadAndWrite.IOHelperFunctions import IOHelperFunctions

# MATLAB guards the TFS path setup with this flag; it is only needed to load raw data.
IncludeRawDataProjects = True


def init_CTBasics(ConfigFile=None, IncludeRawDataProjects=IncludeRawDataProjects):
    """Validates the configuration and returns the resolved CTBasics paths."""
    aIOHelper = IOHelperFunctions(ConfigFile)

    Paths = {
        "GlobalPath": aIOHelper.GlobalPath,
        "GlobalDataPath": aIOHelper.GlobalDataPath,
        "GlobalOutputFolderPath": aIOHelper.GlobalOutputFolderPath,
        "TfsPath": None,
    }

    if IncludeRawDataProjects:
        TfsPath = aIOHelper.TfsPath
        if TfsPath is None or not Path(TfsPath).is_dir():
            raise NotADirectoryError("Invalid TFS path!")

        Paths["TfsPath"] = Path(TfsPath)
        # MATLAB adds general, mex, mexSources and MatlabDicom to the search path here.
        # Python has no equivalent step, so the subfolders are only reported.
        Paths["TfsSubFolders"] = [Path(TfsPath) / name for name in
                                  ("general", "mex/Win32/Release", "mex", "mexSources",
                                   "MatlabDicom")]

    return Paths
