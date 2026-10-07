"""Metal artifact reduction on a hip slice.

Port of LecturesAndTestScripts/TestMAR.m. MATLAB loads a DICOM slice from the data
folder; because that data is not part of the repository the simulated phantom branch
(MATLAB's `if(false)` branch) is the default here.
"""
import numpy as np

from ..Apps.MAR import MAR
from ..ImageManipulation.ImageManipulation import ImageManipulation
from ..ReadAndWrite.IOHelperFunctions import IOHelperFunctions
from ..Reconstruction.Constants import Constants
from ..Simulation.CreatePhantom import CreatePhantom


def main(UsePatientData=False, DataDirectory=None, Nx=256, Ny=256, Npar=512, M=500,
         SelectedPhantom="HipWithMetal"):
    RM = 250.0          # Radius of the field of measurement (FOM).
    thetaN = np.pi
    N2pi = 2 * Npar

    dxi = 2 * RM / (M - 1)
    xi0 = -RM - dxi
    theta0 = 0.0
    dtheta = thetaN / Npar

    dx = 2 * RM / Nx
    dy = 2 * RM / Ny
    xc = 0.0
    yc = 0.0
    x0 = -RM - dx / 2.0 + xc
    y0 = -RM - dy / 2.0 + yc

    if not UsePatientData:
        Voxelphantom, isSpectral = CreatePhantom.CreateVoxelPhantom2D(
            Nx, dx, x0, Ny, dy, y0, SelectedPhantom
        )
        # The MAR algorithm works on HU, the phantom is given as attenuation.
        Voxelphantom = (Voxelphantom - Constants.MuW) / Constants.MuW * 1000
    else:
        from pathlib import Path

        if DataDirectory is None:
            DataDirectory = IOHelperFunctions().GlobalDataPath
        filename = Path(DataDirectory) / "MAR2D" / "hip2.ima"

        isSpectral = False
        V_HU, V_info, V_mapped = IOHelperFunctions.read_Dicom_Slice(str(filename))
        V_HU = np.asarray(V_HU, dtype=float)
        V_HU = ImageManipulation.Resample2D(V_HU, Nx, Ny)
        Voxelphantom = V_HU.T.ravel()

    aMAR = MAR(Voxelphantom, x0, dx, Nx, y0, dy, Ny, theta0, dtheta, Npar, xi0, dxi, M)
    Result = aMAR.Apply()

    return {
        "Voxelphantom": Voxelphantom,
        "isSpectral": isSpectral,
        "Result": Result,
        "Geometry": {
            "RM": RM, "M": M, "Npar": Npar, "N2pi": N2pi,
            "dxi": dxi, "xi0": xi0, "theta0": theta0, "dtheta": dtheta,
            "Nx": Nx, "Ny": Ny, "dx": dx, "dy": dy, "x0": x0, "y0": y0,
        },
    }


if __name__ == "__main__":
    result = main()
    print(f"iMAR result: min {result['Result'].min():.1f}, max {result['Result'].max():.1f}")
