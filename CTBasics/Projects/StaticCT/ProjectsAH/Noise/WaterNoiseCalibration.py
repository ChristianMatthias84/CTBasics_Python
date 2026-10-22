"""Water-noise calibration for N_I0.

Port of Projects/Static CT/Projects AH/Noise/WaterNoiseCalibration.m. A static homogeneous
water phantom (no motion or structure, unlike the cardiac main workflow) is reconstructed
for several photon counts and sigma is measured in HU; Poisson noise scales roughly as
sigma ~ 1/sqrt(N_I0). MATLAB hard codes the settings; the port exposes them as keyword
defaults so the calibration can also run on small grids.
"""
import math

import numpy as np

from CTBasics.MultiCore.Parallelization import Parallelization
from CTBasics.Reconstruction.Backprojection import Backprojection
from CTBasics.Reconstruction.Constants import Constants
from CTBasics.Reconstruction.Convolution import Convolution
from CTBasics.Reconstruction.ForwardProjection import ForwardProjection
from CTBasics.Reconstruction.Kernels import Kernels
from CTBasics.Simulation.Noise import Noise


def main(ShowFigures=False, waterTargetSigmaHU=35,
         N_I0_sweep=(1e5, 2e5, 3e5, 5e5, 8e5, 1e6, 2e6), Reconkernel=Kernels.RamLak,
         RM=60, M=500, Npar=512, Nx=512, Ny=512, TRot=0.5):
    N_I0_sweep = np.asarray(N_I0_sweep, dtype=float)

    # Parallel beam geometry (mirrors the DualSource main script)
    N2pi = 2 * Npar         # full-circle view count (reference grid)
    Npi = N2pi // 2         # half-circle view count (single source, 180 deg)
    dtheta = 2 * math.pi / N2pi
    theta0 = 0

    dxi = 2 * RM / (M - 1)
    xi0 = -RM - dxi

    dx = 2 * RM / Nx
    dy = 2 * RM / Ny
    x0 = -RM - dx / 2.0
    y0 = -RM - dy / 2.0

    Parallelization.Init()

    # Static water phantom projections (noise-free), single source, no motion
    ThetaArraySingle = np.linspace(0, math.pi, Npi + 1)[:-1]

    Projections_clean, _, _, _, _ = \
        ForwardProjection.RayDrivenForwProjParCartesianMultipleWithMotionDual(
            x0, dx, Nx, y0, dy, Ny,
            Npi, xi0, dxi, M, N2pi, TRot, ThetaArraySingle,
            SelectedPhantom="WaterOnly", Motion=False)
    Projections_clean = np.asarray(Projections_clean).ravel()

    NprojSingle = Projections_clean.size // M
    if Projections_clean.size % M != 0:
        raise ValueError("Projection length must be divisible by M.")

    # Reconstruction kernel (built once, reused for every N_I0)
    if Reconkernel == Kernels.SheppLogan:
        Kernel = Kernels.SheppLoganKernelSD(M, dxi)
    else:
        Kernel = Kernels.RamLakKernelSD(M, dxi)

    # Central ROI inside the water disc; MATLAB round, the values are positive.
    waterRoiRadiusPx = math.floor(min(Nx, Ny) * 0.22 + 0.5)
    XGrid, YGrid = np.meshgrid(np.arange(1, Nx + 1), np.arange(1, Ny + 1))
    waterMask = ((XGrid - math.floor(Nx / 2 + 0.5)) ** 2
                 + (YGrid - math.floor(Ny / 2 + 0.5)) ** 2) <= waterRoiRadiusPx ** 2

    numRuns = N_I0_sweep.size
    measuredSigmaHU = np.zeros(numRuns)
    measuredMeanHU = np.zeros(numRuns)

    print("\n===== Water noise calibration (WaterOnly phantom) =====")
    print(f"{'N_I0':>12s} | {'mean [HU]':>10s} | {'sigma [HU]':>10s}")
    print("-------------------------------------------------")

    for k in range(numRuns):
        N_I0 = N_I0_sweep[k]

        Projections = Noise.ApplyPoissonNoise(Projections_clean, M * NprojSingle, N_I0)

        Projections = Convolution.ConvolveSinogram(Projections, Kernel, M, NprojSingle, dxi)

        ReconstructedImage = Backprojection.PixDrivenBackProjParCartesian(
            theta0, dtheta, Npi, N2pi, xi0, dxi, M,
            Projections, x0, dx, Nx, y0, dy, Ny)

        ReconHU = (np.asarray(ReconstructedImage).reshape(Ny, Nx) - Constants.MuW) \
            / Constants.MuW * 1000

        waterVals = ReconHU[waterMask].astype(float)
        measuredMeanHU[k] = np.mean(waterVals)
        measuredSigmaHU[k] = np.std(waterVals, ddof=1)  # MATLAB std normalizes by N-1

        print(f"{N_I0:12.2e} | {measuredMeanHU[k]:10.2f} | {measuredSigmaHU[k]:10.2f}")

    # Fit sigma = c / sqrt(N_I0) and invert for the target sigma.
    cEstimatePerRun = measuredSigmaHU * np.sqrt(N_I0_sweep)
    cEstimate = float(np.mean(cEstimatePerRun))
    recommendedN_I0 = (cEstimate / waterTargetSigmaHU) ** 2

    print("-------------------------------------------------")
    print(f"Target water sigma: {waterTargetSigmaHU:.0f} HU")
    print(f"Fitted model: sigma ~ {cEstimate:.3e} / sqrt(N_I0)")
    print(f"Recommended N_I0 for target: {recommendedN_I0:.2e}")
    print("=================================================\n")

    N_I0_model = np.logspace(math.log10(N_I0_sweep.min()), math.log10(N_I0_sweep.max()), 200)
    sigma_model = cEstimate / np.sqrt(N_I0_model)

    _PlotCalibration(N_I0_sweep, measuredSigmaHU, N_I0_model, sigma_model,
                     waterTargetSigmaHU, recommendedN_I0, ShowFigures)

    return {
        "waterTargetSigmaHU": waterTargetSigmaHU,
        "N_I0_sweep": N_I0_sweep,
        "Kernel": Kernel,
        "Projections_clean": Projections_clean,
        "NprojSingle": NprojSingle,
        "waterRoiRadiusPx": waterRoiRadiusPx,
        "waterMask": waterMask,
        "measuredMeanHU": measuredMeanHU,
        "measuredSigmaHU": measuredSigmaHU,
        "cEstimatePerRun": cEstimatePerRun,
        "cEstimate": cEstimate,
        "recommendedN_I0": recommendedN_I0,
        "N_I0_model": N_I0_model,
        "sigma_model": sigma_model,
        "Geometry": {"RM": RM, "M": M, "Npar": Npar, "N2pi": N2pi, "Npi": Npi,
                     "dtheta": dtheta, "theta0": theta0, "xi0": xi0, "dxi": dxi,
                     "x0": x0, "dx": dx, "Nx": Nx, "y0": y0, "dy": dy, "Ny": Ny,
                     "TRot": TRot},
    }


def _PlotCalibration(N_I0_sweep, measuredSigmaHU, N_I0_model, sigma_model,
                     waterTargetSigmaHU, recommendedN_I0, ShowFigures):
    import matplotlib.pyplot as plt

    figure = plt.figure("Water noise vs N_I0")
    axes = figure.gca()
    axes.loglog(N_I0_sweep, measuredSigmaHU, "o", markerfacecolor="b", label="measured")
    axes.loglog(N_I0_model, sigma_model, "-", label=r"fit: $c/\sqrt{N_{I0}}$")
    axes.axhline(waterTargetSigmaHU, linestyle="--", color="r", label="target")
    axes.axvline(recommendedN_I0, linestyle="--", color="k", label=r"recommended $N_{I0}$")
    axes.grid(True)
    axes.set_xlabel(r"$N_{I0}$ (photons per detection interval)")
    axes.set_ylabel(r"Water noise $\sigma$ [HU]")
    axes.set_title("Water noise calibration")
    axes.legend(loc="lower left")

    if ShowFigures:
        plt.show()
    else:
        plt.close(figure)

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    main(ShowFigures=True)
    plt.show()
