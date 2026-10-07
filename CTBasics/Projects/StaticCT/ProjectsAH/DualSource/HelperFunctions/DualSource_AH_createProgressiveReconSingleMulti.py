"""Progressive single vs. multi source reconstruction of one half turn.

Port of Projects/Static CT/Projects AH/DualSource/HelperFunctions/
DualSource_AH_createProgressiveReconSingleMulti.m.
"""
import math
from pathlib import Path

import numpy as np

from ......Reconstruction.Backprojection import Backprojection


def DualSource_AH_createProgressiveReconSingleMulti(
        Projections, Projections_multi, idx,
        theta0_i, dtheta, Npi, N2pi,
        xi0, dxi, M,
        x0, dx, Nx, y0, dy, Ny,
        Outputdirectory, params, ShowFigures=False):
    idx = np.asarray(idx, dtype=np.int64)

    if idx.size % M != 0:
        raise ValueError("idx length must be a multiple of M.")

    Nproj_total = idx.size // M
    if Nproj_total != Npi:
        raise ValueError("Nproj_total from idx does not match Npi.")
    if Npi % 2 != 0:
        raise ValueError("Npi must be even for multi source splitting.")

    # MATLAB reshapes the flat selection column wise into (M, Npi).
    Sino_single_full = np.asarray(Projections).ravel()[idx].reshape(Npi, M).T
    Sino_multi_full = np.asarray(Projections_multi).ravel()[idx].reshape(Npi, M).T

    Nmulti_half = Npi // 2
    Sino_multi_1_full = Sino_multi_full[:, :Nmulti_half]
    Sino_multi_2_full = Sino_multi_full[:, Nmulti_half:]

    numSteps = 5
    single_steps = [math.floor(k * Npi / 5) for k in (1, 2, 3, 4)] + [Npi]
    multi_steps = [math.floor(k * Nmulti_half / 5) for k in (1, 2, 3, 4)] + [Nmulti_half]

    stepTitles = ["1/5 Scan", "2/5 Scan", "3/5 Scan", "4/5 Scan", "Full Scan"]
    stepNames = ["one_fifth", "two_fifths", "three_fifths", "four_fifths", "full"]

    Recons_single_progress = []
    Recons_multi_progress = []

    for k in range(numSteps):
        Nsingle_k = single_steps[k]
        Sino_single_k = np.zeros((M, Npi))
        Sino_single_k[:, :Nsingle_k] = Sino_single_full[:, :Nsingle_k]
        Recon_single_k = Backprojection.PixDrivenBackProjParCartesian(
            theta0_i, dtheta, Npi, N2pi, xi0, dxi, M,
            Sino_single_k.T.ravel(), x0, dx, Nx, y0, dy, Ny)
        # Rescale so that a partial scan keeps the attenuation level of a full half turn.
        Recons_single_progress.append(Recon_single_k * (Npi / Nsingle_k))

        Nmulti_k = multi_steps[k]
        Sino_multi_k = np.zeros((M, Npi))
        Sino_multi_k[:, :Nmulti_k] = Sino_multi_1_full[:, :Nmulti_k]
        Sino_multi_k[:, Nmulti_half:Nmulti_half + Nmulti_k] = Sino_multi_2_full[:, :Nmulti_k]
        Recon_multi_k = Backprojection.PixDrivenBackProjParCartesian(
            theta0_i, dtheta, Npi, N2pi, xi0, dxi, M,
            Sino_multi_k.T.ravel(), x0, dx, Nx, y0, dy, Ny)
        Recons_multi_progress.append(Recon_multi_k * (Npi / (2 * Nmulti_k)))

    if Outputdirectory is not None:
        _SaveProgressiveImages(Recons_single_progress, Recons_multi_progress,
                               Nx, Ny, numSteps, stepTitles, stepNames,
                               Outputdirectory, params, ShowFigures)

    return Recons_single_progress, Recons_multi_progress


def _SaveProgressiveImages(Recons_single_progress, Recons_multi_progress,
                           Nx, Ny, numSteps, stepTitles, stepNames,
                           Outputdirectory, params, ShowFigures):
    import matplotlib.pyplot as plt

    Outputdirectory = Path(Outputdirectory)
    Outputdirectory.mkdir(parents=True, exist_ok=True)

    tag = (f"HR{int(params['Heartrate'])}"
           f"_TRot{round(params['TRot'] * 1000)}ms"
           f"_N{int(params['Npar'])}"
           f"_PS{round(params['Shift_GT'] * 1000)}")

    fig, axes = plt.subplots(2, numSteps, figsize=(18, 7.5), facecolor="w")
    for k in range(numSteps):
        img_single = np.nan_to_num(Recons_single_progress[k].reshape(Ny, Nx),
                                   nan=0.0, posinf=0.0, neginf=0.0)
        axes[0, k].imshow(img_single, cmap="gray")
        axes[0, k].set_title(f"Single - {stepTitles[k]}")
        axes[0, k].axis("off")

        img_multi = np.nan_to_num(Recons_multi_progress[k].reshape(Ny, Nx),
                                  nan=0.0, posinf=0.0, neginf=0.0)
        axes[1, k].imshow(img_multi, cmap="gray")
        axes[1, k].set_title(f"Multi - {stepTitles[k]}")
        axes[1, k].axis("off")

    fig.suptitle("Progressive reconstruction: Single vs Multi")

    overviewFile = Outputdirectory / f"ProgressiveRecon_Overview_5Steps_{tag}.png"
    fig.savefig(overviewFile, dpi=300)
    print(f"Progressive overview saved: {overviewFile}")

    if ShowFigures:
        plt.show()
    plt.close(fig)

    for k in range(numSteps):
        img_single = np.nan_to_num(Recons_single_progress[k].reshape(Ny, Nx),
                                   nan=0.0, posinf=0.0, neginf=0.0)
        img_multi = np.nan_to_num(Recons_multi_progress[k].reshape(Ny, Nx),
                                  nan=0.0, posinf=0.0, neginf=0.0)
        plt.imsave(Outputdirectory / f"Progressive_Single_{stepNames[k]}_{tag}.png",
                   _MatToGray(img_single), cmap="gray", vmin=0.0, vmax=1.0)
        plt.imsave(Outputdirectory / f"Progressive_Multi_{stepNames[k]}_{tag}.png",
                   _MatToGray(img_multi), cmap="gray", vmin=0.0, vmax=1.0)

    print("Progressive indivimulti images saved.")


def _MatToGray(Image):
    """MATLAB mat2gray: linear rescale to [0, 1], constant images become all zeros."""
    lo = float(np.min(Image))
    hi = float(np.max(Image))
    if hi <= lo:
        return np.zeros_like(Image)
    return (Image - lo) / (hi - lo)
