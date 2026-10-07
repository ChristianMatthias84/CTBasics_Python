"""Displays ground truth, linear and arbitrary order reconstructions side by side.

Port of Projects/Static CT/Projects JP/StaticCT_plotReconstructionResults.m.
"""
import warnings

import numpy as np


def StaticCT_plotReconstructionResults(
        MultiphaseReconHU_GT, MultiphaseReconHU_linear, MultiphaseReconHU_arbitrary,
        MultiphaseRecon_linear_GT, MultiphaseRecon_arbitrary_GT,
        RMSE_linear, RMSE_arbitrary, NCC_linear, NCC_arbitrary,
        ssim_linear, ssim_arbitrary,
        roi_mask, center_mask, Nx, Ny, nRoI, WC, WW, NSubsets, phasesToPlot=None):
    import matplotlib.pyplot as plt

    nRecons = np.asarray(MultiphaseReconHU_GT).shape[0]

    if phasesToPlot is None or np.size(phasesToPlot) == 0:
        phasesToPlot = np.arange(1, nRecons + 1)

    phasesToPlot = np.unique(np.asarray(phasesToPlot).ravel())
    phasesToPlot = phasesToPlot[(phasesToPlot >= 1) & (phasesToPlot <= nRecons)]

    if phasesToPlot.size == 0:
        warnings.warn("No valid phase indices in phasesToPlot. Nothing was plotted.")
        return []

    Figures = []
    vmin, vmax = WC - WW / 2, WC + WW / 2

    for k in phasesToPlot:
        kk = int(k) - 1

        Figures.append(_ShowWithRois(
            MultiphaseReconHU_GT[kk].reshape(Ny, Nx), vmin, vmax,
            roi_mask, center_mask, kk, nRoI, f"Ground Truth (Phase {k})"))

        Figures.append(_ShowWithRois(
            MultiphaseReconHU_linear[kk].reshape(Ny, Nx), vmin, vmax,
            roi_mask, center_mask, kk, nRoI, f"linear forward projection (Phase {k})"))

        # The difference images share one symmetric window derived from the 99th
        # percentile of both difference volumes.
        p1 = np.percentile(np.abs(MultiphaseRecon_linear_GT[kk]), 99)
        p2 = np.percentile(np.abs(MultiphaseRecon_arbitrary_GT[kk]), 99)
        p = max(p1, p2)

        rmseLin = np.nanmean(RMSE_linear[kk, :])
        nccLin = np.nanmean(NCC_linear[kk, :])
        ssimLin = np.nanmean(ssim_linear[kk, :])

        Figures.append(_ShowWithRois(
            MultiphaseRecon_linear_GT[kk].reshape(Ny, Nx), -p, p,
            roi_mask, center_mask, kk, nRoI,
            f"difference image linear/Ground Truth, RMSE = {rmseLin:.2f}, "
            f"NCC = {nccLin:.4f}, SSIM = {ssimLin:.4f}, C = 0, W = {p:.2f} HU "
            f"(Phase {k})"))

        Figures.append(_ShowWithRois(
            MultiphaseReconHU_arbitrary[kk].reshape(Ny, Nx), vmin, vmax,
            roi_mask, center_mask, kk, nRoI,
            f"arbitrary forward with {NSubsets:.0f} Subsets (Phase {k})"))

        rmseArb = np.nanmean(RMSE_arbitrary[kk, :])
        nccArb = np.nanmean(NCC_arbitrary[kk, :])
        ssimArb = np.nanmean(ssim_arbitrary[kk, :])

        Figures.append(_ShowWithRois(
            MultiphaseRecon_arbitrary_GT[kk].reshape(Ny, Nx), -p, p,
            roi_mask, center_mask, kk, nRoI,
            f"difference image arbitrary/Ground Truth, RMSE = {rmseArb:.2f}, "
            f"NCC = {nccArb:.4f}, SSIM = {ssimArb:.4f}, C = 0, W = {p:.2f} HU "
            f"(Phase {k})"))

    plt.show()
    return Figures


def _ShowWithRois(Image, vmin, vmax, roi_mask, center_mask, kk, nRoI, title):
    import matplotlib.pyplot as plt

    figure = plt.figure()
    axes = figure.gca()
    axes.imshow(Image, cmap="gray", vmin=vmin, vmax=vmax)

    for iRoI in range(nRoI):
        Mask = roi_mask.get((kk, iRoI)) if isinstance(roi_mask, dict) \
            else roi_mask[kk][iRoI]
        axes.contour(np.asarray(Mask, dtype=float), [1], colors="r", linewidths=2)

        if nRoI > 1:
            Center = center_mask.get((kk, iRoI)) if isinstance(center_mask, dict) \
                else center_mask[kk][iRoI]
            axes.text(Center[0], Center[1] - 35, str(iRoI + 1), color="red",
                      fontsize=12, horizontalalignment="center")

    axes.set_title(title)
    axes.axis("off")
    return figure
