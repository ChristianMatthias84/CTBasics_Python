"""Visualization and evaluation helpers for the static CT projects.

Port of Projects/Static CT/StaticCTHelperFunctions/StaticCTVisualizationAndEvaluationHelpers.m,
which consolidates the former Projects JP and Projects AH helper functions.
"""
import math
import os
import shutil
import tempfile
import warnings
from pathlib import Path

import numpy as np

from CTBasics.RawData.RawDataHelperFunctions import RawDataHelperFunctions
from CTBasics.Reconstruction.Backprojection import Backprojection
from CTBasics.Simulation.CreatePhantom import CreatePhantom


class StaticCTVisualizationAndEvaluationHelpers:
    """Central helper collection for static CT visualization and evaluation."""

    @staticmethod
    def StaticCT_exportFigure(figureHandle, outputPath, resolution=300):
        """Exports a figure, falling back to a short temporary path if the direct save fails.

        MATLAB works around Windows paths that are too long for the PNG library behind
        exportgraphics; the fallback writes to a temporary file and copies it over.
        """
        try:
            figureHandle.savefig(outputPath, dpi=resolution, bbox_inches="tight")
        except Exception as directExportException:
            extension = Path(outputPath).suffix
            fileDescriptor, temporaryPath = tempfile.mkstemp(suffix=extension)
            os.close(fileDescriptor)
            try:
                figureHandle.savefig(temporaryPath, dpi=resolution, bbox_inches="tight")
                shutil.copyfile(temporaryPath, outputPath)
            except Exception as fallbackException:
                raise directExportException from fallbackException
            finally:
                if os.path.isfile(temporaryPath):
                    os.remove(temporaryPath)

    @staticmethod
    def StaticCT_computeReconstructionMetrics(
            MultiphaseReconHU_linear, MultiphaseReconHU_arbitrary, MultiphaseReconHU_GT,
            RoI_all, Nx, Ny, x0, y0, dx, dy, roi_radius):
        """RMSE, NCC and SSIM per phase and ROI, plus the same metrics on the background.

        The background of a phase is every pixel outside all of its ROIs.
        """
        MultiphaseReconHU_linear = np.asarray(MultiphaseReconHU_linear, dtype=float)
        MultiphaseReconHU_arbitrary = np.asarray(MultiphaseReconHU_arbitrary, dtype=float)
        MultiphaseReconHU_GT = np.asarray(MultiphaseReconHU_GT, dtype=float)

        NRecons = MultiphaseReconHU_linear.shape[0]
        nRoI = np.asarray(RoI_all[0]).shape[1]

        RMSE_linear = np.zeros((NRecons, nRoI))
        RMSE_arbitrary = np.zeros((NRecons, nRoI))
        NCC_linear = np.zeros((NRecons, nRoI))
        NCC_arbitrary = np.zeros((NRecons, nRoI))
        ssim_linear = np.zeros((NRecons, nRoI))
        ssim_arbitrary = np.zeros((NRecons, nRoI))

        RMSE_linear_background = np.zeros(NRecons)
        RMSE_arbitrary_background = np.zeros(NRecons)
        NCC_linear_background = np.zeros(NRecons)
        NCC_arbitrary_background = np.zeros(NRecons)
        ssim_linear_background = np.zeros(NRecons)
        ssim_arbitrary_background = np.zeros(NRecons)

        roi_mask = {}
        center_mask = {}
        background_mask = [None] * NRecons

        r_pix = roi_radius / dx  # Note: exact only for isotropic pixels (dx == dy)

        # MATLAB builds the mask with ndgrid(1:Nx, 1:Ny), so j runs along the first and ii
        # along the second dimension of an (Nx, Ny) mask.
        j, ii = np.meshgrid(np.arange(1, Nx + 1), np.arange(1, Ny + 1), indexing="ij")

        for iRoI in range(nRoI):
            roi_mask_1D = np.zeros((NRecons, Nx * Ny))

            for k in range(NRecons):
                roi_center_x = np.asarray(RoI_all[k])[0, iRoI]
                roi_center_y = np.asarray(RoI_all[k])[1, iRoI]
                cx_pix = (roi_center_x - x0) / dx + 1
                cy_pix = (roi_center_y - y0) / dy + 1

                Mask = (ii - cx_pix) ** 2 + (j - cy_pix) ** 2 <= r_pix ** 2
                roi_mask[(k, iRoI)] = Mask
                roi_mask_1D[k, :] = RawDataHelperFunctions.SimpleLinearizeArray(Mask)
                center_mask[(k, iRoI)] = np.array([cx_pix, cy_pix])

            roi_mask_1D = roi_mask_1D.astype(bool)

            for k in range(NRecons):
                Selected = roi_mask_1D[k, :]
                Linear_roi = MultiphaseReconHU_linear[k, Selected]
                Arbitrary_roi = MultiphaseReconHU_arbitrary[k, Selected]
                GT_roi = MultiphaseReconHU_GT[k, Selected]

                RMSE_linear[k, iRoI] = _RMSE(Linear_roi, GT_roi)
                RMSE_arbitrary[k, iRoI] = _RMSE(Arbitrary_roi, GT_roi)
                NCC_linear[k, iRoI] = _Corr2(Linear_roi, GT_roi)
                NCC_arbitrary[k, iRoI] = _Corr2(Arbitrary_roi, GT_roi)

            for k in range(NRecons):
                Mask = roi_mask[(k, iRoI)]
                row, col = np.nonzero(Mask)
                if row.size == 0:
                    continue

                img1_crop, img2_crop, img3_crop, mask_crop = _CropToMask(
                    MultiphaseReconHU_linear[k], MultiphaseReconHU_GT[k],
                    MultiphaseReconHU_arbitrary[k], Mask, row, col, Nx, Ny)

                img1_crop[~mask_crop] = 0
                img2_crop[~mask_crop] = 0
                img3_crop[~mask_crop] = 0

                dr = _DynamicRange(img2_crop)

                ssim_linear[k, iRoI] = _SSIM(img1_crop, img2_crop, dr)
                ssim_arbitrary[k, iRoI] = _SSIM(img3_crop, img2_crop, dr)

        for k in range(NRecons):
            background_mask[k] = np.ones((Nx, Ny), dtype=bool)
            for iRoI in range(nRoI):
                if (k, iRoI) in roi_mask:
                    background_mask[k] = background_mask[k] & ~roi_mask[(k, iRoI)]

            if not background_mask[k].any():
                continue

            row, col = np.nonzero(background_mask[k])
            img1_crop, img2_crop, img3_crop, mask_crop = _CropToMask(
                MultiphaseReconHU_linear[k], MultiphaseReconHU_GT[k],
                MultiphaseReconHU_arbitrary[k], background_mask[k], row, col, Nx, Ny)

            img1_bg = img1_crop[mask_crop]
            img2_bg = img2_crop[mask_crop]
            img3_bg = img3_crop[mask_crop]

            if img1_bg.size == 0 or img2_bg.size == 0:
                continue

            img1_crop[~mask_crop] = 0
            img2_crop[~mask_crop] = 0
            img3_crop[~mask_crop] = 0

            dr = _DynamicRange(img2_crop)

            RMSE_linear_background[k] = np.sqrt(np.mean((img1_bg - img2_bg) ** 2))
            RMSE_arbitrary_background[k] = np.sqrt(np.mean((img3_bg - img2_bg) ** 2))

            NCC_linear_background[k] = _Corr2(img1_bg, img2_bg)
            NCC_arbitrary_background[k] = _Corr2(img3_bg, img2_bg)

            ssim_linear_background[k] = _SSIM(img1_crop, img2_crop, dr)
            ssim_arbitrary_background[k] = _SSIM(img3_crop, img2_crop, dr)

        return (RMSE_linear, RMSE_arbitrary, NCC_linear, NCC_arbitrary,
                ssim_linear, ssim_arbitrary, nRoI, roi_mask, center_mask,
                RMSE_linear_background, RMSE_arbitrary_background,
                NCC_linear_background, NCC_arbitrary_background,
                ssim_linear_background, ssim_arbitrary_background, background_mask)

    @staticmethod
    def DualSource_buildPhantomRoiMask(SelectedPhantom, motionCenter, Nx, Ny, x0, y0, dx, dy,
                                       fallbackCenterPix, LineWidth=2.0):
        """Phantom-aware ROI mask: the union of all relevant circles as an (Ny, Nx) mask.

        The geometry mirrors CreatePhantom.CreateVoxelPhantom2DWithMotionUnified. Pixel
        centers are 1-based like in MATLAB, so the mask equals the MATLAB mask element-wise.
        Returns (roiMask, roiCentersPix of shape (2, K), roiRadiiPix of shape (K,)).
        """
        name = str(SelectedPhantom).lower()
        xDisp = float(motionCenter[0])
        yDisp = float(motionCenter[1])
        useFallback = False

        if "linepattern" in name:
            # Moving block; radius ~4*L covers all four stripes
            centersPhys = np.array([[xDisp], [yDisp]])
            radiiMm = np.array([4 * LineWidth])
        elif "4arter" in name:
            # Four arteries on a 25 mm orbit (angles pi/4, 3pi/4, 5pi/4, 7pi/4)
            arteryOrbit = 25
            arteryAngles = np.array([math.pi / 4, 3 * math.pi / 4,
                                     5 * math.pi / 4, 7 * math.pi / 4])
            centersPhys = np.vstack((arteryOrbit * np.cos(arteryAngles) + xDisp,
                                     arteryOrbit * np.sin(arteryAngles) + yDisp))
            radiiMm = 8 * np.ones(4)
        elif "respirat" in name:
            # Two static lung centers at +-70 mm (do not follow cardiac motion)
            centersPhys = np.array([[-70.0, 70.0], [0.0, 0.0]])
            radiiMm = np.array([45.0, 45.0])
        elif "wateronly" in name or "waterphantom" in name:
            # Static water disc; NOT shifted by motion (noise calibration ROI)
            centersPhys = np.array([[0.0], [0.0]])
            radiiMm = np.array([40.0])
        elif "cardiac" in name:
            centersPhys = np.array([[xDisp], [yDisp]])
            radiiMm = np.array([10.0])
        else:
            # Unknown phantom: fall back to legacy 80 px circle
            useFallback = True

        X, Y = np.meshgrid(np.arange(1, Nx + 1), np.arange(1, Ny + 1))
        roiMask = np.zeros((Ny, Nx), dtype=bool)

        if useFallback:
            cx = max(1, min(Nx, fallbackCenterPix[0]))
            cy = max(1, min(Ny, fallbackCenterPix[1]))
            rPix = 80.0
            roiMask = ((X - cx) ** 2 + (Y - cy) ** 2) <= rPix ** 2
            return roiMask, np.array([[cx], [cy]], dtype=float), np.array([rPix])

        K = centersPhys.shape[1]
        roiCentersPix = np.zeros((2, K))
        roiRadiiPix = np.zeros(K)

        for k in range(K):
            cx = max(1, min(Nx, _MatlabRound((centersPhys[0, k] - x0) / dx)))
            cy = max(1, min(Ny, _MatlabRound((centersPhys[1, k] - y0) / dy)))
            rPix = radiiMm[k] / dx  # assumes isotropic pixels (dx == dy)

            roiMask = roiMask | (((X - cx) ** 2 + (Y - cy) ** 2) <= rPix ** 2)

            roiCentersPix[:, k] = [cx, cy]
            roiRadiiPix[k] = rPix

        return roiMask, roiCentersPix, roiRadiiPix

    @staticmethod
    def StaticCT_plotReconstructionResults(
            MultiphaseReconHU_GT, MultiphaseReconHU_linear, MultiphaseReconHU_arbitrary,
            MultiphaseRecon_linear_GT, MultiphaseRecon_arbitrary_GT,
            RMSE_linear, RMSE_arbitrary, NCC_linear, NCC_arbitrary,
            ssim_linear, ssim_arbitrary,
            roi_mask, center_mask, Nx, Ny, nRoI, WC, WW, NSubsets, phasesToPlot=None):
        """Displays ground truth, linear and arbitrary order reconstructions side by side."""
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

    @staticmethod
    def DualSource_plotScanTiming(Heartrate, MotionType, TRest, Shift_GT, TRot, Nsources,
                                  THeartCycle, Shift_Single):
        """Plots the single and multi source scan windows on the cardiac motion curve."""
        import matplotlib.pyplot as plt

        figure = plt.figure()
        axes = figure.gca()
        axes.grid(True)

        t_cont = np.linspace(-THeartCycle, 2 * THeartCycle, 1000)
        amp = 10
        y_sin = amp * np.array([
            CreatePhantom.MotionProfile(tt, Heartrate, True, MotionType, TRest)
            for tt in t_cont
        ])
        axes.plot(t_cont, y_sin, "b", linewidth=2)

        y_center_single = amp * CreatePhantom.MotionProfile(
            Shift_GT, Heartrate, True, MotionType, TRest)
        window_single = TRot / 2
        t1_single = Shift_GT - window_single / 2
        t2_single = Shift_GT + window_single / 2
        axes.plot([t1_single, t2_single], [y_center_single, y_center_single], "r",
                  linewidth=4)

        y_center_multi = amp * CreatePhantom.MotionProfile(
            Shift_GT, Heartrate, True, MotionType, TRest)
        window_multi = TRot / (2 * Nsources)
        t1_multi = Shift_GT - window_multi / 2
        t2_multi = Shift_GT + window_multi / 2
        axes.plot([t1_multi, t2_multi], [y_center_multi, y_center_multi], "g", linewidth=2)

        axes.plot(Shift_GT, y_center_single, "o", color=(0.3, 0.3, 0.3),
                  markersize=4, linewidth=2)

        axes.set_xlabel("Time [s]")
        axes.set_ylabel("Displacement [mm]")
        axes.set_title("Scan windows centered on motion")
        axes.legend(["Motion", "Single", "Multi", "GroundTruth"])
        axes.set_xlim(0 - Shift_Single, 2 * THeartCycle)

        return figure

    @staticmethod
    def DualSource_AH_plotScanTimingSegmentedSingle(
            Heartrate, MotionType, TRest, Shift_GT, TRot, Nsources, THeartCycle, Shift_Single,
            TRot_segmented, isSynchronizedSegments=False):
        """Timing plot of a continuous versus a segmented single source scan."""
        import matplotlib.pyplot as plt

        figure = plt.figure()
        axes = figure.gca()
        axes.grid(True)

        amp = 10

        if isSynchronizedSegments:
            segmentCenter = Shift_GT
            segmentHalfWindow = TRot_segmented / 4
            segmentStarts = np.full(Nsources, segmentCenter - segmentHalfWindow)
            segmentEnds = np.full(Nsources, segmentCenter + segmentHalfWindow)
            segmentCenters = np.full(Nsources, segmentCenter)
        else:
            singleReadStartTime = Shift_GT - Shift_Single
            singleReadEndTime = Shift_GT + Shift_Single
            segmentPhaseEdges = np.linspace(singleReadStartTime, singleReadEndTime,
                                            Nsources + 1)
            segmentStarts = segmentPhaseEdges[:-1]
            segmentEnds = segmentPhaseEdges[1:]
            segmentCenters = (segmentStarts + segmentEnds) / 2

        tMin = min(segmentStarts.min(), Shift_GT - Shift_Single) - 0.2 * THeartCycle
        tMax = max(segmentEnds.max(), Shift_GT + Shift_Single) + 0.2 * THeartCycle

        t_cont = np.linspace(tMin, tMax, 1600)
        y_sin = amp * np.array([
            CreatePhantom.MotionProfile(tt, Heartrate, True, MotionType, TRest)
            for tt in t_cont
        ])
        axes.plot(t_cont, y_sin, "b", linewidth=2)

        t1Single = Shift_GT - Shift_Single
        t2Single = Shift_GT + Shift_Single
        ySingle = amp * CreatePhantom.MotionProfile(Shift_GT, Heartrate, True, MotionType,
                                                    TRest)
        axes.plot([t1Single, t2Single], [ySingle, ySingle], "r", linewidth=4)

        if isSynchronizedSegments:
            ySegBase = amp * CreatePhantom.MotionProfile(
                segmentCenters[0], Heartrate, True, MotionType, TRest)
            if Nsources > 1:
                offsetScale = np.linspace(-0.35, 0.35, Nsources)
            else:
                offsetScale = np.zeros(1)

            for sourceIdx in range(Nsources):
                ySeg = ySegBase + offsetScale[sourceIdx]
                axes.plot([segmentStarts[sourceIdx], segmentEnds[sourceIdx]], [ySeg, ySeg],
                          "g", linewidth=2)
                axes.plot(segmentCenters[sourceIdx], ySeg, "ko", markerfacecolor="k",
                          markersize=3)

            titleStr = ("Timing: continuous single vs synchronized segmented single "
                        f"(Nsegments = {Nsources})")
            segmentedLegend = "Segmented (same time)"
        else:
            for sourceIdx in range(Nsources):
                ySeg = amp * CreatePhantom.MotionProfile(
                    segmentCenters[sourceIdx], Heartrate, True, MotionType, TRest)
                axes.plot([segmentStarts[sourceIdx], segmentEnds[sourceIdx]], [ySeg, ySeg],
                          "g", linewidth=3)
                axes.plot(segmentCenters[sourceIdx], ySeg, "ko", markerfacecolor="k",
                          markersize=4)

            titleStr = ("Timing: continuous single vs segmented single "
                        f"(Nsegments = {Nsources})")
            segmentedLegend = "Segmented bins"

        axes.set_xlabel("Time [s]")
        axes.set_ylabel("Displacement [mm]")
        axes.set_title(titleStr)
        axes.legend(["Motion", "Single continuous", segmentedLegend, "Segment centers"])

        txt = (f"TRot_segmented = {TRot_segmented:.3f}s "
               f"({TRot_segmented / TRot:.1f}x TRot)")
        axes.text(tMin + 0.02 * (tMax - tMin), amp * 0.9, txt, color=(0.1, 0.1, 0.1))
        axes.set_xlim(tMin, tMax)

        return figure

    @staticmethod
    def DualSource_AH_createProgressiveReconSingleMulti(
            Projections, Projections_multi, idx,
            theta0_i, dtheta, Npi, N2pi,
            xi0, dxi, M,
            x0, dx, Nx, y0, dy, Ny,
            Outputdirectory, params, ShowFigures=False):
        """Progressive single vs. multi source reconstruction of one half turn."""
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
            Sino_multi_k[:, Nmulti_half:Nmulti_half + Nmulti_k] = \
                Sino_multi_2_full[:, :Nmulti_k]
            Recon_multi_k = Backprojection.PixDrivenBackProjParCartesian(
                theta0_i, dtheta, Npi, N2pi, xi0, dxi, M,
                Sino_multi_k.T.ravel(), x0, dx, Nx, y0, dy, Ny)
            Recons_multi_progress.append(Recon_multi_k * (Npi / (2 * Nmulti_k)))

        if Outputdirectory is not None:
            _SaveProgressiveImages(Recons_single_progress, Recons_multi_progress,
                                   Nx, Ny, numSteps, stepTitles, stepNames,
                                   Outputdirectory, params, ShowFigures)

        return Recons_single_progress, Recons_multi_progress


def _MatlabRound(x):
    """MATLAB round: halves are rounded away from zero (Python's round is banker's)."""
    return float(np.sign(x) * np.floor(np.abs(x) + 0.5))


def _CropToMask(linear, gt, arbitrary, Mask, row, col, Nx, Ny):
    """Crops the three images and the mask to the bounding box of the mask.

    MATLAB crops the transposed (Ny, Nx) images with indices taken from the (Nx, Ny)
    mask; kept verbatim (identical for the square grids used by all callers).
    """
    rmin, rmax = int(row.min()), int(row.max())
    cmin, cmax = int(col.min()), int(col.max())
    img1_crop = linear.reshape(Ny, Nx)[rmin:rmax + 1, cmin:cmax + 1].copy()
    img2_crop = gt.reshape(Ny, Nx)[rmin:rmax + 1, cmin:cmax + 1].copy()
    img3_crop = arbitrary.reshape(Ny, Nx)[rmin:rmax + 1, cmin:cmax + 1].copy()
    mask_crop = Mask[rmin:rmax + 1, cmin:cmax + 1]
    return img1_crop, img2_crop, img3_crop, mask_crop


def _DynamicRange(Image):
    """max - min of the reference crop; MATLAB replaces a vanishing range by 1."""
    dr = float(Image.max() - Image.min())
    if abs(dr) < np.finfo(float).eps:
        dr = 1.0
    return dr


def _RMSE(A, B):
    A = np.asarray(A, dtype=float).ravel()
    B = np.asarray(B, dtype=float).ravel()
    if A.size == 0:
        return np.nan
    return float(np.sqrt(np.mean((A - B) ** 2)))


def _Corr2(A, B):
    A = np.asarray(A, dtype=float).ravel()
    B = np.asarray(B, dtype=float).ravel()
    if A.size == 0:
        return np.nan
    A = A - A.mean()
    B = B - B.mean()
    denominator = np.sqrt(np.sum(A * A) * np.sum(B * B))
    if denominator == 0:
        return np.nan
    return float(np.sum(A * B) / denominator)


def _SSIM(A, B, DynamicRange):
    """MATLAB ssim defaults: 11x11 Gaussian window, sigma 1.5, K = [0.01, 0.03].

    Implemented locally because neither numpy nor scipy ship an SSIM routine and the
    port must not add a new third party dependency for one metric.
    """
    return float(np.mean(_SSIMMap(A, B, DynamicRange)))


def _SSIMMap(A, B, DynamicRange):
    """Local SSIM map, the second output of MATLAB's [ssimval, ssimmap] = ssim(A, B)."""
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)

    C1 = (0.01 * DynamicRange) ** 2
    C2 = (0.03 * DynamicRange) ** 2

    Window = _GaussianWindow(11, 1.5)

    muA = _Filter(A, Window)
    muB = _Filter(B, Window)

    muA2 = muA * muA
    muB2 = muB * muB
    muAB = muA * muB

    sigmaA2 = _Filter(A * A, Window) - muA2
    sigmaB2 = _Filter(B * B, Window) - muB2
    sigmaAB = _Filter(A * B, Window) - muAB

    numerator = (2 * muAB + C1) * (2 * sigmaAB + C2)
    denominator = (muA2 + muB2 + C1) * (sigmaA2 + sigmaB2 + C2)

    return numerator / denominator


def _GaussianWindow(size, sigma):
    coordinates = np.arange(size) - (size - 1) / 2
    Window = np.exp(-(coordinates ** 2) / (2 * sigma ** 2))
    Window = np.outer(Window, Window)
    return Window / Window.sum()


def _Filter(Image, Window):
    """Replicate padding plus correlation, like MATLAB's ssim (imgaussfilt 'replicate')."""
    radius = (Window.shape[0] - 1) // 2
    Padded = np.pad(Image, radius, mode="edge")

    Result = np.zeros_like(Image, dtype=float)
    for dy in range(Window.shape[0]):
        for dx in range(Window.shape[1]):
            Result += Window[dy, dx] * Padded[dy:dy + Image.shape[0],
                                              dx:dx + Image.shape[1]]
    return Result


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
