"""File output helpers (reports, raw images, GIFs) for the static CT projects.

Port of Projects/Static CT/StaticCTHelperFunctions/StaticCTIOHelpers.m, which consolidates
the former Projects JP and Projects AH output functions.
"""
import math
import re
from datetime import datetime
from pathlib import Path

import numpy as np

from CTBasics.Reconstruction.Constants import Constants
from CTBasics.Simulation.CreatePhantom import CreatePhantom


class StaticCTIOHelpers:
    """Central helper collection for static CT file output."""

    @staticmethod
    def StaticCT_createSubsetAnimationGif(ThetaArray, reconType, NSubsets, Outputdirectory,
                                          MaxAngles=100, MaxFrames=30, DelayTime=0.2,
                                          Visible=False):
        """Animates the scanned angles on the unit circle and writes them as a GIF."""
        import matplotlib.pyplot as plt
        from matplotlib.animation import PillowWriter

        ThetaArray = np.asarray(ThetaArray, dtype=float).ravel()

        maxAngles = round(MaxAngles)
        maxFrames = round(MaxFrames)

        NGifAngles = min(maxAngles, ThetaArray.size)
        if NGifAngles < 1:
            raise ValueError("ThetaArray is empty.")

        nFrames = min(maxFrames, NGifAngles)
        frameIndices = np.unique(np.round(np.linspace(1, NGifAngles, nFrames)).astype(int))

        t = np.linspace(0, 2 * math.pi, 500)

        fig = plt.figure(facecolor="w")
        axes = fig.gca()
        axes.plot(np.cos(t), np.sin(t), "k", linewidth=1.5)
        axes.set_aspect("equal")
        axes.set_xlim(-1.2, 1.2)
        axes.set_ylim(-1.2, 1.2)
        axes.grid(True)
        axes.set_title("Animation of angles on unit circle")
        axes.set_xlabel(r"cos($\theta$)")
        axes.set_ylabel(r"sin($\theta$)")

        if reconType == "180":
            subset_angles = np.linspace(0, math.pi, NSubsets + 1)
        else:
            subset_angles = np.linspace(0, 2 * math.pi, NSubsets + 1)

        r_outer = 1.0
        r_inner = 0.92
        for angle in subset_angles:
            axes.plot([r_inner * math.cos(angle), r_outer * math.cos(angle)],
                      [r_inner * math.sin(angle), r_outer * math.sin(angle)],
                      "b", linewidth=2)

        hPts, = axes.plot([], [], "ro", markerfacecolor="r", markersize=6)

        filename = Path(Outputdirectory) / "winkel_animation.gif"
        filename.parent.mkdir(parents=True, exist_ok=True)

        writer = PillowWriter(fps=max(1.0 / DelayTime, 1.0))
        with writer.saving(fig, str(filename), dpi=80):
            for idx in frameIndices:
                hPts.set_data(np.cos(ThetaArray[:idx]), np.sin(ThetaArray[:idx]))
                writer.grab_frame()

        if Visible:
            plt.show()
        else:
            plt.close(fig)

        print(f"GIF saved as: {filename}")
        return str(filename)

    @staticmethod
    def StaticCT_appendReconstructionMetrics(
            Outputdirectory, SelectedPhantom, TRecon,
            RMSE_linear, RMSE_arbitrary,
            NCC_linear, NCC_arbitrary,
            SSIM_linear, SSIM_arbitrary,
            RoIAll=None, RoIRadiusMm=None, Tag="",
            ResultFileName="reconstruction_metrics.txt"):
        """Writes the 4D CT reconstruction metric report and returns the summary dict."""
        Outputdirectory = Path(Outputdirectory)
        Outputdirectory.mkdir(parents=True, exist_ok=True)

        Metrics = [RMSE_linear, RMSE_arbitrary, NCC_linear, NCC_arbitrary,
                   SSIM_linear, SSIM_arbitrary]
        Shapes = [_Shape(m) for m in Metrics]

        nRecons = max(s[0] for s in Shapes)
        nRoI = max(s[1] for s in Shapes)

        if RoIAll:
            roiCols = [np.asarray(x).shape[1] for x in RoIAll if np.size(x) > 0]
            if roiCols:
                nRoI = max(nRoI, max(roiCols))

        if nRecons == 0 or nRoI == 0:
            raise ValueError("Metric arrays are empty.")

        RMSE_linear = StaticCTIOHelpers._iPadMetric(RMSE_linear, nRecons, nRoI)
        RMSE_arbitrary = StaticCTIOHelpers._iPadMetric(RMSE_arbitrary, nRecons, nRoI)
        NCC_linear = StaticCTIOHelpers._iPadMetric(NCC_linear, nRecons, nRoI)
        NCC_arbitrary = StaticCTIOHelpers._iPadMetric(NCC_arbitrary, nRecons, nRoI)
        SSIM_linear = StaticCTIOHelpers._iPadMetric(SSIM_linear, nRecons, nRoI)
        SSIM_arbitrary = StaticCTIOHelpers._iPadMetric(SSIM_arbitrary, nRecons, nRoI)

        if TRecon is None or np.size(TRecon) == 0:
            TRecon = np.arange(1, nRecons + 1, dtype=float)
        TRecon = np.asarray(TRecon, dtype=float).ravel()
        if TRecon.size < nRecons:
            TRecon = np.concatenate((TRecon, np.full(nRecons - TRecon.size, np.nan)))
        elif TRecon.size > nRecons:
            TRecon = TRecon[:nRecons]

        roiCenterX, roiCenterY = StaticCTIOHelpers._iEstimateRoiCenters(RoIAll, nRoI)

        MeanOmitNaN = StaticCTIOHelpers._iMeanOmitNaN

        rmseMeanLinearRoi = MeanOmitNaN(RMSE_linear)
        rmseMeanArbRoi = MeanOmitNaN(RMSE_arbitrary)
        rmseImproveRoi = rmseMeanLinearRoi - rmseMeanArbRoi

        nccMeanLinearRoi = MeanOmitNaN(NCC_linear)
        nccMeanArbRoi = MeanOmitNaN(NCC_arbitrary)
        nccImproveRoi = nccMeanArbRoi - nccMeanLinearRoi

        ssimMeanLinearRoi = MeanOmitNaN(SSIM_linear)
        ssimMeanArbRoi = MeanOmitNaN(SSIM_arbitrary)
        ssimImproveRoi = ssimMeanArbRoi - ssimMeanLinearRoi

        summary = {
            "Phantom": SelectedPhantom,
            "Tag": Tag,
            "NRecons": nRecons,
            "NRoI": nRoI,
            "TRecon": TRecon,
            "RoICenterX": roiCenterX,
            "RoICenterY": roiCenterY,
            "RMSE_linear": RMSE_linear,
            "RMSE_arbitrary": RMSE_arbitrary,
            "NCC_linear": NCC_linear,
            "NCC_arbitrary": NCC_arbitrary,
            "SSIM_linear": SSIM_linear,
            "SSIM_arbitrary": SSIM_arbitrary,
            "RMSE_linear_mean_roi": rmseMeanLinearRoi,
            "RMSE_arbitrary_mean_roi": rmseMeanArbRoi,
            "RMSE_improvement_mean_roi": rmseImproveRoi,
            "NCC_linear_mean_roi": nccMeanLinearRoi,
            "NCC_arbitrary_mean_roi": nccMeanArbRoi,
            "NCC_improvement_mean_roi": nccImproveRoi,
            "SSIM_linear_mean_roi": ssimMeanLinearRoi,
            "SSIM_arbitrary_mean_roi": ssimMeanArbRoi,
            "SSIM_improvement_mean_roi": ssimImproveRoi,
            "RMSE_linear_global": float(MeanOmitNaN(RMSE_linear.reshape(-1, 1))[0]),
            "RMSE_arbitrary_global": float(MeanOmitNaN(RMSE_arbitrary.reshape(-1, 1))[0]),
            "NCC_linear_global": float(MeanOmitNaN(NCC_linear.reshape(-1, 1))[0]),
            "NCC_arbitrary_global": float(MeanOmitNaN(NCC_arbitrary.reshape(-1, 1))[0]),
            "SSIM_linear_global": float(MeanOmitNaN(SSIM_linear.reshape(-1, 1))[0]),
            "SSIM_arbitrary_global": float(MeanOmitNaN(SSIM_arbitrary.reshape(-1, 1))[0]),
        }
        summary["RMSE_global_improvement"] = \
            summary["RMSE_linear_global"] - summary["RMSE_arbitrary_global"]
        summary["NCC_global_improvement"] = \
            summary["NCC_arbitrary_global"] - summary["NCC_linear_global"]
        summary["SSIM_global_improvement"] = \
            summary["SSIM_arbitrary_global"] - summary["SSIM_linear_global"]

        resultFile = Outputdirectory / ResultFileName

        lines = [
            "===== 4DCT Reconstruction Metrics =====\r\n",
            f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\r\n",
            f"Phantom: {SelectedPhantom}\r\n",
        ]
        if Tag:
            lines.append(f"Tag: {Tag}\r\n")
        lines += [
            f"NRecons: {nRecons}\r\n",
            f"NRoI: {nRoI}\r\n",
            "TRecon [s]:",
        ]
        for k in range(nRecons):
            lines.append("  NaN" if np.isnan(TRecon[k]) else f"  {TRecon[k]:.4f}")
        lines += [
            "\r\n\r\n",
            "--- Global Means (avg over all phases & ROIs) ---\r\n",
            "RMSE  linear / arbitrary / improvement: "
            f"{summary['RMSE_linear_global']:.6f} / {summary['RMSE_arbitrary_global']:.6f} / "
            f"{summary['RMSE_global_improvement']:.6f}\r\n",
            "NCC   linear / arbitrary / improvement: "
            f"{summary['NCC_linear_global']:.6f} / {summary['NCC_arbitrary_global']:.6f} / "
            f"{summary['NCC_global_improvement']:.6f}\r\n",
            "SSIM  linear / arbitrary / improvement: "
            f"{summary['SSIM_linear_global']:.6f} / {summary['SSIM_arbitrary_global']:.6f} / "
            f"{summary['SSIM_global_improvement']:.6f}\r\n\r\n",
            "--- Per-ROI Results ---\r\n",
        ]

        for iRoI in range(nRoI):
            lines.append(f"ROI {iRoI + 1}")
            arteryDesc = StaticCTIOHelpers._iGetArteryDescription(SelectedPhantom, iRoI + 1)
            if arteryDesc:
                lines.append(f" ({arteryDesc})")
            lines.append(":\r\n")

            if not np.isnan(roiCenterX[iRoI]) and not np.isnan(roiCenterY[iRoI]):
                lines.append(f"  Mean center [mm]: x={roiCenterX[iRoI]:.3f}, "
                             f"y={roiCenterY[iRoI]:.3f}\r\n")

            if RoIRadiusMm is not None and np.size(RoIRadiusMm) > 0:
                if np.isscalar(RoIRadiusMm) or np.size(RoIRadiusMm) == 1:
                    lines.append(f"  Radius [mm]: {float(np.ravel(RoIRadiusMm)[0]):.3f}\r\n")
                elif iRoI < np.size(RoIRadiusMm):
                    lines.append(
                        f"  Radius [mm]: {float(np.ravel(RoIRadiusMm)[iRoI]):.3f}\r\n")

            lines += [
                "  Mean over all phases:\r\n",
                "    RMSE  linear / arbitrary / improvement: "
                f"{rmseMeanLinearRoi[iRoI]:.6f} / {rmseMeanArbRoi[iRoI]:.6f} / "
                f"{rmseImproveRoi[iRoI]:.6f}\r\n",
                "    NCC   linear / arbitrary / improvement: "
                f"{nccMeanLinearRoi[iRoI]:.6f} / {nccMeanArbRoi[iRoI]:.6f} / "
                f"{nccImproveRoi[iRoI]:.6f}\r\n",
                "    SSIM  linear / arbitrary / improvement: "
                f"{ssimMeanLinearRoi[iRoI]:.6f} / {ssimMeanArbRoi[iRoI]:.6f} / "
                f"{ssimImproveRoi[iRoI]:.6f}\r\n",
                "  Per phase (t [s] | RMSE lin/arb | NCC lin/arb | SSIM lin/arb):\r\n",
            ]

            for k in range(nRecons):
                lines.append(
                    f"    Phase {k + 1:2d} (t={TRecon[k]:.4f} s): "
                    f"RMSE {RMSE_linear[k, iRoI]:.6f}/{RMSE_arbitrary[k, iRoI]:.6f}  "
                    f"NCC {NCC_linear[k, iRoI]:.6f}/{NCC_arbitrary[k, iRoI]:.6f}  "
                    f"SSIM {SSIM_linear[k, iRoI]:.6f}/{SSIM_arbitrary[k, iRoI]:.6f}\r\n")

            lines.append("\r\n")

        lines.append("========================================\r\n\r\n")

        with open(resultFile, "w", newline="") as handle:
            handle.writelines(lines)

        print(f"Metrics saved: {resultFile}")
        print(f"\n--- Global Means [{SelectedPhantom}], all TRecons & RoIs ---")
        print(f"  RMSE  linear / arbitrary / improvement: "
              f"{summary['RMSE_linear_global']:.6f} / {summary['RMSE_arbitrary_global']:.6f}"
              f" / {summary['RMSE_global_improvement']:.6f}")
        print(f"  NCC   linear / arbitrary / improvement: "
              f"{summary['NCC_linear_global']:.6f} / {summary['NCC_arbitrary_global']:.6f}"
              f" / {summary['NCC_global_improvement']:.6f}")
        print(f"  SSIM  linear / arbitrary / improvement: "
              f"{summary['SSIM_linear_global']:.6f} / {summary['SSIM_arbitrary_global']:.6f}"
              f" / {summary['SSIM_global_improvement']:.6f}\n")

        summary["ResultFile"] = str(resultFile)
        return summary

    @staticmethod
    def DualSource_createGroundTruthGif(
            Outputdirectory, Nx, Ny, dx, dy, x0, y0,
            Heartrate, CM, Stent, Plaque, Rotation, Direction, MotionType, TRest,
            Shift_GT, TRot, Nsources, Npar, SelectedPhantom):
        """Renders the ground truth motion as an animated GIF (PillowWriter frames)."""
        import matplotlib.pyplot as plt
        from matplotlib.animation import PillowWriter
        from matplotlib.patches import Ellipse

        ampGif = 10

        window_single = TRot / 2
        t1_single = Shift_GT - window_single / 2
        t2_single = Shift_GT + window_single / 2

        window_multi = TRot / (2 * Nsources)
        t1_multi = Shift_GT - window_multi / 2
        t2_multi = Shift_GT + window_multi / 2

        tStart = t1_single
        tEnd = t2_single

        NframesGif = 100
        dT_gif = (tEnd - tStart) / (NframesGif - 1)
        gifSpeedFactor = 0.2
        gifDelay = max(dT_gif / gifSpeedFactor, 0.02)

        xMargin = window_single / 2
        tCurve = np.linspace(tStart - xMargin, tEnd + xMargin, 500)
        yCurve = ampGif * np.array([
            CreatePhantom.MotionProfile(tt, Heartrate, True, MotionType, TRest)
            for tt in tCurve
        ])

        gifFile = Path(Outputdirectory) / (
            f"GroundTruth_Motion_HR{int(Heartrate)}_TRot{round(TRot * 1000)}ms"
            f"_N{int(Npar)}.gif"
        )
        gifFile.parent.mkdir(parents=True, exist_ok=True)

        figGif = plt.figure(figsize=(13, 4.5), facecolor="w")
        if figGif.canvas.manager:
            figGif.canvas.manager.set_window_title("Ground Truth Motion")
        axCurve = figGif.add_subplot(1, 3, 1)
        axImg = figGif.add_subplot(1, 3, 2)
        axLed = figGif.add_subplot(1, 3, 3)

        ySingle = ampGif * CreatePhantom.MotionProfile(
            (t1_single + t2_single) / 2, Heartrate, True, MotionType, TRest)
        yMulti = ampGif * CreatePhantom.MotionProfile(
            (t1_multi + t2_multi) / 2, Heartrate, True, MotionType, TRest)
        yGT = ampGif * CreatePhantom.MotionProfile(Shift_GT, Heartrate, True, MotionType,
                                                   TRest)

        writer = PillowWriter(fps=max(1.0 / gifDelay, 1.0))

        with writer.saving(figGif, str(gifFile), dpi=80):
            for f in range(1, NframesGif + 1):
                tNow = tStart + (f - 1) * dT_gif
                nT_gif = (tNow - Shift_GT) / dT_gif

                Image_gt, _, _, _ = CreatePhantom.CreateVoxelPhantom2DWithMotionUnified(
                    Nx, dx, x0, Ny, dy, y0,
                    Shift_GT, dT_gif, nT_gif, SelectedPhantom,
                    Heartrate=Heartrate, CM=CM, Stent=Stent, Plaque=Plaque,
                    Motion=True, Rotation=Rotation, Direction=Direction,
                    MotionType=MotionType, TRest=TRest,
                )

                axCurve.clear()
                axCurve.grid(True)
                axCurve.plot(tCurve, yCurve, "b", linewidth=1.5)
                axCurve.plot([t1_single, t2_single], [ySingle, ySingle], "r", linewidth=4)
                axCurve.plot([t1_multi, t2_multi], [yMulti, yMulti], "g", linewidth=4)
                axCurve.plot(Shift_GT, yGT, "o", color=(0.3, 0.3, 0.3),
                             markerfacecolor=(0.8, 0.8, 0.8), markersize=6)
                yNow = ampGif * CreatePhantom.MotionProfile(
                    tNow, Heartrate, True, MotionType, TRest)
                axCurve.plot(tNow, yNow, "ko", markerfacecolor="w", markersize=8)
                axCurve.axvline(tNow, linestyle="--", color="k")
                axCurve.set_xlabel("Time [s]")
                axCurve.set_ylabel("Displacement [mm]")
                axCurve.set_title("Scan position on motion")
                axCurve.legend(["Motion", "Single window", "Multi window", "GroundTruth",
                                "Current"], loc="lower center")
                axCurve.set_xlim(tStart - xMargin, tEnd + xMargin)

                axImg.clear()
                axImg.imshow(np.asarray(Image_gt).reshape(Ny, Nx), cmap="gray")
                axImg.set_title(f"Ground Truth   t = {tNow:.3f} s")
                axImg.axis("off")

                singleActive = t1_single <= tNow <= t2_single
                multiActive = t1_multi <= tNow <= t2_multi

                axLed.clear()
                colSingle = StaticCTIOHelpers._pickLedColor(
                    singleActive, (1.0, 0.2, 0.2), (0.35, 0.15, 0.15))
                colMulti = StaticCTIOHelpers._pickLedColor(
                    multiActive, (0.2, 0.9, 0.2), (0.15, 0.30, 0.15))
                axLed.add_patch(Ellipse((0.3, 0.70), 0.2, 0.2, facecolor=colSingle,
                                        edgecolor="k"))
                axLed.text(0.5, 0.70,
                           f"Single Source  [{StaticCTIOHelpers._onOff(singleActive)}]",
                           va="center", fontweight="bold")
                axLed.add_patch(Ellipse((0.3, 0.35), 0.2, 0.2, facecolor=colMulti,
                                        edgecolor="k"))
                axLed.text(0.5, 0.35,
                           f"Multi Source   [{StaticCTIOHelpers._onOff(multiActive)}]",
                           va="center", fontweight="bold")
                axLed.set_title("Scan status")
                axLed.set_xlim(0, 1.6)
                axLed.set_ylim(0, 1)
                axLed.axis("off")

                writer.grab_frame()

        plt.close(figGif)
        print(f"Ground truth motion GIF saved: {gifFile}")
        return str(gifFile)

    @staticmethod
    def DualSource_showAndSaveImage(dataVector, dims, Outputdirectory, name, doSave, params,
                                    roiMask=None, ShowFigures=False):
        """Shows one image and optionally writes it as HU raw data (float64).

        MATLAB's two trailing ignored arguments are dropped; the figure is only shown
        when ShowFigures is set.
        """
        dataVector = np.asarray(dataVector, dtype=float).ravel()
        aImage = dataVector.reshape(dims[1], dims[0])

        if ShowFigures:
            import matplotlib.pyplot as plt

            plt.figure()
            plt.imshow(aImage, cmap="gray")
            if roiMask is not None:
                plt.contour(np.asarray(roiMask), [1], colors="r", linewidths=2)
            plt.title(name)
            plt.show()

        filename = None
        if doSave:
            phantomTag = params.get("PhantomTag") or "NA"

            filename = Path(Outputdirectory) / (
                f"{name}_HR{int(params['Heartrate'])}"
                f"_TRot{round(params['TRot'] * 1000)}ms"
                f"_N{int(params['Npar'])}"
                f"_PS{round(params['Shift_GT'] * 1000)}"
                f"_Ph_{phantomTag}"
            )
            filename.parent.mkdir(parents=True, exist_ok=True)

            dataVectorHU = (dataVector - Constants.MuW) / Constants.MuW * 1000
            filename.write_bytes(np.asarray(dataVectorHU, dtype=np.float64).tobytes())

        return aImage, (str(filename) if filename is not None else None)

    @staticmethod
    def DualSource_appendReconstructionMetrics(
            Outputdirectory,
            corr2_normal, corr2_multi, corr2_normal_rest, corr2_multi_rest,
            rmse_normal, rmse_multi, rmse_normal_rest, rmse_multi_rest,
            corr2_normal_roi, corr2_multi_roi, corr2_normal_rest_roi, corr2_multi_rest_roi,
            rmse_normal_roi, rmse_multi_roi, rmse_normal_rest_roi, rmse_multi_rest_roi):
        """Appends the reconstruction metrics of a dual source experiment."""
        resultFile = Path(Outputdirectory) / "reconstruction_metrics.txt"
        resultFile.parent.mkdir(parents=True, exist_ok=True)

        lines = [
            "===== Reconstruction Results =====\r\n\r\n",
            "--- Global ---\r\n",
            f"Correlation normal: {corr2_normal:.6f}\r\n",
            f"Correlation multi:   {corr2_multi:.6f}\r\n",
            f"Correlation normal vs rest GT: {corr2_normal_rest:.6f}\r\n",
            f"Correlation multi   vs rest GT: {corr2_multi_rest:.6f}\r\n",
            f"RMSE normal:        {rmse_normal:.6f}\r\n",
            f"RMSE multi:          {rmse_multi:.6f}\r\n\r\n",
            f"RMSE normal vs rest GT: {rmse_normal_rest:.6f}\r\n",
            f"RMSE multi   vs rest GT: {rmse_multi_rest:.6f}\r\n\r\n",
            "--- ROI ---\n",
            f"Correlation normal (ROI): {corr2_normal_roi:.6f}\r\n",
            f"Correlation multi   (ROI): {corr2_multi_roi:.6f}\r\n",
            f"Correlation normal vs rest GT (ROI): {corr2_normal_rest_roi:.6f}\r\n",
            f"Correlation multi   vs rest GT (ROI): {corr2_multi_rest_roi:.6f}\r\n",
            f"RMSE normal (ROI):        {rmse_normal_roi:.6f}\r\n",
            f"RMSE multi   (ROI):        {rmse_multi_roi:.6f}\r\n\r\n",
            f"RMSE normal vs rest GT (ROI): {rmse_normal_rest_roi:.6f}\r\n",
            f"RMSE multi   vs rest GT (ROI): {rmse_multi_rest_roi:.6f}\r\n\r\n",
            "--- Improvements (Multi vs Normal) ---\r\n",
            f"Correlation improvement: {corr2_multi - corr2_normal:.6f}\r\n",
            f"RMSE improvement:       {rmse_normal - rmse_multi:.6f}\r\n\r\n",
            "===============================\n\n",
        ]

        with open(resultFile, "a", newline="") as handle:
            handle.writelines(lines)

        print(f"Results saved: {resultFile}")
        return str(resultFile)

    @staticmethod
    def DualSource_buildOutputNaming(selectedPhantom, plaque, cm, stent):
        """Builds the output naming dict for a dual source experiment."""
        if selectedPhantom is None:
            selectedPhantom = ""
        selectedPhantom = str(selectedPhantom)

        if not selectedPhantom:
            selectedPhantom = "UnknownPhantom"

        selectedPhantom = selectedPhantom.strip()

        phantomLabel = re.sub(r"[^A-Za-z0-9_-]", "", selectedPhantom)
        if not phantomLabel:
            phantomLabel = "UnknownPhantom"

        if "linepattern" in selectedPhantom.lower():
            rootFolderLabel = "linePattern"
        else:
            rootFolderLabel = phantomLabel

        phantomCode = f"P{round(plaque)}C{int(bool(cm))}S{int(bool(stent))}"

        return {
            "SelectedPhantom": selectedPhantom,
            "PhantomLabel": phantomLabel,
            "RootFolderLabel": rootFolderLabel,
            "PhantomCode": phantomCode,
            "PhantomTag": f"{phantomLabel}_{phantomCode}",
        }

    @staticmethod
    def DualSource_writeSimulationParameters(
            Outputdirectory, Nsources, TRot, Heartrate, dT,
            RM, M, Npar, N2pi, dtheta, Nx, Ny, dx, dy,
            Shift_GT, Shift_GT_Rest, Rotation, Direction, MotionType, TRest,
            ApplyMoireFilter, SimNoise, N_I0, naming=None):
        """Writes the simulation parameter report of a dual source experiment."""
        if naming is None:
            naming = {}

        paramFile = Path(Outputdirectory) / "simulation_parameters.txt"
        paramFile.parent.mkdir(parents=True, exist_ok=True)

        THeart = 60 / Heartrate

        lines = [
            "===== Simulation Parameters =====\r\n\r\n",
            f"N Sources: {int(Nsources)}\r\n",
            "--- Time ---\n",
            f"Rotation time (TRot): {TRot:.4f} s\r\n",
            f"Heart rate (Heartrate): {int(Heartrate)} bpm\r\n",
            f"Heart cycle duration: {THeart:.4f} s\r\n",
            f"Time step (dT): {dT:.6f} s\r\n\r\n",
            "--- Geometry ---\r\n",
            f"Radius (RM): {RM:.2f} mm\r\n",
            f"Detector elements (M): {int(M)}\r\n",
            f"Number of projections (Npar): {int(Npar)}\r\n",
            f"Total projections (N2pi): {int(N2pi)}\r\n",
            f"Angular step (dtheta): {dtheta:.6f} rad\r\n\r\n",
            "--- Reconstruction grid ---\r\n",
            f"Nx: {int(Nx)}, Ny: {int(Ny)}\r\n",
            f"dx: {dx:.4f} mm, dy: {dy:.4f} mm\r\n\r\n",
            "--- Phantom ---\r\n",
            f"Shift_GT: {Shift_GT:.4f} s\r\n",
            f"Shift_GT_Rest: {Shift_GT_Rest:.4f} s\r\n",
            f"Rotation: {int(Rotation)}\r\n",
            f"Direction: {int(Direction)}\r\n",
            f"MotionType: {MotionType}\r\n",
            f"TRest: {int(TRest)}\r\n\r\n",
            "--- Naming ---\r\n",
        ]

        if "SelectedPhantom" in naming:
            lines.append(f"SelectedPhantom: {naming['SelectedPhantom']}\r\n")
        if "RootFolderLabel" in naming:
            lines.append(f"RootFolderLabel: {naming['RootFolderLabel']}\r\n")
        if "PhantomTag" in naming:
            lines.append(f"PhantomTag: {naming['PhantomTag']}\r\n")
        if "PhantomCode" in naming:
            lines.append(f"PhantomCode: {naming['PhantomCode']}\r\n\r\n")
        else:
            lines.append("\r\n")

        lines += [
            "--- Simulation ---\r\n",
            f"Moire filter: {int(ApplyMoireFilter)}\r\n",
            f"Noise enabled: {int(SimNoise)}\r\n",
            f"Photon count (N_I0): {N_I0:.2e}\r\n\r\n",
        ]

        # newline="" keeps the explicit \r\n of the MATLAB fprintf calls byte identical.
        with open(paramFile, "w", newline="") as handle:
            handle.writelines(lines)

        print(f"Parameter file saved: {paramFile}")
        return str(paramFile)

    @staticmethod
    def _iPadMetric(metric, nRows, nCols):
        Padded = np.full((nRows, nCols), np.nan)
        if metric is None or np.size(metric) == 0:
            return Padded

        Array = np.asarray(metric, dtype=float)
        if Array.ndim == 1:
            Array = Array.reshape(-1, 1)

        r = min(Array.shape[0], nRows)
        c = min(Array.shape[1], nCols)
        Padded[:r, :c] = Array[:r, :c]
        return Padded

    @staticmethod
    def _iMeanOmitNaN(x, dim=0):
        """Mean along dim that ignores NaN and yields NaN for all-NaN slices (no warning)."""
        x = np.asarray(x, dtype=float)
        valid = ~np.isnan(x)
        count = valid.sum(axis=dim)
        total = np.where(valid, x, 0.0).sum(axis=dim)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = total / count
        mean[count == 0] = np.nan
        return mean

    @staticmethod
    def _iEstimateRoiCenters(RoIAll, nRoI):
        xMean = np.full(nRoI, np.nan)
        yMean = np.full(nRoI, np.nan)
        if not RoIAll:
            return xMean, yMean

        xVals = np.full((len(RoIAll), nRoI), np.nan)
        yVals = np.full((len(RoIAll), nRoI), np.nan)

        for k, roiK in enumerate(RoIAll):
            if np.size(roiK) == 0:
                continue
            roiK = np.asarray(roiK, dtype=float)
            nCols = min(roiK.shape[1], nRoI)
            xVals[k, :nCols] = roiK[0, :nCols]
            yVals[k, :nCols] = roiK[1, :nCols]

        return (StaticCTIOHelpers._iMeanOmitNaN(xVals),
                StaticCTIOHelpers._iMeanOmitNaN(yVals))

    @staticmethod
    def _iGetArteryDescription(phantomName, roiIndex):
        if phantomName != "WaterWith4Arteries":
            return ""
        return {1: "Stent", 2: "Half Plaque", 3: "Quarter Plaque",
                4: "Three-Quarter Plaque"}.get(roiIndex, "")

    @staticmethod
    def _pickLedColor(isActive, onColor, offColor):
        return onColor if isActive else offColor

    @staticmethod
    def _onOff(isActive):
        return "ON" if isActive else "OFF"


def _Shape(metric):
    if metric is None or np.size(metric) == 0:
        return (0, 0)
    Array = np.atleast_2d(np.asarray(metric, dtype=float))
    return Array.shape
