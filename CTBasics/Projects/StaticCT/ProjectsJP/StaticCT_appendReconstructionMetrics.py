"""Writes the 4D CT reconstruction metric report and returns the summary struct.

Port of Projects/Static CT/Projects JP/StaticCT_appendReconstructionMetrics.m.
"""
from datetime import datetime
from pathlib import Path

import numpy as np


def StaticCT_appendReconstructionMetrics(
        Outputdirectory, SelectedPhantom, TRecon,
        RMSE_linear, RMSE_arbitrary,
        NCC_linear, NCC_arbitrary,
        SSIM_linear, SSIM_arbitrary,
        RoIAll=None, RoIRadiusMm=None, Tag="",
        ResultFileName="reconstruction_metrics.txt"):
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

    RMSE_linear = _PadMetric(RMSE_linear, nRecons, nRoI)
    RMSE_arbitrary = _PadMetric(RMSE_arbitrary, nRecons, nRoI)
    NCC_linear = _PadMetric(NCC_linear, nRecons, nRoI)
    NCC_arbitrary = _PadMetric(NCC_arbitrary, nRecons, nRoI)
    SSIM_linear = _PadMetric(SSIM_linear, nRecons, nRoI)
    SSIM_arbitrary = _PadMetric(SSIM_arbitrary, nRecons, nRoI)

    if TRecon is None or np.size(TRecon) == 0:
        TRecon = np.arange(1, nRecons + 1, dtype=float)
    TRecon = np.asarray(TRecon, dtype=float).ravel()
    if TRecon.size < nRecons:
        TRecon = np.concatenate((TRecon, np.full(nRecons - TRecon.size, np.nan)))
    elif TRecon.size > nRecons:
        TRecon = TRecon[:nRecons]

    roiCenterX, roiCenterY = _EstimateRoiCenters(RoIAll, nRoI)

    rmseMeanLinearRoi = _MeanOmitNaN(RMSE_linear)
    rmseMeanArbRoi = _MeanOmitNaN(RMSE_arbitrary)
    rmseImproveRoi = rmseMeanLinearRoi - rmseMeanArbRoi

    nccMeanLinearRoi = _MeanOmitNaN(NCC_linear)
    nccMeanArbRoi = _MeanOmitNaN(NCC_arbitrary)
    nccImproveRoi = nccMeanArbRoi - nccMeanLinearRoi

    ssimMeanLinearRoi = _MeanOmitNaN(SSIM_linear)
    ssimMeanArbRoi = _MeanOmitNaN(SSIM_arbitrary)
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
        "RMSE_linear_global": _MeanOmitNaNFlat(RMSE_linear),
        "RMSE_arbitrary_global": _MeanOmitNaNFlat(RMSE_arbitrary),
        "NCC_linear_global": _MeanOmitNaNFlat(NCC_linear),
        "NCC_arbitrary_global": _MeanOmitNaNFlat(NCC_arbitrary),
        "SSIM_linear_global": _MeanOmitNaNFlat(SSIM_linear),
        "SSIM_arbitrary_global": _MeanOmitNaNFlat(SSIM_arbitrary),
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
        arteryDesc = _GetArteryDescription(SelectedPhantom, iRoI + 1)
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
                lines.append(f"  Radius [mm]: {float(np.ravel(RoIRadiusMm)[iRoI]):.3f}\r\n")

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
          f"{summary['RMSE_linear_global']:.6f} / {summary['RMSE_arbitrary_global']:.6f} / "
          f"{summary['RMSE_global_improvement']:.6f}")
    print(f"  NCC   linear / arbitrary / improvement: "
          f"{summary['NCC_linear_global']:.6f} / {summary['NCC_arbitrary_global']:.6f} / "
          f"{summary['NCC_global_improvement']:.6f}")
    print(f"  SSIM  linear / arbitrary / improvement: "
          f"{summary['SSIM_linear_global']:.6f} / {summary['SSIM_arbitrary_global']:.6f} / "
          f"{summary['SSIM_global_improvement']:.6f}\n")

    summary["ResultFile"] = str(resultFile)
    return summary


def _Shape(metric):
    if metric is None or np.size(metric) == 0:
        return (0, 0)
    Array = np.atleast_2d(np.asarray(metric, dtype=float))
    return Array.shape


def _PadMetric(metric, nRows, nCols):
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


def _MeanOmitNaN(x):
    """Column wise mean that ignores NaN and yields NaN for all-NaN columns."""
    x = np.asarray(x, dtype=float)
    valid = ~np.isnan(x)
    count = valid.sum(axis=0)
    total = np.where(valid, x, 0.0).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = total / count
    mean[count == 0] = np.nan
    return mean


def _MeanOmitNaNFlat(x):
    return float(_MeanOmitNaN(np.asarray(x, dtype=float).reshape(-1, 1))[0])


def _EstimateRoiCenters(RoIAll, nRoI):
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

    return _MeanOmitNaN(xVals), _MeanOmitNaN(yVals)


def _GetArteryDescription(phantomName, roiIndex):
    if phantomName != "WaterWith4Arteries":
        return ""
    return {1: "Stent", 2: "Half Plaque", 3: "Quarter Plaque",
            4: "Three-Quarter Plaque"}.get(roiIndex, "")
