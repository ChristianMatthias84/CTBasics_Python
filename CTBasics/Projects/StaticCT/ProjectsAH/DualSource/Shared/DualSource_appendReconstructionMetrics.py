"""Appends the reconstruction metrics of a dual source experiment.

Port of Projects/Static CT/Projects AH/DualSource/Shared/DualSource_appendReconstructionMetrics.m.
"""
from pathlib import Path


def DualSource_appendReconstructionMetrics(
        Outputdirectory,
        corr2_normal, corr2_multi, corr2_normal_rest, corr2_multi_rest,
        rmse_normal, rmse_multi, rmse_normal_rest, rmse_multi_rest,
        corr2_normal_roi, corr2_multi_roi, corr2_normal_rest_roi, corr2_multi_rest_roi,
        rmse_normal_roi, rmse_multi_roi, rmse_normal_rest_roi, rmse_multi_rest_roi):
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
