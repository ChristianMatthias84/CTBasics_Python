"""ROI based comparison of a linear and an arbitrary order 4D CT reconstruction.

Port of Projects/Static CT/Projects JP/StaticCT_computeReconstructionMetrics.m.
"""
import numpy as np

from ....RawData.RawDataHelperFunctions import RawDataHelperFunctions


def StaticCT_computeReconstructionMetrics(
        MultiphaseReconHU_linear, MultiphaseReconHU_arbitrary, MultiphaseReconHU_GT,
        RoI_all, Nx, Ny, x0, y0, dx, dy, roi_radius):
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

    roi_mask = {}
    center_mask = {}

    r_pix = roi_radius / dx  # Note: only correct if dx = dy

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
                ssim_linear[k, iRoI] = np.nan
                ssim_arbitrary[k, iRoI] = np.nan
                continue

            rmin, rmax = int(row.min()), int(row.max())
            cmin, cmax = int(col.min()), int(col.max())

            # MATLAB crops the transposed (Ny, Nx) images with indices taken from the
            # (Nx, Ny) mask; kept verbatim because the ROI is circular.
            img1_crop = MultiphaseReconHU_linear[k].reshape(Ny, Nx)[rmin:rmax + 1,
                                                                    cmin:cmax + 1].copy()
            img2_crop = MultiphaseReconHU_GT[k].reshape(Ny, Nx)[rmin:rmax + 1,
                                                                cmin:cmax + 1].copy()
            img3_crop = MultiphaseReconHU_arbitrary[k].reshape(Ny, Nx)[rmin:rmax + 1,
                                                                       cmin:cmax + 1].copy()
            mask_crop = Mask[rmin:rmax + 1, cmin:cmax + 1]

            img1_crop[~mask_crop] = 0
            img2_crop[~mask_crop] = 0
            img3_crop[~mask_crop] = 0

            dr = float(img2_crop.max() - img2_crop.min())

            ssim_linear[k, iRoI] = _SSIM(img1_crop, img2_crop, dr)
            ssim_arbitrary[k, iRoI] = _SSIM(img3_crop, img2_crop, dr)

    return (RMSE_linear, RMSE_arbitrary, NCC_linear, NCC_arbitrary,
            ssim_linear, ssim_arbitrary, nRoI, roi_mask, center_mask)


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
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)

    if DynamicRange == 0:
        return np.nan

    C1 = (0.01 * DynamicRange) ** 2
    C2 = (0.03 * DynamicRange) ** 2

    Window = _GaussianWindow(11, 1.5)
    # MATLAB shrinks the window for images smaller than the default 11x11 support.
    radius = min(5, (min(A.shape) - 1) // 2)
    if radius < 5:
        if radius < 1:
            return np.nan
        Window = _GaussianWindow(2 * radius + 1, 1.5)

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

    return float(np.mean(numerator / denominator))


def _GaussianWindow(size, sigma):
    coordinates = np.arange(size) - (size - 1) / 2
    Window = np.exp(-(coordinates ** 2) / (2 * sigma ** 2))
    Window = np.outer(Window, Window)
    return Window / Window.sum()


def _Filter(Image, Window):
    """Symmetric padding plus correlation, matching MATLAB's imfilter default for ssim."""
    radius = (Window.shape[0] - 1) // 2
    Padded = np.pad(Image, radius, mode="symmetric")

    Result = np.zeros_like(Image, dtype=float)
    for dy in range(Window.shape[0]):
        for dx in range(Window.shape[1]):
            Result += Window[dy, dx] * Padded[dy:dy + Image.shape[0],
                                              dx:dx + Image.shape[1]]
    return Result
