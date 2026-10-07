"""Cardiac 4D CT comparison of a linear and a subset interleaved scan order.

Port of Projects/Static CT/Projects JP/TestCardiac4DCT.m.
"""
import math
from pathlib import Path

import numpy as np

from ....LecturesAndTestScripts.CTRecon101 import _Save
from ....Memory.Allocation import Allocation
from ....ReadAndWrite.IOHelperFunctions import IOHelperFunctions
from ....Reconstruction.Backprojection import Backprojection
from ....Reconstruction.Constants import Constants
from ....Reconstruction.Convolution import Convolution
from ....Reconstruction.ForwardProjection import ForwardProjection
from ....Reconstruction.Kernels import Kernels
from ....Reconstruction.MoireFilter import MoireFilter


def main(Outputdirectory=None, writeData=True, reconType="multiphase",
         Reconkernel=Kernels.RamLak, ApplyMoireFilter=False, isplot=False,
         TRot=0.5, Heartrate=60, RM=250, M=500, N2pi=1024, Nx=512, Ny=512,
         SelectedPhantom="CardiacPhantom", PhantomOld=False,
         CM=False, Stent=False, Plaque=1, Motion=True, NSubsets=64, Seed=None):
    if reconType != "multiphase":
        raise ValueError("This script supports only reconType = 'multiphase'.")

    if Outputdirectory is None:
        Outputdirectory = Path(IOHelperFunctions().GlobalOutputFolderPath) \
            / "cardiacPhantom" / "data" / "TestCardiac4DCT"
    Outputdirectory = Path(Outputdirectory)
    Outputdirectory.mkdir(parents=True, exist_ok=True)

    THeart = 60 / Heartrate          # duration of one heart cycle
    NumRot = THeart / TRot           # rotations needed to scan one full heart cycle

    thetaN = 2 * math.pi * NumRot
    N = int(N2pi * thetaN / (2 * math.pi))

    dxi = 2 * RM / (M - 1)
    xi0 = -RM - dxi
    theta0 = 0
    dtheta = thetaN / N
    dT = TRot / N2pi

    dx = 2 * RM / Nx
    dy = 2 * RM / Ny
    x0 = -RM - dx / 2.0
    y0 = -RM - dy / 2.0

    # 0 to 2*pi-dtheta, so every theta appears exactly once and 2*pi is not duplicated.
    ThetaArray = np.linspace(0, 2 * math.pi - dtheta, N2pi)
    ThetaArray_2 = np.linspace(0, math.pi - dtheta, N2pi // 2)

    rng = np.random.default_rng(Seed)
    idx_rand = rng.permutation(N2pi // 2)
    ThetaArray_rand = np.concatenate((ThetaArray[idx_rand],
                                      ThetaArray[idx_rand + N2pi // 2]))

    ThetaArray_sub = _BuildSubsetOrder(ThetaArray_2, NSubsets)

    print("Forward Projection running...")
    Projections, Angles_start, center_points_1, PhantomTruth, nT_Phases, nTs, _ = \
        ForwardProjection.RayDrivenForwProjParCartesianMultipleWithMotionArbitraryOrder(
            x0, dx, Nx, y0, dy, Ny, N, xi0, dxi, M, N2pi, TRot, ThetaArray,
            Heartrate=Heartrate, CM=CM, Plaque=Plaque, Stent=Stent,
            PhantomOld=PhantomOld, Motion=Motion)

    Projections_arbitrary, Angles_rearr, center_points_2, _, _, nTs_arbitrary, _ = \
        ForwardProjection.RayDrivenForwProjParCartesianMultipleWithMotionArbitraryOrder(
            x0, dx, Nx, y0, dy, Ny, N, xi0, dxi, M, N2pi, TRot, ThetaArray_sub,
            Heartrate=Heartrate, CM=CM, Plaque=Plaque, Stent=Stent,
            PhantomOld=PhantomOld, Motion=Motion)

    Projections = np.asarray(Projections).ravel()
    Projections_arbitrary = np.asarray(Projections_arbitrary).ravel()

    if ApplyMoireFilter:
        Projections = MoireFilter.ApplyMoireFilterOnProjections(
            Projections, N, M, dxi, 0.0 * (1 / dxi), 0.5 * (1 / dxi))
        Projections_arbitrary = MoireFilter.ApplyMoireFilterOnProjections(
            Projections_arbitrary, N, M, dxi, 0.0 * (1 / dxi), 0.5 * (1 / dxi))

    Kernel = None
    if Reconkernel == Kernels.SheppLogan:
        Kernel = Kernels.SheppLoganKernelSD(M, dxi)
    elif Reconkernel == Kernels.RamLak:
        Kernel = Kernels.RamLakKernelSD(M, dxi)

    Projections = Convolution.ConvolveSinogram(Projections, Kernel, M, N, dxi)
    Projections_arbitrary = Convolution.ConvolveSinogram(
        Projections_arbitrary, Kernel, M, N, dxi)

    Npi = N2pi // 2
    Nphases = N // Npi
    MultiphaseRecon = Allocation.zeros(Nphases, Nx * Ny)
    MultiphaseRecon_rand = Allocation.zeros(Nphases, Nx * Ny)

    print("Multiphase Reconstruction running...")
    for i in range(1, Nphases + 1):
        theta0_i = theta0 + (i - 1) * Npi * dtheta
        MultiphaseRecon[i - 1, :] = Backprojection.PixDrivenBackProjParCartesian(
            theta0_i, dtheta, Npi, N2pi, xi0, dxi, M,
            Projections[(i - 1) * Npi * M:i * Npi * M], x0, dx, Nx, y0, dy, Ny)
        MultiphaseRecon_rand[i - 1, :] = Backprojection.PixDrivenBackProjParCartesian(
            theta0_i, dtheta, Npi, N2pi, xi0, dxi, M,
            Projections_arbitrary[(i - 1) * Npi * M:i * Npi * M], x0, dx, Nx, y0, dy, Ny)

    MultiphaseReconHU_3dim = (MultiphaseRecon - Constants.MuW) / Constants.MuW * 1000
    MultiphaseReconHU_rand_3dim = \
        (MultiphaseRecon_rand - Constants.MuW) / Constants.MuW * 1000
    PhantomTruthHU_3dim = (np.asarray(PhantomTruth) - Constants.MuW) / Constants.MuW * 1000

    t_Phases = dT * np.asarray(nT_Phases)  # time in s of the discrete time steps

    Diff_all = np.zeros((Nphases, Nx * Ny))
    for i in range(Nphases):
        # MATLAB reshapes to (Nx, Ny) and transposes twice, which is the identity, so the
        # stored difference stays in the row-major image layout.
        Diff_all[i, :] = MultiphaseReconHU_3dim[i, :] - MultiphaseReconHU_rand_3dim[i, :]

    if isplot:
        _ShowPhaseComparison(MultiphaseReconHU_3dim, MultiphaseReconHU_rand_3dim,
                             PhantomTruthHU_3dim, Nphases, Nx, Ny, t_Phases)

    # MATLAB transposes before flattening, so the raw files are phase interleaved.
    MultiphaseReconFlat = MultiphaseRecon.T.ravel()
    MultiphaseReconHU = MultiphaseReconHU_3dim.T.ravel()
    MultiphaseReconHU_rand = MultiphaseReconHU_rand_3dim.T.ravel()
    Diff_all_flat = Diff_all.T.ravel()
    PhantomTruthHU = PhantomTruthHU_3dim.T.ravel()

    Diff_arbitrary = PhantomTruthHU - MultiphaseReconHU_rand
    Diff_linear = PhantomTruthHU - MultiphaseReconHU

    t = np.linspace(0, thetaN / (2 * math.pi) * N2pi / (N2pi / TRot), 1000)
    omega = 2 * math.pi / (60 / Heartrate)
    r = 5
    yDisplacement = 2 * r * np.sin(omega * t)
    # nT_Phases holds one based MATLAB time indices into the sampled center points.
    PhaseIndices = np.asarray(nT_Phases).astype(np.int64).ravel() - 1
    yDisplacement_discrete = np.asarray(center_points_1)[1, PhaseIndices]

    if isplot:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(t, yDisplacement, "-", color="blue")
        plt.plot(np.asarray(nT_Phases) * dT, yDisplacement_discrete, ".", color="red")
        plt.xlabel("time in s")
        plt.ylabel("displacement in mm")
        plt.show()

    if writeData:
        print("Save data")
        rot = NumRot
        tag = f"{Heartrate:.0f}_{TRot:.2f}s_{rot:.0f}rot_{NSubsets:.0f}subsets"
        _Save(str(Outputdirectory) + "\\", f"MultiphaseReconHU_{tag}", MultiphaseReconHU)
        _Save(str(Outputdirectory) + "\\", f"MultiphaseReconHU_rand_{tag}",
              MultiphaseReconHU_rand)
        _Save(str(Outputdirectory) + "\\", f"GroundTruthHU{tag}", PhantomTruthHU)
        _Save(str(Outputdirectory) + "\\", f"Difference_linear_arbitrary{tag}",
              Diff_all_flat)
        _Save(str(Outputdirectory) + "\\", f"Difference_linear_truth{tag}", Diff_linear)
        _Save(str(Outputdirectory) + "\\", f"Difference_arbitrary_truth{tag}",
              Diff_arbitrary)

    return {
        "Outputdirectory": str(Outputdirectory),
        "Projections": Projections,
        "Projections_arbitrary": Projections_arbitrary,
        "Angles_start": Angles_start,
        "Angles_rearr": Angles_rearr,
        "center_points_1": center_points_1,
        "center_points_2": center_points_2,
        "nT_Phases": nT_Phases,
        "nTs": nTs,
        "nTs_arbitrary": nTs_arbitrary,
        "t_Phases": t_Phases,
        "Kernel": Kernel,
        "MultiphaseRecon": MultiphaseRecon,
        "MultiphaseRecon_rand": MultiphaseRecon_rand,
        "MultiphaseReconHU": MultiphaseReconHU_3dim,
        "MultiphaseReconHU_rand": MultiphaseReconHU_rand_3dim,
        "PhantomTruthHU": PhantomTruthHU_3dim,
        "MultiphaseReconFlat": MultiphaseReconFlat,
        "Diff_all": Diff_all,
        "Diff_linear": Diff_linear,
        "Diff_arbitrary": Diff_arbitrary,
        "ThetaArray": ThetaArray,
        "ThetaArray_sub": ThetaArray_sub,
        "ThetaArray_rand": ThetaArray_rand,
        "Geometry": {"RM": RM, "M": M, "N": N, "N2pi": N2pi, "Npi": Npi,
                     "Nphases": Nphases, "NumRot": NumRot, "thetaN": thetaN,
                     "theta0": theta0, "dtheta": dtheta, "dT": dT,
                     "xi0": xi0, "dxi": dxi,
                     "x0": x0, "dx": dx, "Nx": Nx, "y0": y0, "dy": dy, "Ny": Ny,
                     "NSubsets": NSubsets},
    }


def _BuildSubsetOrder(ThetaArray_2, NSubsets):
    """Splits the half turn into NSubsets blocks and reads them round robin.

    The result is extended by the same order shifted by pi to cover the full turn.
    """
    ThetaArray_2 = np.asarray(ThetaArray_2, dtype=float).ravel()
    length = ThetaArray_2.size

    baseSize = length // NSubsets
    remainder = length % NSubsets
    blockSizes = np.full(NSubsets, baseSize, dtype=np.int64)
    blockSizes[:remainder] += 1

    groups = []
    start = 0
    for g in range(NSubsets):
        groups.append(ThetaArray_2[start:start + blockSizes[g]])
        start += blockSizes[g]

    maxLen = int(blockSizes.max())
    ThetaArray_sub = np.zeros(length)
    for k in range(maxLen):
        for g in range(NSubsets):
            if k < blockSizes[g]:
                ThetaArray_sub[k * NSubsets + g] = groups[g][k]

    return np.concatenate((ThetaArray_sub, ThetaArray_sub + math.pi))


def _ShowPhaseComparison(MultiphaseReconHU_3dim, MultiphaseReconHU_rand_3dim,
                         PhantomTruthHU_3dim, Nphases, Nx, Ny, t_Phases):
    import matplotlib.pyplot as plt

    for i in range(Nphases):
        Linear = MultiphaseReconHU_3dim[i].reshape(Ny, Nx)
        Arbitrary = MultiphaseReconHU_rand_3dim[i].reshape(Ny, Nx)
        Truth = PhantomTruthHU_3dim[i].reshape(Ny, Nx)

        fig, axes = plt.subplots(1, 3)
        fig.suptitle("Comparison image linear/arbitrary Scan, "
                     f"time = {t_Phases[i]:.2f}s")
        for ax, Data, title in ((axes[0], Linear, "linear scan"),
                                (axes[1], Arbitrary, "arbitrary scan"),
                                (axes[2], Linear - Arbitrary, "difference")):
            ax.imshow(Data, cmap="gray")
            ax.set_title(title)
            ax.axis("off")

        fig, axes = plt.subplots(2, 2)
        fig.suptitle(f"Comparison to Ground Truth, time = {t_Phases[i]:.2f}s")
        for ax, Data, title in ((axes[0, 0], Linear, "linear scan"),
                                (axes[0, 1], Truth - Linear, ""),
                                (axes[1, 0], Arbitrary, "arbitrary scan"),
                                (axes[1, 1], Truth - Arbitrary, "")):
            ax.imshow(Data, cmap="gray")
            if title:
                ax.set_title(title)
            ax.axis("off")

    plt.show()
