"""Writes the simulation parameter report of a dual source experiment.

Port of Projects/Static CT/Projects AH/DualSource/Shared/DualSource_writeSimulationParameters.m.
"""
from pathlib import Path


def DualSource_writeSimulationParameters(
        Outputdirectory, Nsources, TRot, Heartrate, dT,
        RM, M, Npar, N2pi, dtheta, Nx, Ny, dx, dy,
        Shift_GT, Shift_GT_Rest, Rotation, Direction, MotionType, TRest,
        ApplyMoireFilter, SimNoise, N_I0, naming=None):
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
