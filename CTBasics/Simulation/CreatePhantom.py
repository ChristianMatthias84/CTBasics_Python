import numpy as np

from ..Memory.Allocation import Allocation
from ..Reconstruction.Constants import Constants
from .Materials import Materials




_MuW = Constants.MuW

# Attenuation values of the dynamic phantoms, given in 1/mm at the 70 keV reference energy.
_MU_ART = 0.019968      # vessel wall,    40 HU
_MU_BLOOD = 0.020256    # blood,          55 HU
_MU_BLOOD_CM = 0.0250   # blood with CM, 300 HU
_MU_PLAQUE = 0.0240     # calcified plaque, 250 HU
_MU_STENT = 0.02592     # stent,          350 HU

_R_WATER = 50.0   # water disc radius in mm
_R_ARTERY = 5.0   # outer artery radius in mm
_R_LUMEN = 4.5    # inner artery radius in mm
_R_STENT = 0.8    # stent strut radius in mm
_N_STENTS = 8

# Bronchi and alveoli of the respiratory phantom:
# (xc0, yc0, a, b, dirX, dirY, amplitude); the two lungs are mirrored in x.
_RESPIRATORY_STRUCTURES = (
    (-70, 20, 7, 5, 0, 1, 8),
    (-85, -10, 6, 6, -1, 0, 6),
    (-55, -25, 8, 4, 1, -1, 5),
    (-80, 30, 3, 3, 0, 1, 4),
    (-60, 10, 4, 3, -1, 1, 5),
    (-90, -30, 3, 4, -1, -1, 6),
    (-55, 35, 3, 2, 1, 0, 3),
    (70, 20, 7, 5, 0, 1, 8),
    (85, -10, 6, 6, 1, 0, 6),
    (55, -25, 8, 4, -1, -1, 5),
    (80, 30, 3, 3, 0, 1, 4),
    (60, 10, 4, 3, 1, 1, 5),
    (90, -30, 3, 4, 1, -1, 6),
    (55, 35, 3, 2, -1, 0, 3),
)


def _Grid(Nx, dx, x0, Ny, dy, y0):
    """Voxel centre coordinates as (Ny, Nx) arrays, matching the MATLAB nx/ny loops."""
    x = x0 + np.arange(1, Nx + 1) * dx
    y = y0 + np.arange(1, Ny + 1) * dy
    return np.meshgrid(x, y)


def _StentCenters(r0):
    """Eight struts spaced equidistantly on the inner artery wall."""
    theta = np.linspace(0, 2 * np.pi, _N_STENTS + 1)[:-1]
    return list(zip(r0 * np.cos(theta), r0 * np.sin(theta)))


def _PaintArtery(Slice, x, y, r, r0, CM, Plaque):
    """Vessel wall, lumen and the selected plaque geometry on the already shifted grid."""
    dist = x * x + y * y

    lumen = dist <= r0 * r0
    Slice[(dist <= r * r) & ~lumen] = _MU_ART
    Slice[lumen] = _MU_BLOOD_CM if CM else _MU_BLOOD

    if Plaque == 0:
        return
    if Plaque == 1:      # half
        Slice[lumen & (x >= 0)] = _MU_PLAQUE
    elif Plaque == 2:    # quarter
        Slice[lumen & (x >= 0) & (y <= 0)] = _MU_PLAQUE
    elif Plaque == 3:    # eighth
        Slice[lumen & (x >= 0) & (y <= 0) & (y < -x)] = _MU_PLAQUE
    elif Plaque == 4:    # elliptical, and not clipped to the lumen in MATLAB
        a, b = 2.5, 4.5
        xcenterPlaque, ycenterPlaque = -1.5, 0.0
        ellipse = ((x - xcenterPlaque) ** 2 / a ** 2
                   + (y - ycenterPlaque) ** 2 / b ** 2) < 1
        Slice[ellipse] = _MU_PLAQUE


def _PaintStents(Slice, X, Y, xDisplacement, yDisplacement, r0):
    for xs, ys in _StentCenters(r0):
        dX = X - (xDisplacement + xs)
        dY = Y - (yDisplacement + ys)
        Slice[dX * dX + dY * dY <= _R_STENT ** 2] = _MU_STENT


# Each phantom is a stack of filled ellipses (mu, a, b, xcenter, ycenter) painted in order,
# so later entries overwrite earlier ones. Spectral phantoms store Materials codes instead
# of attenuation values; the spectral simulation resolves them per energy bin later on.
_PHANTOMS_2D = {
    "WaterPhantom": (False, ((_MuW, 150, 150, 0, 0),)),
    "SimpleEllipse": (False, ((_MuW, 150, 100, 0, 0),)),
    "SimpleEllipseWithHips": (
        False,
        (
            (_MuW, 150, 100, 0, 0),
            (1.8 * _MuW, 25, 20, -100, 0),
            (1.8 * _MuW, 25, 20, 100, 0),
        ),
    ),
    "HipWithMetal": (
        False,
        (
            (_MuW, 150, 100, 0, 0),
            (1.8 * _MuW, 25, 20, -100, 0),
            (1.8 * _MuW, 25, 20, 100, 0),
            (5 * _MuW, 10, 10, -100, 0),
            (5 * _MuW, 10, 10, 100, 0),
        ),
    ),
    "SpectralHipWithMetal": (
        True,
        (
            (Materials.Nyst_water, 150, 100, 0, 0),
            (Materials.Nyst_CorticalBone, 25, 20, -100, 0),
            (Materials.Nyst_CorticalBone, 25, 20, 100, 0),
            (Materials.Nyst_aluminum, 10, 10, -100, 0),
            (Materials.Nyst_aluminum, 10, 10, 100, 0),
        ),
    ),
    "SpectralHip": (
        True,
        (
            (Materials.Nyst_water, 150, 100, 0, 0),
            (Materials.Nyst_CorticalBone, 25, 20, -100, 0),
            (Materials.Nyst_CorticalBone, 25, 20, 100, 0),
            (Materials.Nyst_water, 20, 15, -100, 0),
            (Materials.Nyst_water, 20, 15, 100, 0),
        ),
    ),
    "SpectralMetalOnly": (
        True,
        (
            (Materials.Nyst_Iron, 15, 15, -100, 0),
            (Materials.Nyst_Iron, 15, 15, 100, 0),
        ),
    ),
    "WaterPhantomSpectral30": (True, ((Materials.Nyst_water, 150, 150, 0, 0),)),
    "WaterPhantomSpectral20_CBinsert": (
        True,
        (
            (Materials.Nyst_water, 100, 100, 0, 0),
            (Materials.Nyst_CorticalBone, 18, 18, 0, 0),
        ),
    ),
    "WaterPhantomSpectral40": (True, ((Materials.Nyst_water, 200, 200, 0, 0),)),
    "WaterPhantomSpectral40_CBinsert": (
        True,
        (
            (Materials.Nyst_water, 200, 200, 0, 0),
            (Materials.Nyst_CorticalBone, 18, 18, 0, 0),
        ),
    ),
    "SpectralWaterEllipse": (True, ((Materials.Nyst_water, 150, 100, 0, 0),)),
}


class CreatePhantom:
    @staticmethod
    def _EllipseMask(Nx, dx, x0, Ny, dy, y0, a, b, xcenter, ycenter):
        """Boolean (Ny, Nx) mask of the filled ellipse, matching the MATLAB voxel test.

        MATLAB samples voxel centres at x0 + nx*dx for nx = 1..Nx and accepts a voxel when
        |y| <= |b/a*sqrt(a^2 - x^2)|, i.e. a hard in/out decision without partial volume.
        """
        x = x0 + np.arange(1, Nx + 1) * dx - xcenter
        y = y0 + np.arange(1, Ny + 1) * dy - ycenter
        X, Y = np.meshgrid(x, y)

        half_height = np.abs(b / a * np.sqrt(np.clip(a * a - X * X, 0.0, None)))
        return (np.abs(X) <= a) & (np.abs(Y) <= half_height)

    @staticmethod
    def CreateVoxelPhantom2D(Nx, dx, x0, Ny, dy, y0, SelectedPhantom_in):
        if SelectedPhantom_in == "SimpleEllipseWithHips3D":
            # MATLAB defines this case inside CreateVoxelPhantom2D but indexes Nz, which is
            # not a parameter there, so the MATLAB call fails. Use CreateVoxelPhantom3D.
            raise ValueError("SimpleEllipseWithHips3D is a 3D phantom; use CreateVoxelPhantom3D.")

        if SelectedPhantom_in not in _PHANTOMS_2D:
            raise ValueError(f"Unknown phantom: {SelectedPhantom_in}")

        isSpectral, ellipses = _PHANTOMS_2D[SelectedPhantom_in]

        Phantom = Allocation.zeros(Nx * Ny)
        Slice = Phantom.reshape(Ny, Nx)
        for mu, a, b, xcenter, ycenter in ellipses:
            Slice[CreatePhantom._EllipseMask(Nx, dx, x0, Ny, dy, y0, a, b, xcenter, ycenter)] = mu

        return Phantom, isSpectral

    @staticmethod
    def CreateVoxelPhantom3D(
        SelectedPhantom_in,
        SimulationGeometry_in=None,
        Nx=None, dx=None, x0=None,
        Ny=None, dy=None, y0=None,
        Nz=None, dz=None, z0=None,
        zstart=None, zstop=None,
    ):
        if Nx is None and SimulationGeometry_in is not None:
            Nx = SimulationGeometry_in.Nxsim
            dx = SimulationGeometry_in.dxsim
            x0 = SimulationGeometry_in.x0sim
            Ny = SimulationGeometry_in.Nysim
            dy = SimulationGeometry_in.dysim
            y0 = SimulationGeometry_in.y0sim
            Nz = SimulationGeometry_in.Nzsim
            dz = SimulationGeometry_in.dzsim
            z0 = SimulationGeometry_in.z0sim
            zstart = SimulationGeometry_in.zStartsim
            zstop = SimulationGeometry_in.zStopsim

        if SelectedPhantom_in != "SimpleEllipseWithHips3D":
            raise ValueError(f"Unknown 3D phantom: {SelectedPhantom_in}")

        isSpectral = False
        mu_w = Constants.MuW
        mu_b = 1.8 * mu_w

        Phantom = Allocation.zeros(Nx * Ny * Nz)
        Volume = Phantom.reshape(Nz, Ny, Nx)

        zstart = int(zstart)
        zstop = int(zstop)

        for nz in range(zstart, zstop + 1):
            Volume[nz - 1][CreatePhantom._EllipseMask(Nx, dx, x0, Ny, dy, y0, 150, 100, 0, 0)] = mu_w

        # The left hip drifts in x with increasing slice index.
        for nz in range(zstart, zstop + 1):
            xcenter = -100 + nz * 0.5
            Volume[nz - 1][CreatePhantom._EllipseMask(Nx, dx, x0, Ny, dy, y0, 20, 15, xcenter, 0)] = mu_b

        # The right hip tapers with increasing slice index.
        for nz in range(zstart, zstop + 1):
            a = 25 - nz * 0.5
            b = 20 - nz * 0.5
            if a <= 0 or b <= 0:
                continue
            Volume[nz - 1][CreatePhantom._EllipseMask(Nx, dx, x0, Ny, dy, y0, a, b, 100, 0)] = mu_b

        # The spine alternates between vertebra and gap every `height` millimetres.
        height = 5.0
        isSpine = True
        deltaz = 0.0
        for nz in range(zstart, zstop + 1):
            deltaz += dz
            if deltaz >= height:
                isSpine = not isSpine
                deltaz = 0.0
            if isSpine:
                Volume[nz - 1][CreatePhantom._EllipseMask(Nx, dx, x0, Ny, dy, y0, 18, 12, 0, -50)] = mu_b

        # Central high contrast ellipsoid with semi-axes (10, 10, 5) mm.
        a = b = 10.0
        c = 5.0
        x = x0 + np.arange(1, Nx + 1) * dx
        y = y0 + np.arange(1, Ny + 1) * dy
        X, Y = np.meshgrid(x, y)
        for nz in range(zstart, zstop + 1):
            z = z0 + (nz - 1) * dz
            inside = (X * X / (a * a) + Y * Y / (b * b) + z * z / (c * c)) <= 1
            Volume[nz - 1][(np.abs(X) <= a) & inside] = mu_b

        return Phantom, isSpectral

    @staticmethod
    def MotionProfile(t, HeartRate, Motion, MotionType="full", TRest=0.0):
        """Normalized cardiac displacement in [-1, 1]; zero during the resting phase."""
        if not Motion:
            return 0.0

        if TRest is None:
            TRest = 0.0
        if MotionType is None:
            MotionType = "full"

        Tcycle = 60.0 / HeartRate
        TMotion = Tcycle - TRest

        if TMotion <= 0:
            raise ValueError(
                f"TRest ({TRest:.3f} s) must be smaller than the cycle duration ({Tcycle:.3f} s)."
            )

        tLocal = np.mod(t, Tcycle)

        if tLocal >= TMotion:
            return 0.0

        if MotionType == "full":
            return np.sin(2 * np.pi * tLocal / TMotion)
        if MotionType == "half":
            return np.sin(np.pi * tLocal / TMotion)

        raise ValueError(f'Unknown MotionType "{MotionType}". Use "full" or "half".')

    @staticmethod
    def CreateVoxelPhantom2DWithMotionCardiac(Nx, dx, x0, Ny, dy, y0, T0, dT, nT,
                                              Heartrate=60, CM=False, Stent=False,
                                              Plaque=0, Motion=True, Direction=0):
        """Water disc with a moving coronary artery, optional contrast, plaque and stents."""
        HeartBeatCycleDuration = 60.0 / Heartrate
        t = T0 + nT * dT

        isSpectral = False

        Phantom = Allocation.zeros(Nx * Ny)
        Slice = Phantom.reshape(Ny, Nx)

        X, Y = _Grid(Nx, dx, x0, Ny, dy, y0)

        Slice[X * X + Y * Y <= _R_WATER ** 2] = _MuW

        # Displacement model follows the LR/PA motion reported in PMC2396264.
        omega = 2 * np.pi / HeartBeatCycleDuration
        norm_displacement = np.sin(omega * t) if Motion else 0.0

        xDisplacement = 2 * _R_ARTERY * norm_displacement * np.sin(Direction)
        yDisplacement = 2 * _R_ARTERY * norm_displacement * np.cos(Direction)

        _PaintArtery(Slice, X - xDisplacement, Y - yDisplacement,
                     _R_ARTERY, _R_LUMEN, CM, Plaque)

        if Stent:
            _PaintStents(Slice, X, Y, xDisplacement, yDisplacement, _R_LUMEN)

        return Phantom, isSpectral, xDisplacement, yDisplacement

    @staticmethod
    def CreateVoxelPhantom2DWithMotionUnified(Nx, dx, x0, Ny, dy, y0, T0, dT, nT,
                                              SelectedPhantom, CM=False, Stent=False,
                                              Plaque=0, Heartrate=60, Motion=True,
                                              Direction=0, MotionType="full", TRest=0,
                                              LineWidth=2.0, Rotation=0.0, Center=(0, 0),
                                              IodinePeakFactor=3.0, IodineSigmaMin=0.25,
                                              IodineSigmaMax=4.5):
        """Dispatches to the dynamic phantoms that share one global motion profile."""
        t = T0 + nT * dT

        norm_disp = CreatePhantom.MotionProfile(t, Heartrate, Motion, MotionType, TRest)

        amp = 10  # global amplitude in mm
        xDisplacement = amp * norm_disp * np.sin(Direction)
        yDisplacement = amp * norm_disp * np.cos(Direction)

        Phantom = Allocation.zeros(Nx * Ny)
        Slice = Phantom.reshape(Ny, Nx)
        isSpectral = False
        RoI = np.array([[xDisplacement], [yDisplacement]], dtype=float)

        X, Y = _Grid(Nx, dx, x0, Ny, dy, y0)
        water = X * X + Y * Y <= _R_WATER ** 2

        if SelectedPhantom == "WaterOnly":
            # Deliberately static so a ROI measures pure noise.
            Slice[water] = _MuW

        elif SelectedPhantom == "Cardiac":
            Slice[water] = _MuW
            _PaintArtery(Slice, X - xDisplacement, Y - yDisplacement,
                         _R_ARTERY, _R_LUMEN, CM, Plaque)
            if Stent:
                _PaintStents(Slice, X, Y, xDisplacement, yDisplacement, _R_LUMEN)

        elif SelectedPhantom == "TemporalIodinePhantom":
            # Only the iodine concentration changes; the vessels stay in place.
            xDisplacement = 0
            yDisplacement = 0
            RoI = np.array([[0.0], [0.0]])

            mu_art = (1 + 30 / 1000) * _MuW
            mu_blood = (1 + 55 / 1000) * _MuW
            vesselWallThickness = 0.5

            Slice[water] = _MuW

            vesselRadii = (1, 2, 3, 5)
            vesselCenterX = (-15, 15, -15, 15)
            vesselCenterY = (-15, -15, 15, 15)

            TcycleCM = 60.0 / Heartrate
            phaseBase = np.pi * np.mod(t, TcycleCM) / TcycleCM
            mu_iodine_max = IodinePeakFactor * _MuW

            for iVessel in range(1, 5):
                phase = np.mod(phaseBase + (iVessel - 1) * np.pi / 8, np.pi)
                cmNorm = np.sin(phase)

                sigmaCM = IodineSigmaMin + (IodineSigmaMax - IodineSigmaMin) * cmNorm
                sigmaCM = max(sigmaCM, np.finfo(float).eps)

                rVessel = vesselRadii[iVessel - 1]
                rWallOuter = rVessel + vesselWallThickness

                dX = X - vesselCenterX[iVessel - 1]
                dY = Y - vesselCenterY[iVessel - 1]
                dist2 = dX * dX + dY * dY

                Slice[(dist2 <= rWallOuter ** 2) & (dist2 > rVessel ** 2)] = mu_art

                lumen = dist2 <= rVessel ** 2
                iodine = mu_iodine_max * cmNorm * np.exp(-dist2 / (2 * sigmaCM ** 2))
                Slice[lumen] = mu_blood + iodine[lumen]

        elif SelectedPhantom == "WaterWith4Arteries":
            Slice[water] = _MuW

            arteryOrbit = 25
            arteryAngles = np.array([np.pi / 4, 3 * np.pi / 4, 5 * np.pi / 4, 7 * np.pi / 4])
            xArteryCenter = arteryOrbit * np.cos(arteryAngles)
            yArteryCenter = arteryOrbit * np.sin(arteryAngles)

            # Artery 1 carries stents, arteries 2..4 carry half/quarter/eighth plaque.
            for iArtery in range(1, 5):
                xLocal = X - xDisplacement - xArteryCenter[iArtery - 1]
                yLocal = Y - yDisplacement - yArteryCenter[iArtery - 1]
                dist = xLocal * xLocal + yLocal * yLocal

                lumen = dist <= _R_LUMEN ** 2

                Slice[(dist <= _R_ARTERY ** 2) & ~lumen] = _MU_ART
                Slice[lumen] = _MU_BLOOD_CM if CM else _MU_BLOOD

                if iArtery == 2:
                    Slice[lumen & (xLocal >= 0)] = _MU_PLAQUE
                elif iArtery == 3:
                    Slice[lumen & (xLocal >= 0) & (yLocal <= 0)] = _MU_PLAQUE
                elif iArtery == 4:
                    # MATLAB negates the quarter condition here, so this covers 3/4 of the lumen.
                    Slice[lumen & ~((xLocal >= 0) & (yLocal <= 0))] = _MU_PLAQUE
                elif iArtery == 1:
                    for xs, ys in _StentCenters(_R_LUMEN):
                        dXs = xLocal - xs
                        dYs = yLocal - ys
                        Slice[lumen & (dXs * dXs + dYs * dYs <= _R_STENT ** 2)] = _MU_STENT

            RoI = np.vstack([xArteryCenter + xDisplacement, yArteryCenter + yDisplacement])

        elif SelectedPhantom == "LinePattern":
            mu_line = _MU_PLAQUE

            cx, cy = Center[0], Center[1]
            cosA = np.cos(Rotation)
            sinA = np.sin(Rotation)

            halfWidth = LineWidth / 2
            halfLength = 2.5 * LineWidth
            centers = (-3 * LineWidth, -LineWidth, LineWidth, 3 * LineWidth)

            # MATLAB centres the water disc on `Center` as well as the line pattern.
            inside = (X - cx) ** 2 + (Y - cy) ** 2 <= _R_WATER ** 2
            Slice[inside] = _MuW

            x1 = (X - cx) - xDisplacement
            y1 = (Y - cy) - yDisplacement
            x_rot = cosA * x1 + sinA * y1
            y_rot = -sinA * x1 + cosA * y1

            for x_i in centers:
                stripe = (np.abs(x_rot - x_i) <= halfWidth) & (np.abs(y_rot) <= halfLength)
                Slice[stripe & (np.abs(X - cx) <= _R_WATER)] = mu_line

        elif SelectedPhantom == "Respiratory":
            # Heartrate is reused as the breathing rate in cycles per minute.
            BreathingCycleDuration = 60.0 / Heartrate

            mu_lung = 0.0
            mu_tissue = _MuW

            Slice[CreatePhantom._EllipseMask(Nx, dx, x0, Ny, dy, y0, 150, 100, 0, 0)] = _MuW

            xcL, xcR = -70, 70
            Slice[CreatePhantom._EllipseMask(Nx, dx, x0, Ny, dy, y0, 55, 45, xcL, 0)] = mu_lung
            Slice[CreatePhantom._EllipseMask(Nx, dx, x0, Ny, dy, y0, 55, 45, xcR, 0)] = mu_lung

            nStructures = len(_RESPIRATORY_STRUCTURES)
            for iStr, (xc0_i, yc0_i, a_i, b_i, dirX, dirY, amp_i) in enumerate(
                    _RESPIRATORY_STRUCTURES, start=1):

                dirLen = np.hypot(dirX, dirY)
                if dirLen > 0:
                    dirX, dirY = dirX / dirLen, dirY / dirLen

                phaseOffset = (iStr - 1) * np.pi / nStructures
                norm_i = np.sin(2 * np.pi * t / BreathingCycleDuration + phaseOffset) if Motion else 0.0

                xcStr = xc0_i + amp_i * norm_i * dirX
                ycStr = yc0_i + amp_i * norm_i * dirY

                dX = X - xcStr
                dY = Y - ycStr
                Slice[(np.abs(dX) <= a_i) &
                      (dX * dX / (a_i * a_i) + dY * dY / (b_i * b_i) <= 1)] = mu_tissue

            RoI = np.array([[xcL, xcR], [0.0, 0.0]], dtype=float)

        else:
            raise ValueError("Unknown phantom type!")

        return Phantom, isSpectral, xDisplacement, yDisplacement, RoI

    @staticmethod
    def CreateVoxelPhantom2DWithMotionRespiratory(Nx, dx, x0, Ny, dy, y0, T0, dT, nT,
                                                  SelectedPhantom_in):
        """Large-scale cardiac and lung phantoms driven by a fixed-period sine."""
        t = T0 + nT * dT

        Phantom = Allocation.zeros(Nx * Ny)
        Slice = Phantom.reshape(Ny, Nx)
        isSpectral = False

        X, Y = _Grid(Nx, dx, x0, Ny, dy, y0)

        xpoint = 0.0
        ypoint = 0.0

        if SelectedPhantom_in in ("CardiacPhantom", "CardiacPhantomWithPlaque",
                                  "CardiacPhantomMultipleArteries"):
            HeartBeatCycleDuration = 1.0
            Normalizeddisplacement = np.sin(t / HeartBeatCycleDuration * 2 * np.pi)

            Slice[X * X + Y * Y <= 200 ** 2] = _MuW

            r, r0 = 30.0, 25.0

            if SelectedPhantom_in == "CardiacPhantom":
                mu_plaque = 0.02112
                motionrangex, motionrangey = 2.0, 10.0
                xcenter = -50 + motionrangex * t / HeartBeatCycleDuration * 2 * np.pi
                ycenter = motionrangey * Normalizeddisplacement

                dX, dY = X - xcenter, Y - ycenter
                dist = np.hypot(dX, dY)
                Slice[(dist <= r) & (dist > r0)] = _MU_ART
                Slice[dist <= r0] = _MU_BLOOD

                # MATLAB leaves xpoint at the current time and ypoint at the last grid row.
                xpoint = t
                ypoint = Y[-1, -1]

            elif SelectedPhantom_in == "CardiacPhantomWithPlaque":
                mu_plaque = 0.0240
                xcenter = -50.0
                ycenter = 2 * r * Normalizeddisplacement

                dX, dY = X - xcenter, Y - ycenter
                dist = np.hypot(dX, dY)
                Slice[(dist <= r) & (dist > r0)] = _MU_ART
                Slice[dist <= r0] = _MU_BLOOD
                # Half plaque on the positive x side of the lumen.
                Slice[(dX > 0) & (np.abs(dX) <= r0) & (dX * dX + dY * dY <= r0 * r0)] = mu_plaque

            else:
                r_arteries = 125.0
                n_arteries = 8
                theta = np.linspace(0, 2 * np.pi, n_arteries + 1)[:-1]

                for i in range(n_arteries):
                    xcenter_i = r_arteries * np.cos(theta[i]) + np.cos(theta[i]) * Normalizeddisplacement * 10
                    ycenter_i = r_arteries * np.sin(theta[i]) + np.sin(theta[i]) * Normalizeddisplacement * 10

                    dist = np.hypot(X - xcenter_i, Y - ycenter_i)
                    Slice[(dist <= r) & (dist > r0)] = _MU_ART
                    Slice[dist <= r0] = _MU_BLOOD

        elif SelectedPhantom_in == "SimpleLungWithTumor4D":
            BreathingCycleDuration = 4.0
            Normalizeddisplacement = np.sin(t / BreathingCycleDuration * 2 * np.pi)

            mu_Lung = 0.1 * _MuW
            mu_Tumor = _MuW
            MotionRange = 10.0

            def paint(a, b, xcenter, ycenter, mu):
                Slice[CreatePhantom._EllipseMask(Nx, dx, x0, Ny, dy, y0, a, b, xcenter, ycenter)] = mu

            paint(150, 100, 0, 0, _MuW)
            paint(60, 50, -75, 0, mu_Lung)
            # The left tumour moves in x, the right one in y.
            paint(10, 10, -75 + MotionRange * Normalizeddisplacement, 0, mu_Tumor)
            paint(60, 50, 75, 0, mu_Lung)
            paint(10, 10, 75, MotionRange * Normalizeddisplacement, mu_Tumor)

        else:
            raise ValueError(f"Unknown phantom: {SelectedPhantom_in}")

        return Phantom, isSpectral, xpoint, ypoint

    @staticmethod
    def CreateVoxelPhantomCardiac2DWithMotion(Nx, dx, x0, Ny, dy, y0, T0, dT, nT,
                                              ContrastMedium=False, Stent=False,
                                              PlaqueType="none"):
        """Large water disc with one moving artery; contrast medium is the only active option.

        MATLAB parses Stent and PlaqueType but never uses them, and it reads an undefined
        `ContrastFlag` instead of the parsed `ContrastMedium`, so the MATLAB function always
        errors. The port uses the parsed option, which is the evident intent.
        """
        if PlaqueType not in ("none", "half", "quarter", "eighth", "elliptical"):
            raise ValueError(f"Invalid PlaqueType: {PlaqueType}")

        HeartBeatCycleDuration = 1.0
        t = T0 + nT * dT
        Normalizeddisplacement = np.sin(t / HeartBeatCycleDuration * 2 * np.pi)

        Phantom = Allocation.zeros(Nx * Ny)
        Slice = Phantom.reshape(Ny, Nx)
        isSpectral = False

        X, Y = _Grid(Nx, dx, x0, Ny, dy, y0)
        Slice[X * X + Y * Y <= 200 ** 2] = _MuW

        r, r0 = 30.0, 25.0
        motionrangex, motionrangey = 2.0, 10.0
        xcenter = -50 + motionrangex * t / HeartBeatCycleDuration * 2 * np.pi
        ycenter = motionrangey * Normalizeddisplacement

        dist = np.hypot(X - xcenter, Y - ycenter)
        Slice[(dist <= r) & (dist > r0)] = _MU_ART
        Slice[dist <= r0] = _MU_BLOOD_CM if ContrastMedium else _MU_BLOOD

        return Phantom, isSpectral, t, Y[-1, -1]
