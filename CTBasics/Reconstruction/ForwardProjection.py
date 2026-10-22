import numba
import numpy as np

from CTBasics.Memory.Allocation import Allocation
from CTBasics.MultiCore.Parallelization import Parallelization


def _fix(value):
    """MATLAB fix(): truncate toward zero."""
    return int(value)


def _fix_array(values):
    """Vectorized MATLAB fix() returning integer indices."""
    return np.trunc(values).astype(np.int64)


# The numba kernels below loop like the MATLAB originals and sum every ray sequentially,
# so they match MATLAB's summation order. Rays are independent and run as threads.

@numba.njit(parallel=True, cache=True)
def _ProjectParallelViews(c, s, xi0, dxi, M, x0, dx, Nx, y0, dy, Ny, Image, Projections):
    """Parallel beam views with direction (c[n], s[n]) into Projections[n, :]."""
    for k in numba.prange(c.size * M):
        n = k // M
        m = k % M + 1
        xi = xi0 + m * dxi
        acc = 0.0
        # Traverse along the axis the ray crosses most steeply so that exactly one
        # interpolated sample is taken per row resp. column.
        if abs(c[n]) < abs(s[n]):
            for nx in range(1, Nx + 1):
                ny_real = ((xi - (x0 + nx * dx) * c[n]) / s[n] - y0) / dy
                ny = int(ny_real)
                if 1 <= ny < Ny:
                    w = ny_real - ny
                    acc += (1 - w) * Image[(ny - 1) * Nx + nx - 1] + w * Image[ny * Nx + nx - 1]
            Projections[n, m - 1] = acc * abs(dx / s[n])
        else:
            for ny in range(1, Ny + 1):
                nx_real = ((xi - (y0 + ny * dy) * s[n]) / c[n] - x0) / dx
                nx = int(nx_real)
                if 1 <= nx < Nx:
                    w = nx_real - nx
                    acc += (1 - w) * Image[(ny - 1) * Nx + nx - 1] + w * Image[(ny - 1) * Nx + nx]
            Projections[n, m - 1] = acc * abs(dy / c[n])


@numba.njit(parallel=True, cache=True)
def _TraceParallelRays(xi, c, s, keep, x0, dx, Nx, y0, dy, Ny, Ima, Projection):
    """Integrates every kept ray given in the parallel parametrisation xi = x*c + y*s.

    Shared by the fan and static CT projectors; all arguments are flat per-ray arrays.
    """
    for k in numba.prange(xi.size):
        if not keep[k]:
            continue
        acc = 0.0
        if abs(c[k]) < abs(s[k]):
            for nx in range(1, Nx + 1):
                ny_real = (xi[k] - x0 * c[k] - nx * dx * c[k] - y0 * s[k]) / (dy * s[k])
                ny = int(ny_real)
                if 1 <= ny < Ny:
                    w = ny_real - ny
                    acc += (1 - w) * Ima[(ny - 1) * Nx + nx - 1] + w * Ima[ny * Nx + nx - 1]
            Projection[k] = acc * abs(dx / s[k])
        else:
            for ny in range(1, Ny + 1):
                nx_real = (xi[k] - y0 * s[k] - ny * dy * s[k] - x0 * c[k]) / (dx * c[k])
                nx = int(nx_real)
                if 1 <= nx < Nx:
                    w = nx_real - nx
                    acc += (1 - w) * Ima[(ny - 1) * Nx + nx - 1] + w * Ima[(ny - 1) * Nx + nx]
            Projection[k] = acc * abs(dy / c[k])


def _Image(Image):
    return np.ascontiguousarray(Image, dtype=float).ravel()


def _ProjectParallel(theta, x0, dx, Nx, y0, dy, Ny, Image, xi0, dxi, M):
    """(N, M) projections of the parallel beam views theta."""
    theta = np.atleast_1d(np.asarray(theta, dtype=float))
    Projections = Allocation.zeros(theta.size, M)
    _ProjectParallelViews(np.cos(theta), np.sin(theta), float(xi0), float(dxi), int(M),
                          float(x0), float(dx), int(Nx), float(y0), float(dy), int(Ny),
                          _Image(Image), Projections)
    return Projections


def _TraceRays(xi, c, s, keep, x0, dx, Nx, y0, dy, Ny, Ima):
    """Line integrals of the rays (xi, c, s); rays outside `keep` stay 0."""
    Projection = Allocation.zeros(xi.size)
    _TraceParallelRays(xi.ravel(), c.ravel(), s.ravel(), keep.ravel(), float(x0), float(dx),
                       int(Nx), float(y0), float(dy), int(Ny), _Image(Ima), Projection)
    return Projection


def _FanRays(RF, alpha, beta, M):
    """(xi, c, s) of the fan rays beta[:M] for the source angles alpha, shape (N, M)."""
    alpha = np.atleast_1d(np.asarray(alpha, dtype=float))[:, None]
    beta = np.asarray(beta, dtype=float).ravel()[None, :M]

    sx = RF * np.sin(alpha)
    sy = -RF * np.cos(alpha)
    s = np.sin(alpha + beta)
    c = np.cos(alpha + beta)
    # Orthogonal projection of the source vector onto the xi axis; equals -RF*sin(beta).
    return sx * c + sy * s, c, s


def _StaticCTRays(RF, RD, alpha, delta0, dDelta, MDelta):
    """(xi, c, s, keep) of the static CT detector rays for the source angles alpha."""
    alpha = np.atleast_1d(np.asarray(alpha, dtype=float))[:, None]

    sx = RF * np.sin(alpha)
    sy = -RF * np.cos(alpha)

    # For static CT the detector raster is the betaDash raster.
    delta = (delta0 + np.arange(1, MDelta + 1) * dDelta)[None, :]
    betaDash = delta - alpha - np.pi / 2.0
    det_x = np.cos(delta)
    det_y = np.sin(delta)
    beta = np.arctan((np.sin(betaDash) * RD) / (np.cos(betaDash) * RD + RF))
    s = np.sin(alpha + beta)
    c = np.cos(alpha + beta)
    eta_x = -s
    eta_y = c

    # Limits the fan to rays within pi/2 of the detector element normal.
    keep = np.abs(np.arccos(np.clip(eta_x * det_x + eta_y * det_y, -1.0, 1.0))) < np.pi / 2.0

    return sx * c + sy * s, c, s, keep




def _SpectralBins(spectrum, E0, N_Ebins, deltaE):
    """Material properties plus the sampled tube spectrum of the requested energy bins."""
    from CTBasics.Simulation.MaterialProperties import MaterialProperties
    from CTBasics.Simulation.SpectralProperties import _SpectralBins as _Bins

    I0Values, SumOverI0 = _Bins(spectrum, E0, N_Ebins, deltaE)
    return MaterialProperties(), I0Values, SumOverI0


def _SpectralAttenuation(SpectralProjections, I0Values, SumOverI0):
    """Spectrally weighted intensity average turned back into a line integral."""
    Intensity = (I0Values[:, None] * np.exp(-SpectralProjections)).sum(axis=0) / SumOverI0
    return -np.log(Intensity)


@numba.njit(cache=True, error_model="numpy")
def _ConeRaySum(xi, B, c, s, A, cos_gamma, zsource_over_dz, z0_over_dz,
                x0, dx, Nx, y0, dy, Ny, dz, Nz, Ima):
    """Line integral of one cone-beam ray, given in the parallel parametrisation xi.

    `A` converts the in-plane distance from the source into the z offset of the ray. The
    z position follows the MATLAB expression nz_real = zsource/dz + (B-x*s+y*s)*A/dz - z0/dz,
    the samples are bilinear in the traversed in-plane axis and in z.
    """
    NxNy = Nx * Ny
    tempsum = 0.0
    if abs(c) < abs(s):
        dy_s = dy * s
        y0_s = y0 * s
        for nx in range(1, Nx + 1):
            x = x0 + nx * dx
            ny_real = (xi - x * c - y0_s) / dy_s
            # 1 <= fix(v) < N is the same as 1 <= v < N and also rejects NaN.
            if 1 <= ny_real < Ny:
                ny = int(ny_real)
                y = y0 + ny_real * dy
                nz_real = zsource_over_dz + ((B - x * s + y * s) * A) / dz - z0_over_dz
                if 1 <= nz_real < Nz:
                    nz = int(nz_real)
                    wy = ny_real - ny
                    wz = nz_real - nz
                    k = (nz - 1) * NxNy + (ny - 1) * Nx + nx - 1
                    pnz = (1 - wy) * Ima[k] + wy * Ima[k + Nx]
                    pnz_plus1 = (1 - wy) * Ima[k + NxNy] + wy * Ima[k + NxNy + Nx]
                    tempsum = tempsum + (1 - wz) * pnz + wz * pnz_plus1
        return tempsum * abs(dx / (s * cos_gamma))

    dx_c = dx * c
    x0_c = x0 * c
    for ny in range(1, Ny + 1):
        y = y0 + ny * dy
        nx_real = (xi - y * s - x0_c) / dx_c
        if 1 <= nx_real < Nx:
            nx = int(nx_real)
            x = x0 + nx_real * dx
            nz_real = zsource_over_dz + ((B - x * s + y * s) * A) / dz - z0_over_dz
            if 1 <= nz_real < Nz:
                nz = int(nz_real)
                wx = nx_real - nx
                wz = nz_real - nz
                k = (nz - 1) * NxNy + (ny - 1) * Nx + nx - 1
                pnz = (1 - wx) * Ima[k] + wx * Ima[k + 1]
                pnz_plus1 = (1 - wx) * Ima[k + NxNy] + wx * Ima[k + NxNy + 1]
                tempsum = tempsum + (1 - wz) * pnz + wz * pnz_plus1
    return tempsum * abs(dy / (c * cos_gamma))


@numba.njit(parallel=True, cache=True, error_model="numpy")
def _FanConeViews(alpha, zsource, beta, RF, s0, ds, M, Ns,
                  x0, dx, Nx, y0, dy, Ny, z0, dz, Nz, Ima, Projections):
    """Cone-beam fan views alpha[n] with focus height zsource[n] into Projections[n, :].

    Every view holds Ns rows of M channels; each ray is one thread.
    """
    RFsquare = RF * RF
    z0_over_dz = z0 / dz
    for k in numba.prange(alpha.size * Ns * M):
        n = k // (Ns * M)
        ray = k % (Ns * M)
        ns = ray // M + 1
        m = ray % M + 1

        sx = RF * np.sin(alpha[n])
        sy = -RF * np.cos(alpha[n])
        A = ds / (RF * 2.0) * (2 * ns - Ns - 1)
        sl = s0 + ns * ds
        cos_gamma = np.cos(abs(np.arctan(sl / RF)))

        s = np.sin(alpha[n] + beta[m - 1])
        c = np.cos(alpha[n] + beta[m - 1])
        xi = sx * c + sy * s
        B = np.sqrt(RFsquare - xi * xi)

        Projections[n, ray] = _ConeRaySum(xi, B, c, s, A, cos_gamma, zsource[n] / dz,
                                          z0_over_dz, x0, dx, Nx, y0, dy, Ny, dz, Nz, Ima)


@numba.njit(parallel=True, cache=True, error_model="numpy")
def _StaticCTConeViews(alpha, zsource, RF, RD, delta0, dDelta, MDelta, s0, ds, soffset, Ns,
                       x0, dx, Nx, y0, dy, Ny, z0, dz, Nz, Ima, Projections):
    """Static CT cone-beam views alpha[n] with focus height zsource[n] into Projections[n, :].

    The detector elements sit on the ring at RD; elements that a ray would hit from behind
    keep their zero.
    """
    RFsquare = RF * RF
    RFsquarePlusRDsquare = RFsquare + RD * RD
    TwoRDRF = 2 * RD * RF
    z0_over_dz = z0 / dz
    for k in numba.prange(alpha.size * Ns * MDelta):
        n = k // (Ns * MDelta)
        ray = k % (Ns * MDelta)
        ns = ray // MDelta + 1
        m = ray % MDelta + 1

        delta = delta0 + m * dDelta
        betaDash = delta - alpha[n] - np.pi / 2
        beta = np.arctan((np.sin(betaDash) * RD) / (np.cos(betaDash) * RD + RF))
        s = np.sin(alpha[n] + beta)
        c = np.cos(alpha[n] + beta)

        # Arbitrary fan limitation: the ray must not hit the detector element from behind.
        cos_angle = min(max(-s * np.cos(delta) + c * np.sin(delta), -1.0), 1.0)
        if abs(np.arccos(cos_angle)) >= np.pi / 2:
            continue

        sx = RF * np.sin(alpha[n])
        sy = -RF * np.cos(alpha[n])
        A_nom = soffset + ds * (ns - Ns - 0.5)
        sl = s0 + ns * ds
        # L in the geometry description: source-to-element distance on the ring.
        A_denom = np.sqrt(RFsquarePlusRDsquare + TwoRDRF * np.sin(delta - alpha[n]))
        A = A_nom / A_denom
        cos_gamma = np.cos(abs(np.arctan(sl / A_denom)))

        xi = sx * c + sy * s
        B = np.sqrt(RFsquare - xi * xi)

        Projections[n, ray] = _ConeRaySum(xi, B, c, s, A, cos_gamma, zsource[n] / dz,
                                          z0_over_dz, x0, dx, Nx, y0, dy, Ny, dz, Nz, Ima)


def _FanCone(alpha, zsource, RF, beta, s0, ds, x0, dx, Nx, y0, dy, Ny, z0, dz, Nz,
             Ima, M, Ns):
    """Flat (n, ns, m) projections of the fan views alpha with focus heights zsource."""
    alpha = np.atleast_1d(np.asarray(alpha, dtype=float))
    zsource = np.broadcast_to(np.asarray(zsource, dtype=float), alpha.shape).copy()
    Projections = Allocation.zeros(alpha.size, M * Ns)
    _FanConeViews(alpha, zsource, np.asarray(beta, dtype=float).ravel(), float(RF),
                  float(s0), float(ds), int(M), int(Ns), float(x0), float(dx), int(Nx),
                  float(y0), float(dy), int(Ny), float(z0), float(dz), int(Nz),
                  _Image(Ima), Projections)
    return Projections.ravel()


def _StaticCTCone(alpha, zsource, RF, RD, delta0, dDelta, s0, ds, soffset,
                  x0, dx, Nx, y0, dy, Ny, z0, dz, Nz, Ima, MDelta, Ns):
    """Flat (n, ns, m) projections of the static CT views alpha with focus heights zsource."""
    alpha = np.atleast_1d(np.asarray(alpha, dtype=float))
    zsource = np.broadcast_to(np.asarray(zsource, dtype=float), alpha.shape).copy()
    Projections = Allocation.zeros(alpha.size, MDelta * Ns)
    _StaticCTConeViews(alpha, zsource, float(RF), float(RD), float(delta0), float(dDelta),
                       int(MDelta), float(s0), float(ds), float(soffset), int(Ns),
                       float(x0), float(dx), int(Nx), float(y0), float(dy), int(Ny),
                       float(z0), float(dz), int(Nz), _Image(Ima), Projections)
    return Projections.ravel()


def _Subsets(N, NrSubsets=10):
    """MATLAB splits the rotation into NrSubsets equal chunks plus a remainder chunk."""
    K = _fix(N / NrSubsets)
    return K, N - NrSubsets * K


def _MotionView(n, theta, x0, dx, Nx, y0, dy, Ny, xi0, dxi, M, T0, dT, PhantomOld, Phantom):
    """parfor body of the dynamic projectors: phantom at time step n, projected at theta.

    Returns (Projection, px, py, RoI); the old phantom has no tracked point and no RoI.
    """
    from CTBasics.Simulation.CreatePhantom import CreatePhantom

    if PhantomOld:
        Image, _, _, _ = CreatePhantom.CreateVoxelPhantom2DWithMotionRespiratory(
            Nx, dx, x0, Ny, dy, y0, T0, dT, n, "CardiacPhantomWithPlaque"
        )
        px = py = 0
        RoI = None
    else:
        Image, _, px, py, RoI = CreatePhantom.CreateVoxelPhantom2DWithMotionUnified(
            Nx, dx, x0, Ny, dy, y0, T0, dT, n, **Phantom
        )

    Projection = ForwardProjection.RayDrivenForwProjParCartesianSingle(
        x0, dx, Nx, y0, dy, Ny, Image, theta, xi0, dxi, M
    )
    return Projection, px, py, RoI


def _MotionProjections(N, N360, ThetaArray, PhantomOld, Phantom, **broadcast):
    """Runs the parfor over all N views and stacks the projections and tracked points."""
    Views = Parallelization.ParFor(
        _MotionView,
        [{"n": n, "theta": ThetaArray[(n - 1) % N360]} for n in range(1, N + 1)],
        PhantomOld=PhantomOld, Phantom=Phantom, **broadcast
    )

    Projections2D = Allocation.zeros(N, broadcast["M"])
    points = Allocation.zeros(2, N)

    for n, (Projection, px, py, _) in enumerate(Views):
        Projections2D[n, :] = Projection
        points[:, n] = (px, py)

    return Projections2D, points, [View[3] for View in Views]


class ForwardProjection:
    """Ray-driven forward projectors with Josephson (linear) interpolation.

    Index convention follows the MATLAB original: images are flat arrays in row-major
    order where `Image[(ny-1)*Nx + nx - 1]` addresses the 1-based MATLAB element (nx, ny).
    Angle loops run `n = 1..N`, so the first projection angle is `theta0 + dtheta`.
    """

    # ------------------------------------------------------------------ parallel

    @staticmethod
    def RayDrivenForwProjParCartesianSingle(x0, dx, Nx, y0, dy, Ny, Image, theta, xi0, dxi, M):
        return _ProjectParallel(theta, x0, dx, Nx, y0, dy, Ny, Image, xi0, dxi, M)[0]

    @staticmethod
    def RayDrivenForwProjParCartesianMultiple(x0, dx, Nx, y0, dy, Ny, Image, theta0, dtheta, N, xi0, dxi, M):
        theta = theta0 + np.arange(1, N + 1) * dtheta
        return _ProjectParallel(theta, x0, dx, Nx, y0, dy, Ny, Image, xi0, dxi, M).ravel()

    # ----------------------------------------------------------------------- fan

    @staticmethod
    def RayDrivenForwardProj2DFanSingle(RF, alpha, beta, x0, dx, Nx, y0, dy, Ny, Ima, M):
        xi, c, s = _FanRays(RF, alpha, beta, M)
        return _TraceRays(xi, c, s, np.ones(xi.shape, dtype=bool), x0, dx, Nx, y0, dy, Ny, Ima)

    @staticmethod
    def RayDrivenForwardProj2DFanMultiple(RF, alpha0, dalpha, beta, x0, dx, Nx, y0, dy, Ny, Ima, M, N):
        alpha = alpha0 + np.arange(1, N + 1) * dalpha
        xi, c, s = _FanRays(RF, alpha, beta, M)
        return _TraceRays(xi, c, s, np.ones(xi.shape, dtype=bool), x0, dx, Nx, y0, dy, Ny, Ima)

    # ----------------------------------------------------------------- static CT

    @staticmethod
    def RayDrivenForwardProj2DStaticCTSingle(RF, RD, alpha, delta0, dDelta, x0, dx, Nx, y0, dy, Ny, Ima, MDelta):
        xi, c, s, keep = _StaticCTRays(RF, RD, alpha, delta0, dDelta, MDelta)
        return _TraceRays(xi, c, s, keep, x0, dx, Nx, y0, dy, Ny, Ima)

    @staticmethod
    def RayDrivenForwardProj2DStaticCTMultiple(RF, RD, alpha0, dalpha, delta0, dDelta, x0, dx, Nx, y0, dy, Ny, Ima, MDelta, N):
        alpha = alpha0 + np.arange(1, N + 1) * dalpha
        xi, c, s, keep = _StaticCTRays(RF, RD, alpha, delta0, dDelta, MDelta)
        return _TraceRays(xi, c, s, keep, x0, dx, Nx, y0, dy, Ny, Ima)

    # ----------------------------------------------------------- dynamic phantoms

    @staticmethod
    def RayDrivenForwProjParCartesianMultipleWithMotion(x0, dx, Nx, y0, dy, Ny, SelectedPhantom_in,
                                                        theta0, dtheta, N, xi0, dxi, M, N360, TRot):
        """Projects a respiratory phantom that is re-created for every view."""
        from CTBasics.Simulation.CreatePhantom import CreatePhantom

        Projections2D = Allocation.zeros(N, M)

        T0 = 0
        dT = TRot / N360

        for n in range(1, N + 1):
            nT = n
            Image, _, _, _ = CreatePhantom.CreateVoxelPhantom2DWithMotionRespiratory(
                Nx, dx, x0, Ny, dy, y0, T0, dT, nT, SelectedPhantom_in
            )

            # (n-1) so the forward projection actually starts at theta0.
            theta = theta0 + (n - 1) * dtheta

            Projections2D[n - 1, :] = ForwardProjection.RayDrivenForwProjParCartesianSingle(
                x0, dx, Nx, y0, dy, Ny, Image, theta, xi0, dxi, M
            )

        return Projections2D.ravel()

    @staticmethod
    def RayDrivenForwProjParCartesianMultipleWithMotionArbitraryOrder(
            x0, dx, Nx, y0, dy, Ny, N, xi0, dxi, M, N360, TRot, ThetaArray,
            Heartrate=60, CM=False, Stent=False, Plaque=0, PhantomOld=False, Motion=True,
            T0=0, Direction=0, SelectedPhantom="Cardiac", Rotation=0, LineWidth=2.0,
            Center=(0, 0), MotionType="full", TRest=0,
            IodinePeakFactor=3.0, IodineSigmaMin=0.25, IodineSigmaMax=4.5):
        """Projects a dynamic phantom under an arbitrary scan order and re-sorts by angle.

        Returns (Projections2D, Angles_new, points, PhantomTruth, nT_Phases, nTs_new, RoI).
        """
        from CTBasics.Simulation.CreatePhantom import CreatePhantom

        ThetaArray = np.asarray(ThetaArray, dtype=float).ravel()

        Projections2D = Allocation.zeros(N, M)
        PhantomTruth = Allocation.zeros(_fix(2 * N / N360), Nx * Ny)
        nT_Phases = []
        points = Allocation.zeros(2, N)
        nTs = Allocation.zeros(N)
        RoI = [None] * N

        dT = TRot / N360
        phase = 1

        for n in range(1, N + 1):
            if PhantomOld:
                Image, _, _, _ = CreatePhantom.CreateVoxelPhantom2DWithMotionRespiratory(
                    Nx, dx, x0, Ny, dy, y0, T0, dT, n, "CardiacPhantomWithPlaque"
                )
            else:
                Image, _, px, py, RoI[n - 1] = CreatePhantom.CreateVoxelPhantom2DWithMotionUnified(
                    Nx, dx, x0, Ny, dy, y0, T0, dT, n, SelectedPhantom,
                    Heartrate=Heartrate, CM=CM, Stent=Stent, Plaque=Plaque, Motion=Motion,
                    Direction=Direction, Rotation=Rotation, LineWidth=LineWidth, Center=Center,
                    MotionType=MotionType, TRest=TRest
                )
                points[0, n - 1] = px
                points[1, n - 1] = py

            # Reuse the same angle after a full rotation.
            theta = ThetaArray[(n - 1) % N360]

            Projections2D[n - 1, :] = ForwardProjection.RayDrivenForwProjParCartesianSingle(
                x0, dx, Nx, y0, dy, Ny, Image, theta, xi0, dxi, M
            )

            if n % (N360 // 2) == N360 // 4:
                PhantomTruth[phase - 1, :] = Image
                phase += 1
                nT_Phases.append(n)

            nTs[n - 1] = n

        Projections2D, Angles_new, nTs_new = ForwardProjection.RearrangeProjections(
            ThetaArray, Projections2D, nTs
        )

        return (Projections2D.ravel(), Angles_new, points, PhantomTruth,
                np.asarray(nT_Phases, dtype=float), nTs_new, RoI)

    @staticmethod
    def RayDrivenForwProjParCartesianMultipleWithMotionArbitraryOrderParallelized(
            x0, dx, Nx, y0, dy, Ny, N, xi0, dxi, M, N360, TRot, ThetaArray,
            Heartrate=60, CM=False, Stent=False, Plaque=0, PhantomOld=False, Motion=True,
            T0=0, Direction=0, SelectedPhantom="Cardiac", Rotation=0, LineWidth=2.0,
            Center=(0, 0), MotionType="full", TRest=0,
            IodinePeakFactor=3.0, IodineSigmaMin=0.25, IodineSigmaMax=4.5):
        """parfor variant of the arbitrary order projector without the phantom truth.

        Returns (Projections2D, Angles_new, points, RoI).
        """
        ThetaArray = np.asarray(ThetaArray, dtype=float).ravel()

        Phantom = {
            "SelectedPhantom": SelectedPhantom, "Heartrate": Heartrate, "CM": CM,
            "Stent": Stent, "Plaque": Plaque, "Motion": Motion, "Direction": Direction,
            "Rotation": Rotation, "LineWidth": LineWidth, "Center": Center,
            "MotionType": MotionType, "TRest": TRest, "IodinePeakFactor": IodinePeakFactor,
            "IodineSigmaMin": IodineSigmaMin, "IodineSigmaMax": IodineSigmaMax,
        }

        Projections2D, points, RoI = _MotionProjections(
            N, N360, ThetaArray, PhantomOld, Phantom,
            x0=x0, dx=dx, Nx=Nx, y0=y0, dy=dy, Ny=Ny, xi0=xi0, dxi=dxi, M=M,
            T0=T0, dT=TRot / N360
        )

        Projections2D, Angles_new, _ = ForwardProjection.RearrangeProjections(
            ThetaArray, Projections2D, np.arange(1, N + 1)
        )

        return Projections2D.ravel(), Angles_new, points, RoI

    @staticmethod
    def RayDrivenForwProjParCartesianMultipleWithMotionDual(x0, dx, Nx, y0, dy, Ny, N, xi0, dxi, M,
                                                            N360, TRot, ThetaArray, **options):
        """Dual-source entry point; MATLAB dispatches to the ParallelSimple variant."""
        return ForwardProjection.RayDrivenForwProjParCartesianMultipleWithMotionDual_ParallelSimple(
            x0, dx, Nx, y0, dy, Ny, N, xi0, dxi, M, N360, TRot, ThetaArray, **options
        )

    @staticmethod
    def RayDrivenForwProjParCartesianMultipleWithMotionDual_linear(
            x0, dx, Nx, y0, dy, Ny, N, xi0, dxi, M, N360, TRot, ThetaArray,
            Heartrate=60, CM=False, Stent=False, Plaque=0, PhantomOld=False, Motion=True,
            Shift_GT=0, Direction=0, SelectedPhantom="Cardiac", Rotation=0, LineWidth=2.0,
            Center=(0, 0), MotionType="full", TRest=0):
        """Returns (Projections2D, Angles_new, points, PhantomTruth, nT_Phases).

        MATLAB allocates PhantomTruth but never fills it here, so it stays all zeros.
        """
        from CTBasics.Simulation.CreatePhantom import CreatePhantom

        ThetaArray = np.asarray(ThetaArray, dtype=float).ravel()

        Projections2D = Allocation.zeros(N, M)
        PhantomTruth = Allocation.zeros(_fix(N / N360 * 2), Nx * Ny)
        nT_Phases = np.asarray([], dtype=float)
        points = Allocation.zeros(2, N)

        T0 = Shift_GT
        dT = TRot / N360

        for n in range(1, N + 1):
            if PhantomOld:
                Image, _, _, _ = CreatePhantom.CreateVoxelPhantom2DWithMotionRespiratory(
                    Nx, dx, x0, Ny, dy, y0, T0, dT, n, "CardiacPhantomWithPlaque"
                )
            else:
                Image, _, px, py, _ = CreatePhantom.CreateVoxelPhantom2DWithMotionUnified(
                    Nx, dx, x0, Ny, dy, y0, T0, dT, n, SelectedPhantom,
                    Heartrate=Heartrate, CM=CM, Stent=Stent, Plaque=Plaque, Motion=Motion,
                    Direction=Direction, Rotation=Rotation, LineWidth=LineWidth, Center=Center
                )
                points[0, n - 1] = px
                points[1, n - 1] = py

            theta = ThetaArray[(n - 1) % N360]

            Projections2D[n - 1, :] = ForwardProjection.RayDrivenForwProjParCartesianSingle(
                x0, dx, Nx, y0, dy, Ny, Image, theta, xi0, dxi, M
            )

        Projections2D, Angles_new, _ = ForwardProjection.RearrangeProjections(
            ThetaArray, Projections2D, Allocation.zeros(N)
        )

        return Projections2D.ravel(), Angles_new, points, PhantomTruth, nT_Phases

    @staticmethod
    def RayDrivenForwProjParCartesianMultipleWithMotionDual_ParallelSimple(
            x0, dx, Nx, y0, dy, Ny, N, xi0, dxi, M, N360, TRot, ThetaArray,
            Heartrate=60, CM=False, Stent=False, Plaque=0, PhantomOld=False, Motion=True,
            Shift_GT=0, Direction=0, SelectedPhantom="Cardiac", Rotation=0, LineWidth=2.0,
            Center=(0, 0), MotionType="full", TRest=0):
        """Parallelized dual-source variant; MATLAB sizes PhantomTruth (N, M) here."""
        ThetaArray = np.asarray(ThetaArray, dtype=float).ravel()

        PhantomTruth = Allocation.zeros(N, M)
        nT_Phases = np.asarray([], dtype=float)

        Phantom = {
            "SelectedPhantom": SelectedPhantom, "Heartrate": Heartrate, "CM": CM,
            "Stent": Stent, "Plaque": Plaque, "Motion": Motion, "Direction": Direction,
            "Rotation": Rotation, "LineWidth": LineWidth, "Center": Center,
            "MotionType": MotionType, "TRest": TRest,
        }

        Projections2D, points, _ = _MotionProjections(
            N, N360, ThetaArray, PhantomOld, Phantom,
            x0=x0, dx=dx, Nx=Nx, y0=y0, dy=dy, Ny=Ny, xi0=xi0, dxi=dxi, M=M,
            T0=Shift_GT, dT=TRot / N360
        )

        Projections2D, Angles_new, _ = ForwardProjection.RearrangeProjections(
            ThetaArray, Projections2D, Allocation.zeros(N)
        )

        return Projections2D.ravel(), Angles_new, points, PhantomTruth, nT_Phases

    @staticmethod
    def RearrangeProjections(Angles, Projections, nTs):
        """Sorts each full rotation of an arbitrary scan order into ascending angle.

        The sort permutation of one rotation is reused for every full rotation; a trailing
        incomplete rotation is sorted with the permutation of its own leading angles.
        """
        Angles = np.asarray(Angles, dtype=float).ravel()
        Projections = np.asarray(Projections, dtype=float)
        nTs = np.asarray(nTs, dtype=float).ravel()

        r, c = Projections.shape
        N2pi = len(Angles)

        NFullRot = r // N2pi
        Nrest = r % N2pi

        idx_full = np.argsort(Angles, kind="stable")
        Angles_sorted = Angles[idx_full]

        Projections_rearranged = np.zeros((r, c))
        nTs_new = np.zeros(r)

        for i in range(1, NFullRot + 1):
            start = (i - 1) * N2pi
            block = slice(start, start + N2pi)
            Projections_rearranged[block, :] = Projections[start + idx_full, :]
            nTs_new[block] = nTs[start + idx_full]

        if Nrest > 0:
            start = NFullRot * N2pi
            idx_rest = np.argsort(Angles[:Nrest], kind="stable")
            Projections_rearranged[start:r, :] = Projections[start + idx_rest, :]
            nTs_new[start:r] = nTs[start + idx_rest]

        return Projections_rearranged, Angles_sorted, nTs_new

    @staticmethod
    def SpectralRayDrivenForwProjParCartesianMultiple(x0, dx, Nx, y0, dy, Ny, MaterialImage,
                                                      theta0, dtheta, N, xi0, dxi, M,
                                                      spectrum, E0, N_Ebins, deltaE):
        """Polychromatic parallel-beam projection of a material-labelled phantom."""
        aMaterialProperties, I0Values, SumOverI0 = _SpectralBins(spectrum, E0, N_Ebins, deltaE)

        Images = [
            {"Image": aMaterialProperties.MaterialToSpectralAttenuationCoefficientConversion(
                MaterialImage, Nx, Ny, _fix(E0 + nbin * deltaE))}
            for nbin in range(1, N_Ebins + 1)
        ]
        SpectralProjections = np.asarray(Parallelization.ParFor(
            ForwardProjection.RayDrivenForwProjParCartesianMultiple, Images,
            x0=x0, dx=dx, Nx=Nx, y0=y0, dy=dy, Ny=Ny,
            theta0=theta0, dtheta=dtheta, N=N, xi0=xi0, dxi=dxi, M=M
        ))

        return _SpectralAttenuation(SpectralProjections, I0Values, SumOverI0)

    @staticmethod
    def SpectralRayDrivenForwardProj2DFanMultiple(RF, alpha0, dalpha, beta, x0, dx, Nx, y0, dy, Ny,
                                                  MaterialImage, M, N,
                                                  spectrum, E0, N_Ebins, deltaE):
        """Polychromatic fan-beam projection of a material-labelled phantom."""
        aMaterialProperties, I0Values, SumOverI0 = _SpectralBins(spectrum, E0, N_Ebins, deltaE)

        Images = [
            {"Ima": aMaterialProperties.MaterialToSpectralAttenuationCoefficientConversion(
                MaterialImage, Nx, Ny, _fix(E0 + nbin * deltaE))}
            for nbin in range(1, N_Ebins + 1)
        ]
        SpectralProjections = np.asarray(Parallelization.ParFor(
            ForwardProjection.RayDrivenForwardProj2DFanMultiple, Images,
            RF=RF, alpha0=alpha0, dalpha=dalpha, beta=beta,
            x0=x0, dx=dx, Nx=Nx, y0=y0, dy=dy, Ny=Ny, M=M, N=N
        ))

        return _SpectralAttenuation(SpectralProjections, I0Values, SumOverI0)

    # ------------------------------------------------------------------- 3D fan

    @staticmethod
    def RayDrivenForwardProj3DFanSingle(RF, alpha, beta, s0, ds, x0, dx, Nx, y0, dy, Ny,
                                        z0, dz, Nz, Ima, M, Ns, zsource):
        """One cone-beam fan projection of (M fan channels x Ns detector rows)."""
        return _FanCone(alpha, zsource, RF, beta, s0, ds, x0, dx, Nx, y0, dy, Ny,
                        z0, dz, Nz, Ima, M, Ns)

    @staticmethod
    def RayDrivenForwardProj3DFanMultipleSequence(RF, alpha0, dalpha, beta, s0, ds,
                                                  x0, dx, Nx, y0, dy, Ny, z0, dz, Nz,
                                                  Ima, M, Ns, N, zn):
        """All N readings of one sequence scan at the fixed table position zn."""
        return _FanCone(alpha0 + np.arange(1, N + 1) * dalpha, zn, RF, beta, s0, ds,
                        x0, dx, Nx, y0, dy, Ny, z0, dz, Nz, Ima, M, Ns)

    @staticmethod
    def RayDrivenForwardProj3DFanSequence(Ima_in, FanBeamScanDescriptor_in,
                                          SimulationVolumeGeometry_in):
        Geometry = FanBeamScanDescriptor_in.Geometry
        Parameter = FanBeamScanDescriptor_in.Parameter

        RF = Geometry.RF
        beta = Geometry.beta
        s0 = Geometry.s0
        ds = Geometry.ds
        M = Geometry.MFan
        Ns = Geometry.Ns
        dalpha = Geometry.Sampling.dalpha

        alpha0 = np.atleast_1d(np.asarray(Parameter.alpha0, dtype=float))
        N = Parameter.N
        z0focus = Parameter.z0focus
        dzfocus = Parameter.dzfocus
        NzScans = Parameter.NzScans

        G = SimulationVolumeGeometry_in

        Projections = Allocation.zeros(NzScans * M * N * Ns)
        block = M * N * Ns

        for nzfocus in range(1, NzScans + 1):
            zn = z0focus + (nzfocus - 1) * dzfocus
            Projections[(nzfocus - 1) * block:nzfocus * block] = (
                ForwardProjection.RayDrivenForwardProj3DFanMultipleSequence(
                    RF, alpha0[nzfocus - 1], dalpha, beta, s0, ds,
                    G.x0sim, G.dxsim, G.Nxsim, G.y0sim, G.dysim, G.Nysim,
                    G.z0sim, G.dzsim, G.Nzsim, Ima_in, M, Ns, N, zn
                )
            )

        return Projections

    @staticmethod
    def RayDrivenForwardProj3DFanSpiral(Ima_in, FanBeamScanDescriptor_in,
                                        SimulationVolumeGeometry_in):
        Geometry = FanBeamScanDescriptor_in.Geometry
        Parameter = FanBeamScanDescriptor_in.Parameter

        RF = Geometry.RF
        beta = Geometry.beta
        s0 = Geometry.s0
        ds = Geometry.ds
        M = Geometry.MFan
        Ns = Geometry.Ns
        dalpha = Geometry.Sampling.dalpha

        alpha0 = Parameter.alpha0
        N = Parameter.N
        zfocus0 = Parameter.z0focus
        pitch = Parameter.pitch

        G = SimulationVolumeGeometry_in

        alpha = alpha0 + np.arange(1, N + 1) * dalpha
        # The focus advances by pitch * collimation per full rotation.
        zsource = zfocus0 + (alpha - alpha0) * pitch * Ns * ds / (2 * np.pi)

        return _FanCone(alpha, zsource, RF, beta, s0, ds,
                        G.x0sim, G.dxsim, G.Nxsim, G.y0sim, G.dysim, G.Nysim,
                        G.z0sim, G.dzsim, G.Nzsim, Ima_in, M, Ns)

    # ------------------------------------------------------------- 3D static CT

    @staticmethod
    def RayDrivenForwardProj3DStaticCTSingle(RF, RD, alpha, delta0, dDelta, s0, ds, soffset,
                                             x0, dx, Nx, y0, dy, Ny, z0, dz, Nz, Ima,
                                             MDelta, Ns, zsource):
        """One static CT cone-beam projection; detector elements sit on the ring at RD."""
        return _StaticCTCone(alpha, zsource, RF, RD, delta0, dDelta, s0, ds, soffset,
                             x0, dx, Nx, y0, dy, Ny, z0, dz, Nz, Ima, MDelta, Ns)

    @staticmethod
    def RayDrivenForwardProj3DStaticCTMultipleSequence(RF, RD, alpha0, dalpha, delta0, dDelta,
                                                       s0, ds, soffset, x0, dx, Nx, y0, dy, Ny,
                                                       z0, dz, Nz, Ima, MDelta, Ns, N, zn):
        return _StaticCTCone(alpha0 + np.arange(1, N + 1) * dalpha, zn, RF, RD, delta0, dDelta,
                             s0, ds, soffset, x0, dx, Nx, y0, dy, Ny, z0, dz, Nz,
                             Ima, MDelta, Ns)

    @staticmethod
    def _StaticCTProjectionParameters(StaticCTScanDescriptor_in, SimulationVolumeGeometry_in):
        """Expands the MATLAB `nargin==3` branch of the static CT 3D projectors."""
        Geometry = StaticCTScanDescriptor_in.Geometry
        Parameter = StaticCTScanDescriptor_in.Parameter
        G = SimulationVolumeGeometry_in

        return {
            "RF": Geometry.RF,
            "RD": Geometry.RD,
            "delta0": Geometry.delta0,
            "dDelta": Geometry.dDelta,
            "s0": Geometry.s0,
            "ds": Geometry.ds,
            "soffset": Geometry.soffset,
            "MDelta": Geometry.MDelta,
            "Ns": Geometry.Ns,
            "dalpha": Geometry.Sampling.dalpha,
            "alpha0": Parameter.alpha0,
            "N": Parameter.N,
            "z0focus": Parameter.z0focus,
            "x0": G.x0sim, "dx": G.dxsim, "Nx": G.Nxsim,
            "y0": G.y0sim, "dy": G.dysim, "Ny": G.Nysim,
            "z0": G.z0sim, "dz": G.dzsim, "Nz": G.Nzsim,
        }

    @staticmethod
    def RayDrivenForwardProj3DStaticCTSequence(Ima_in, StaticCTScanDescriptor_in,
                                               SimulationVolumeGeometry_in,
                                               RF=None, RD=None, alpha0=None, dalpha=None,
                                               delta0=None, dDelta=None,
                                               s0=None, ds=None, soffset=None,
                                               x0=None, dx=None, Nx=None,
                                               y0=None, dy=None, Ny=None,
                                               z0=None, dz=None, Nz=None,
                                               MDelta=None, Ns=None, N=None,
                                               z0focus=None, dzfocus=None, NzScans=None):
        if RF is None:
            P = ForwardProjection._StaticCTProjectionParameters(
                StaticCTScanDescriptor_in, SimulationVolumeGeometry_in
            )
            RF, RD = P["RF"], P["RD"]
            alpha0, dalpha = P["alpha0"], P["dalpha"]
            delta0, dDelta = P["delta0"], P["dDelta"]
            s0, ds, soffset = P["s0"], P["ds"], P["soffset"]
            x0, dx, Nx = P["x0"], P["dx"], P["Nx"]
            y0, dy, Ny = P["y0"], P["dy"], P["Ny"]
            z0, dz, Nz = P["z0"], P["dz"], P["Nz"]
            MDelta, Ns, N = P["MDelta"], P["Ns"], P["N"]
            z0focus = P["z0focus"]
            dzfocus = StaticCTScanDescriptor_in.Parameter.dzfocus
            NzScans = StaticCTScanDescriptor_in.Parameter.NzScans

        alpha0 = np.atleast_1d(np.asarray(alpha0, dtype=float))

        Projections = Allocation.zeros(NzScans * MDelta * N * Ns)
        block = MDelta * N * Ns

        for nzfocus in range(1, NzScans + 1):
            zn = z0focus + (nzfocus - 1) * dzfocus
            Projections[(nzfocus - 1) * block:nzfocus * block] = (
                ForwardProjection.RayDrivenForwardProj3DStaticCTMultipleSequence(
                    RF, RD, alpha0[min(nzfocus, len(alpha0)) - 1], dalpha, delta0, dDelta,
                    s0, ds, soffset, x0, dx, Nx, y0, dy, Ny, z0, dz, Nz,
                    Ima_in, MDelta, Ns, N, zn
                )
            )

        return Projections

    @staticmethod
    def RayDrivenForwardProj3DStaticCTSpiral(Ima_in, StaticCTScanDescriptor_in,
                                             SimulationVolumeGeometry_in,
                                             RF=None, RD=None, alpha0=None, dalpha=None,
                                             delta0=None, dDelta=None,
                                             s0=None, ds=None, soffset=None,
                                             x0=None, dx=None, Nx=None,
                                             y0=None, dy=None, Ny=None,
                                             z0=None, dz=None, Nz=None,
                                             MDelta=None, Ns=None, N=None,
                                             z0focus=None, pitch=None):
        if RF is None:
            P = ForwardProjection._StaticCTProjectionParameters(
                StaticCTScanDescriptor_in, SimulationVolumeGeometry_in
            )
            RF, RD = P["RF"], P["RD"]
            alpha0, dalpha = P["alpha0"], P["dalpha"]
            delta0, dDelta = P["delta0"], P["dDelta"]
            s0, ds, soffset = P["s0"], P["ds"], P["soffset"]
            x0, dx, Nx = P["x0"], P["dx"], P["Nx"]
            y0, dy, Ny = P["y0"], P["dy"], P["Ny"]
            z0, dz, Nz = P["z0"], P["dz"], P["Nz"]
            MDelta, Ns, N = P["MDelta"], P["Ns"], P["N"]
            z0focus = P["z0focus"]
            pitch = StaticCTScanDescriptor_in.Parameter.pitch

        RFD = RF + RD
        alpha = alpha0 + np.arange(1, N + 1) * dalpha
        # The static CT source ring advances with the magnified collimation RF/RFD.
        zsource = z0focus + (alpha - alpha0) * pitch * Ns * ds * RF / RFD / (2 * np.pi)

        return _StaticCTCone(alpha, zsource, RF, RD, delta0, dDelta, s0, ds, soffset,
                             x0, dx, Nx, y0, dy, Ny, z0, dz, Nz, Ima_in, MDelta, Ns)

    # ------------------------------------------------- static CT z-collimation

    @staticmethod
    def _ShiftedFocalSpot(Geometry, alpha_in, dFVec):
        """Apply the focal spot offset dFVec = [dzF, drF, dxyF] to the source position."""
        dzF, drF, dxyF = float(dFVec[0]), float(dFVec[1]), float(dFVec[2])

        RFbar = Geometry.RF + drF
        dalphaF = np.arctan(dxyF / RFbar)
        RF = np.sqrt(RFbar**2 + dxyF**2)
        alpha = alpha_in + dalphaF

        return RF, alpha, dzF

    @staticmethod
    def _RayPassesCollimator(RF, RD, RC, alpha, delta, d, dzF, zCollUpper, zCollLower, thicknessInZLowerCollimator):
        """True if the ray enters the collimator opening and clears its lower lip."""
        x1, x2, y1, y2 = ForwardProjection.CalculateIntersectionPointsCircleAndLineIn2D_StaticCTGeo(
            RF, RD, RC, alpha, delta
        )
        if not (np.isfinite(x1) and np.isfinite(x2)):
            # The ray misses the collimator ring entirely, so it is blocked.
            return False

        l1 = np.sqrt((x1 - RF * np.sin(alpha)) ** 2 + (y1 + RF * np.cos(alpha)) ** 2)
        l2 = np.sqrt((x2 - RF * np.sin(alpha)) ** 2 + (y2 + RF * np.cos(alpha)) ** 2)

        # The near intersection is the collimator edge on the source side.
        lclose, ldistant = (l1, l2) if l1 < l2 else (l2, l1)

        L = np.sqrt(RD * RD + RF * RF + 2 * RD * RF * np.sin(delta - alpha))

        # Intercept theorem along the ray: z grows linearly from dzF at the source to d at the detector.
        zNear = d / L * lclose + dzF * (1 - lclose / L)
        if not (zCollLower < zNear < zCollUpper):
            return False

        # Step 2: even inside the opening the ray can still hit the lower lip on the detector side.
        zFar = d / L * ldistant + dzF * (1 - ldistant / L)
        return bool(zFar < zCollLower - thicknessInZLowerCollimator)

    @staticmethod
    def StaticCTCollimationSingle(StaticCTScanDescriptor_in, alpha_in, dFVec):
        Geometry = StaticCTScanDescriptor_in.Geometry
        RF, alpha, dzF = ForwardProjection._ShiftedFocalSpot(Geometry, alpha_in, dFVec)

        RC = Geometry.RC
        RD = Geometry.RD
        zCollUpper = Geometry.zCollimatorUpper
        zCollLower = Geometry.zCollimatorLower
        thicknessInZLowerCollimator = Geometry.thicknessInZLowerCollimator
        delta0 = Geometry.delta0
        dDelta = Geometry.dDelta
        ds = Geometry.ds
        soffset = Geometry.soffset
        MDelta = Geometry.MDelta
        Ns = Geometry.Ns

        Projections = Allocation.zeros(Ns * MDelta)

        for mdelta in range(1, MDelta + 1):
            delta = delta0 + mdelta * dDelta
            for ns in range(1, Ns + 1):
                d = soffset + ns * ds - Ns * ds - ds / 2.0
                if ForwardProjection._RayPassesCollimator(
                    RF, RD, RC, alpha, delta, d, dzF,
                    zCollUpper, zCollLower, thicknessInZLowerCollimator,
                ):
                    Projections[(ns - 1) * MDelta + mdelta - 1] = 1

        return Projections

    @staticmethod
    def StaticCTCollimationSingleSourceSampled(StaticCTScanDescriptor_in, alpha_in, dFVecArray_in):
        Geometry = StaticCTScanDescriptor_in.Geometry
        MDelta = Geometry.MDelta
        Ns = Geometry.Ns

        dFVecArray_in = np.atleast_2d(np.asarray(dFVecArray_in, dtype=float))

        ArrayOfCollimationOfOneProjection = np.asarray(Parallelization.ParFor(
            ForwardProjection.StaticCTCollimationSingle,
            [{"dFVec": dFVec} for dFVec in dFVecArray_in],
            StaticCTScanDescriptor_in=StaticCTScanDescriptor_in, alpha_in=alpha_in
        )).reshape(dFVecArray_in.shape[0], Ns * MDelta)

        # MATLAB accumulates without dividing by the sample count; kept verbatim.
        AveragedCollimationOfOneProjection = ArrayOfCollimationOfOneProjection.sum(axis=0)

        return AveragedCollimationOfOneProjection, ArrayOfCollimationOfOneProjection

    @staticmethod
    def StaticCTCollimationLeafSingle(StaticCTScanDescriptor_in, alpha_in, dFVec, LeafConfig):
        Geometry = StaticCTScanDescriptor_in.Geometry
        RF, alpha, dzF = ForwardProjection._ShiftedFocalSpot(Geometry, alpha_in, dFVec)

        RC = Geometry.RC
        RD = Geometry.RD
        thicknessInZLowerCollimator = Geometry.thicknessInZLowerCollimator
        delta0 = Geometry.delta0
        dDelta = Geometry.dDelta
        ds = Geometry.ds
        soffset = Geometry.soffset
        MDelta = Geometry.MDelta
        Ns = Geometry.Ns

        ForwardProjection.ValidateLeafCollimatorConfig(LeafConfig, MDelta)

        Projections = Allocation.zeros(Ns * MDelta)

        for mdelta in range(1, MDelta + 1):
            delta = delta0 + mdelta * dDelta
            leafIdx = ForwardProjection.GetLeafIndexFromDetectorElement(mdelta, MDelta, LeafConfig)
            zCollUpper, zCollLower = ForwardProjection.GetLeafCollimatorBounds(LeafConfig, leafIdx)

            for ns in range(1, Ns + 1):
                d = soffset + ns * ds - Ns * ds - ds / 2.0
                if ForwardProjection._RayPassesCollimator(
                    RF, RD, RC, alpha, delta, d, dzF,
                    zCollUpper, zCollLower, thicknessInZLowerCollimator,
                ):
                    Projections[(ns - 1) * MDelta + mdelta - 1] = 1

        return Projections

    @staticmethod
    def StaticCTCollimationLeafSingleSourceSampled(StaticCTScanDescriptor_in, alpha_in, dFVecArray_in, LeafConfig):
        Geometry = StaticCTScanDescriptor_in.Geometry
        MDelta = Geometry.MDelta
        Ns = Geometry.Ns

        ForwardProjection.ValidateLeafCollimatorConfig(LeafConfig, MDelta)

        dFVecArray_in = np.atleast_2d(np.asarray(dFVecArray_in, dtype=float))

        ArrayOfCollimationOfOneProjection = np.asarray(Parallelization.ParFor(
            ForwardProjection.StaticCTCollimationLeafSingle,
            [{"dFVec": dFVec} for dFVec in dFVecArray_in],
            StaticCTScanDescriptor_in=StaticCTScanDescriptor_in, alpha_in=alpha_in,
            LeafConfig=LeafConfig
        )).reshape(dFVecArray_in.shape[0], Ns * MDelta)

        AveragedCollimationOfOneProjection = ArrayOfCollimationOfOneProjection.sum(axis=0)

        return AveragedCollimationOfOneProjection, ArrayOfCollimationOfOneProjection

    # ---------------------------------------------------------------- helpers

    @staticmethod
    def CalculateIntersectionPointsCircleAndLineIn2D_StaticCTGeo(RF, RD, RC, alpha, delta):
        """Intersect the source-to-detector line with the collimator circle of radius RC.

        Returns NaN coordinates when the line misses the circle. MATLAB yields complex
        values in that case; NaN keeps the result real and lets callers reject the ray.
        """
        m = (RD * np.sin(delta) + RF * np.cos(alpha)) / (RD * np.cos(delta) - RF * np.sin(alpha))
        t = -RF * np.cos(alpha) - m * RF * np.sin(alpha)

        discriminant = 4 * m * m * t * t - 4 * (1 + m * m) * (t * t - RC * RC)
        if discriminant < 0:
            return np.nan, np.nan, np.nan, np.nan

        root = np.sqrt(discriminant)
        x1 = (-2 * m * t + root) / (2 * (1 + m * m))
        x2 = (-2 * m * t - root) / (2 * (1 + m * m))
        return x1, x2, m * x1 + t, m * x2 + t

    @staticmethod
    def ValidateLeafCollimatorConfig(LeafConfig, MDelta):
        for field in ("N_collimatorLeafs", "zUpperPerLeaf", "zLowerPerLeaf"):
            if field not in LeafConfig:
                raise ValueError(f"LeafConfig missing required field: {field}")

        NLeafs = LeafConfig["N_collimatorLeafs"]
        if not (np.isscalar(NLeafs) and np.isfinite(NLeafs) and NLeafs >= 1 and _fix(NLeafs) == NLeafs):
            raise ValueError("LeafConfig.N_collimatorLeafs must be a positive integer scalar.")

        zUpper = np.asarray(LeafConfig["zUpperPerLeaf"], dtype=float).ravel()
        zLower = np.asarray(LeafConfig["zLowerPerLeaf"], dtype=float).ravel()

        if zUpper.size != NLeafs or zLower.size != NLeafs:
            raise ValueError("LeafConfig.zUpperPerLeaf and LeafConfig.zLowerPerLeaf must have length N_collimatorLeafs.")

        if not (np.isfinite(zUpper).all() and np.isfinite(zLower).all()):
            raise ValueError("LeafConfig z-vectors must only contain finite values.")

        if "leafIndexPerDetectorElement" in LeafConfig:
            idx = np.asarray(LeafConfig["leafIndexPerDetectorElement"], dtype=float).ravel()
            if idx.size != MDelta:
                raise ValueError("LeafConfig.leafIndexPerDetectorElement must have length MDelta.")
            if not np.isfinite(idx).all() or not (idx == np.trunc(idx)).all() or idx.min() < 1 or idx.max() > NLeafs:
                raise ValueError(
                    "LeafConfig.leafIndexPerDetectorElement must contain integer values in [1, N_collimatorLeafs]."
                )

    @staticmethod
    def GetLeafIndexFromDetectorElement(mdelta, MDelta, LeafConfig):
        if "leafIndexPerDetectorElement" in LeafConfig:
            return int(LeafConfig["leafIndexPerDetectorElement"][mdelta - 1])

        NLeafs = int(LeafConfig["N_collimatorLeafs"])
        # Default mapping: uniform segmentation along the detector elements (delta direction).
        leafIdx = int(np.ceil(mdelta * NLeafs / MDelta))
        return max(1, min(NLeafs, leafIdx))

    @staticmethod
    def GetLeafCollimatorBounds(LeafConfig, leafIdx):
        return (
            float(LeafConfig["zUpperPerLeaf"][leafIdx - 1]),
            float(LeafConfig["zLowerPerLeaf"][leafIdx - 1]),
        )
