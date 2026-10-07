import numpy as np

from ..Memory.Allocation import Allocation


def _fix(value):
    """MATLAB fix(): truncate toward zero."""
    return int(value)


def _fix_array(values):
    """Vectorized MATLAB fix() returning integer indices."""
    return np.trunc(values).astype(np.int64)


def _GatherInterpolated(Image, index_low, index_high, w, valid):
    """Linear interpolation between two flat indices, contributing 0 where invalid."""
    safe_low = np.where(valid, index_low, 0)
    safe_high = np.where(valid, index_high, 0)
    contribution = (1 - w) * Image[safe_low] + w * Image[safe_high]
    return np.where(valid, contribution, 0.0)




def _SpectralBins(spectrum, E0, N_Ebins, deltaE):
    """Material properties plus the sampled tube spectrum of the requested energy bins."""
    from ..Simulation.MaterialProperties import MaterialProperties
    from ..Simulation.SpectralProperties import _SpectralBins as _Bins

    I0Values, SumOverI0 = _Bins(spectrum, E0, N_Ebins, deltaE)
    return MaterialProperties(), I0Values, SumOverI0


def _SpectralAttenuation(SpectralProjections, I0Values, SumOverI0):
    """Spectrally weighted intensity average turned back into a line integral."""
    Intensity = (I0Values[:, None] * np.exp(-SpectralProjections)).sum(axis=0) / SumOverI0
    return -np.log(Intensity)


def _Gather3D(Ima, base, step_inplane, wplane, wz, NxNy, valid):
    """Trilinear sample: linear in the traversed in-plane axis and linear in z."""
    safe = np.where(valid, base, 0)

    pnz = (1 - wplane) * Ima[safe] + wplane * Ima[safe + step_inplane]
    pnz_plus1 = (1 - wplane) * Ima[safe + NxNy] + wplane * Ima[safe + NxNy + step_inplane]

    return np.where(valid, (1 - wz) * pnz + wz * pnz_plus1, 0.0)


def _TraceRay3D(xi, B, c, s, A, cos_gamma, zsource,
                x0, dx, Nx, y0, dy, Ny, z0, dz, Nz, Ima):
    """Integrate one cone-beam ray for every detector row at once.

    `A` and `cos_gamma` carry the row dependence and have shape (Ns,); everything else is
    fixed by the in-plane ray. The z position of the sample follows the MATLAB expression
    nz_real = zsource/dz + (B - x*s + y*s)*A/dz - z0/dz, which is kept verbatim.
    """
    NxNy = Nx * Ny

    A = np.atleast_1d(np.asarray(A, dtype=float))
    cos_gamma = np.atleast_1d(np.asarray(cos_gamma, dtype=float))

    if abs(c) < abs(s):
        nx = np.arange(1, Nx + 1)
        x = x0 + nx * dx

        plane_real = (xi - x * c - y0 * s) / (dy * s)
        nplane = _fix_array(plane_real)
        wplane = plane_real - nplane
        plane_valid = (nplane >= 1) & (nplane < Ny)

        y = y0 + plane_real * dy
        step_inplane = Nx
        flat = (nplane - 1) * Nx + nx - 1
        L = np.abs(dx / (s * cos_gamma))
    else:
        ny = np.arange(1, Ny + 1)
        y = y0 + ny * dy

        plane_real = (xi - y * s - x0 * c) / (dx * c)
        nplane = _fix_array(plane_real)
        wplane = plane_real - nplane
        plane_valid = (nplane >= 1) & (nplane < Nx)

        x = x0 + plane_real * dx
        step_inplane = 1
        flat = (ny - 1) * Nx + nplane - 1
        L = np.abs(dy / (c * cos_gamma))

    nz_real = zsource / dz + ((B - x * s + y * s)[None, :] * A[:, None]) / dz - z0 / dz
    nz = _fix_array(nz_real)
    wz = nz_real - nz

    valid = plane_valid[None, :] & (nz >= 1) & (nz < Nz)
    base = (nz - 1) * NxNy + flat[None, :]

    samples = _Gather3D(Ima, base, step_inplane, wplane[None, :], wz, NxNy, valid)
    return samples.sum(axis=1) * L


def _Subsets(N, NrSubsets=10):
    """MATLAB splits the rotation into NrSubsets equal chunks plus a remainder chunk."""
    K = _fix(N / NrSubsets)
    return K, N - NrSubsets * K


class ForwardProjection:
    """Ray-driven forward projectors with Josephson (linear) interpolation.

    Index convention follows the MATLAB original: images are flat arrays in row-major
    order where `Image[(ny-1)*Nx + nx - 1]` addresses the 1-based MATLAB element (nx, ny).
    Angle loops run `n = 1..N`, so the first projection angle is `theta0 + dtheta`.
    """

    # ------------------------------------------------------------------ parallel

    @staticmethod
    def RayDrivenForwProjParCartesianSingle(x0, dx, Nx, y0, dy, Ny, Image, theta, xi0, dxi, M):
        Image = np.asarray(Image, dtype=float)

        c = np.cos(theta)
        s = np.sin(theta)

        m = np.arange(1, M + 1)
        xi = xi0 + m * dxi

        # xi = x*c + y*s. Traverse along the axis the ray crosses most steeply so that
        # exactly one interpolated sample is taken per row resp. column.
        if abs(c) < abs(s):
            L = abs(dx / s)
            nx = np.arange(1, Nx + 1)
            ny_real = ((xi[:, None] - (x0 + nx[None, :] * dx) * c) / s - y0) / dy
            ny = _fix_array(ny_real)
            w = ny_real - ny
            valid = (ny >= 1) & (ny < Ny)
            samples = _GatherInterpolated(
                Image, (ny - 1) * Nx + nx[None, :] - 1, ny * Nx + nx[None, :] - 1, w, valid
            )
        else:
            L = abs(dy / c)
            ny = np.arange(1, Ny + 1)
            nx_real = ((xi[:, None] - (y0 + ny[None, :] * dy) * s) / c - x0) / dx
            nx = _fix_array(nx_real)
            w = nx_real - nx
            valid = (nx >= 1) & (nx < Nx)
            samples = _GatherInterpolated(
                Image, (ny[None, :] - 1) * Nx + nx - 1, (ny[None, :] - 1) * Nx + nx, w, valid
            )

        Projection2D = Allocation.zeros(M)
        Projection2D[:] = samples.sum(axis=1) * L
        return Projection2D

    @staticmethod
    def RayDrivenForwProjParCartesianMultiple(x0, dx, Nx, y0, dy, Ny, Image, theta0, dtheta, N, xi0, dxi, M):
        Projections2D = Allocation.zeros(N * M)
        for n in range(1, N + 1):
            theta = theta0 + n * dtheta
            Projections2D[(n - 1) * M:n * M] = ForwardProjection.RayDrivenForwProjParCartesianSingle(
                x0, dx, Nx, y0, dy, Ny, Image, theta, xi0, dxi, M
            )
        return Projections2D

    # ----------------------------------------------------------------------- fan

    @staticmethod
    def _TraceParallelRay(xi, c, s, x0, dx, Nx, y0, dy, Ny, Ima):
        """Accumulate one ray once xi and the direction (c, s) are known.

        Shared by the fan and static CT projectors, which both reduce their ray to the
        parallel parametrisation xi = x*c + y*s before integrating.
        """
        if abs(c) < abs(s):
            L = abs(dx / s)
            nx = np.arange(1, Nx + 1)
            ny_real = (xi - x0 * c - nx * dx * c - y0 * s) / (dy * s)
            ny = _fix_array(ny_real)
            w = ny_real - ny
            valid = (ny >= 1) & (ny < Ny)
            samples = _GatherInterpolated(Ima, (ny - 1) * Nx + nx - 1, ny * Nx + nx - 1, w, valid)
            return samples.sum() * L

        L = abs(dy / c)
        ny = np.arange(1, Ny + 1)
        nx_real = (xi - y0 * s - ny * dy * s - x0 * c) / (dx * c)
        nx = _fix_array(nx_real)
        w = nx_real - nx
        valid = (nx >= 1) & (nx < Nx)
        samples = _GatherInterpolated(Ima, (ny - 1) * Nx + nx - 1, (ny - 1) * Nx + nx, w, valid)
        return samples.sum() * L

    @staticmethod
    def RayDrivenForwardProj2DFanSingle(RF, alpha, beta, x0, dx, Nx, y0, dy, Ny, Ima, M):
        Projection = Allocation.zeros(M)
        Ima = np.asarray(Ima, dtype=float)
        beta = np.atleast_1d(np.asarray(beta, dtype=float))

        sx = RF * np.sin(alpha)
        sy = -RF * np.cos(alpha)

        for m in range(1, M + 1):
            s = np.sin(alpha + beta[m - 1])
            c = np.cos(alpha + beta[m - 1])
            # Orthogonal projection of the source vector onto the xi axis; equals -RF*sin(beta).
            xi = sx * c + sy * s
            Projection[m - 1] = ForwardProjection._TraceParallelRay(xi, c, s, x0, dx, Nx, y0, dy, Ny, Ima)

        return Projection

    @staticmethod
    def RayDrivenForwardProj2DFanMultiple(RF, alpha0, dalpha, beta, x0, dx, Nx, y0, dy, Ny, Ima, M, N):
        Projection2D = Allocation.zeros(N * M)
        for n in range(1, N + 1):
            alpha = alpha0 + n * dalpha
            Projection2D[(n - 1) * M:n * M] = ForwardProjection.RayDrivenForwardProj2DFanSingle(
                RF, alpha, beta, x0, dx, Nx, y0, dy, Ny, Ima, M
            )
        return Projection2D

    # ----------------------------------------------------------------- static CT

    @staticmethod
    def RayDrivenForwardProj2DStaticCTSingle(RF, RD, alpha, delta0, dDelta, x0, dx, Nx, y0, dy, Ny, Ima, MDelta):
        Projection = Allocation.zeros(MDelta)
        Ima = np.asarray(Ima, dtype=float)

        sx = RF * np.sin(alpha)
        sy = -RF * np.cos(alpha)

        for m in range(1, MDelta + 1):
            # For static CT the detector raster is the betaDash raster.
            delta = delta0 + m * dDelta
            betaDash = delta - alpha - np.pi / 2.0
            det_x = np.cos(delta)
            det_y = np.sin(delta)
            beta = np.arctan((np.sin(betaDash) * RD) / (np.cos(betaDash) * RD + RF))
            s = np.sin(alpha + beta)
            c = np.cos(alpha + beta)
            eta_x = -s
            eta_y = c

            # Limits the fan to rays within pi/2 of the detector element normal.
            if abs(np.arccos(np.clip(eta_x * det_x + eta_y * det_y, -1.0, 1.0))) >= np.pi / 2.0:
                continue

            xi = sx * c + sy * s
            Projection[m - 1] = ForwardProjection._TraceParallelRay(xi, c, s, x0, dx, Nx, y0, dy, Ny, Ima)

        return Projection

    @staticmethod
    def RayDrivenForwardProj2DStaticCTMultiple(RF, RD, alpha0, dalpha, delta0, dDelta, x0, dx, Nx, y0, dy, Ny, Ima, MDelta, N):
        Projection2D = Allocation.zeros(N * MDelta)
        for n in range(1, N + 1):
            alpha = alpha0 + n * dalpha
            Projection2D[(n - 1) * MDelta:n * MDelta] = ForwardProjection.RayDrivenForwardProj2DStaticCTSingle(
                RF, RD, alpha, delta0, dDelta, x0, dx, Nx, y0, dy, Ny, Ima, MDelta
            )
        return Projection2D

    # ----------------------------------------------------------- dynamic phantoms

    @staticmethod
    def RayDrivenForwProjParCartesianMultipleWithMotion(x0, dx, Nx, y0, dy, Ny, SelectedPhantom_in,
                                                        theta0, dtheta, N, xi0, dxi, M, N360, TRot):
        """Projects a respiratory phantom that is re-created for every view."""
        from ..Simulation.CreatePhantom import CreatePhantom

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
        from ..Simulation.CreatePhantom import CreatePhantom

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
            x0, dx, Nx, y0, dy, Ny, N, xi0, dxi, M, N360, TRot, ThetaArray, **options):
        """Same projection as the serial variant; MATLAB only differs by using parfor.

        Returns (Projections2D, Angles_new, points, RoI).
        """
        Projections2D, Angles_new, points, _, _, _, RoI = \
            ForwardProjection.RayDrivenForwProjParCartesianMultipleWithMotionArbitraryOrder(
                x0, dx, Nx, y0, dy, Ny, N, xi0, dxi, M, N360, TRot, ThetaArray, **options
            )

        return Projections2D, Angles_new, points, RoI

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
        from ..Simulation.CreatePhantom import CreatePhantom

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
        from ..Simulation.CreatePhantom import CreatePhantom

        ThetaArray = np.asarray(ThetaArray, dtype=float).ravel()

        Projections2D = Allocation.zeros(N, M)
        PhantomTruth = Allocation.zeros(N, M)
        nT_Phases = np.asarray([], dtype=float)
        points = Allocation.zeros(2, N)

        T0 = Shift_GT
        dT = TRot / N360

        for n in range(1, N + 1):
            if PhantomOld:
                Image, _, _, _ = CreatePhantom.CreateVoxelPhantom2DWithMotionRespiratory(
                    Nx, dx, x0, Ny, dy, y0, T0, dT, n, "CardiacPhantomWithPlaque"
                )
                px = py = 0
            else:
                Image, _, px, py, _ = CreatePhantom.CreateVoxelPhantom2DWithMotionUnified(
                    Nx, dx, x0, Ny, dy, y0, T0, dT, n, SelectedPhantom,
                    Heartrate=Heartrate, CM=CM, Stent=Stent, Plaque=Plaque, Motion=Motion,
                    Direction=Direction, Rotation=Rotation, LineWidth=LineWidth, Center=Center,
                    MotionType=MotionType, TRest=TRest
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

        SpectralProjections = Allocation.zeros(N_Ebins, N * M)
        for nbin in range(1, N_Ebins + 1):
            energy = _fix(E0 + nbin * deltaE)
            Image = aMaterialProperties.MaterialToSpectralAttenuationCoefficientConversion(
                MaterialImage, Nx, Ny, energy
            )
            SpectralProjections[nbin - 1, :] = ForwardProjection.RayDrivenForwProjParCartesianMultiple(
                x0, dx, Nx, y0, dy, Ny, Image, theta0, dtheta, N, xi0, dxi, M
            )

        return _SpectralAttenuation(SpectralProjections, I0Values, SumOverI0)

    @staticmethod
    def SpectralRayDrivenForwardProj2DFanMultiple(RF, alpha0, dalpha, beta, x0, dx, Nx, y0, dy, Ny,
                                                  MaterialImage, M, N,
                                                  spectrum, E0, N_Ebins, deltaE):
        """Polychromatic fan-beam projection of a material-labelled phantom."""
        aMaterialProperties, I0Values, SumOverI0 = _SpectralBins(spectrum, E0, N_Ebins, deltaE)

        SpectralProjections = Allocation.zeros(N_Ebins, N * M)
        for nbin in range(1, N_Ebins + 1):
            energy = _fix(E0 + nbin * deltaE)
            Image = aMaterialProperties.MaterialToSpectralAttenuationCoefficientConversion(
                MaterialImage, Nx, Ny, energy
            )
            SpectralProjections[nbin - 1, :] = ForwardProjection.RayDrivenForwardProj2DFanMultiple(
                RF, alpha0, dalpha, beta, x0, dx, Nx, y0, dy, Ny, Image, M, N
            )

        return _SpectralAttenuation(SpectralProjections, I0Values, SumOverI0)

    # ------------------------------------------------------------------- 3D fan

    @staticmethod
    def RayDrivenForwardProj3DFanSingle(RF, alpha, beta, s0, ds, x0, dx, Nx, y0, dy, Ny,
                                        z0, dz, Nz, Ima, M, Ns, zsource):
        """One cone-beam fan projection of (M fan channels x Ns detector rows)."""
        Ima = np.asarray(Ima, dtype=float).ravel()
        beta = np.atleast_1d(np.asarray(beta, dtype=float))

        sx = RF * np.sin(alpha)
        sy = -RF * np.cos(alpha)

        ns = np.arange(1, Ns + 1)
        # A converts the in-plane distance from the source into a z offset of the ray.
        A = ds / (RF * 2.0) * (2 * ns - Ns - 1)
        sl = s0 + ns * ds
        cos_gamma = np.cos(np.abs(np.arctan(sl / RF)))

        Projection = Allocation.zeros(M * Ns)
        Rows = Projection.reshape(Ns, M)

        for m in range(1, M + 1):
            theta = alpha + beta[m - 1]
            c = np.cos(theta)
            s = np.sin(theta)

            xi = sx * c + sy * s
            B = np.sqrt(RF * RF - xi * xi)

            Rows[:, m - 1] = _TraceRay3D(xi, B, c, s, A, cos_gamma, zsource,
                                         x0, dx, Nx, y0, dy, Ny, z0, dz, Nz, Ima)

        return Projection

    @staticmethod
    def RayDrivenForwardProj3DFanMultipleSequence(RF, alpha0, dalpha, beta, s0, ds,
                                                  x0, dx, Nx, y0, dy, Ny, z0, dz, Nz,
                                                  Ima, M, Ns, N, zn):
        """All N readings of one sequence scan at the fixed table position zn."""
        Projections = Allocation.zeros(N * M * Ns)

        for n in range(1, N + 1):
            alpha = alpha0 + n * dalpha
            Projections[(n - 1) * M * Ns:n * M * Ns] = (
                ForwardProjection.RayDrivenForwardProj3DFanSingle(
                    RF, alpha, beta, s0, ds, x0, dx, Nx, y0, dy, Ny,
                    z0, dz, Nz, Ima, M, Ns, zn
                )
            )

        return Projections

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

        Projections = Allocation.zeros(N * M * Ns)

        for n in range(1, N + 1):
            alpha = alpha0 + n * dalpha
            # The focus advances by pitch * collimation per full rotation.
            zn = zfocus0 + (alpha - alpha0) * pitch * Ns * ds / (2 * np.pi)
            Projections[(n - 1) * M * Ns:n * M * Ns] = (
                ForwardProjection.RayDrivenForwardProj3DFanSingle(
                    RF, alpha, beta, s0, ds,
                    G.x0sim, G.dxsim, G.Nxsim, G.y0sim, G.dysim, G.Nysim,
                    G.z0sim, G.dzsim, G.Nzsim, Ima_in, M, Ns, zn
                )
            )

        return Projections

    # ------------------------------------------------------------- 3D static CT

    @staticmethod
    def RayDrivenForwardProj3DStaticCTSingle(RF, RD, alpha, delta0, dDelta, s0, ds, soffset,
                                             x0, dx, Nx, y0, dy, Ny, z0, dz, Nz, Ima,
                                             MDelta, Ns, zsource):
        """One static CT cone-beam projection; detector elements sit on the ring at RD."""
        Ima = np.asarray(Ima, dtype=float).ravel()

        sinalpha = np.sin(alpha)
        cosalpha = np.cos(alpha)
        sx = RF * sinalpha
        sy = -RF * cosalpha

        RFsquare = RF * RF
        RFsquarePlusRDsquare = RFsquare + RD * RD
        TwoRDRF = 2 * RD * RF

        ns = np.arange(1, Ns + 1)
        A_nom = soffset + ds * (ns - Ns - 0.5)
        sl = s0 + ns * ds

        Projection = Allocation.zeros(MDelta * Ns)
        Rows = Projection.reshape(Ns, MDelta)

        for m in range(1, MDelta + 1):
            delta = delta0 + m * dDelta

            # L in the geometry description: source-to-element distance on the ring.
            A_denom = np.sqrt(RFsquarePlusRDsquare + TwoRDRF * np.sin(delta - alpha))
            A = A_nom / A_denom
            cos_gamma = np.cos(np.abs(np.arctan(sl / A_denom)))

            betaDash = delta - alpha - np.pi / 2
            beta = np.arctan((np.sin(betaDash) * RD) / (np.cos(betaDash) * RD + RF))

            theta = alpha + beta
            c = np.cos(theta)
            s = np.sin(theta)

            # Arbitrary fan limitation: the ray must not hit the detector element from behind.
            if np.abs(np.arccos(np.clip(-s * np.cos(delta) + c * np.sin(delta), -1.0, 1.0))) >= np.pi / 2:
                continue

            xi = sx * c + sy * s
            B = np.sqrt(RFsquare - xi * xi)

            Rows[:, m - 1] = _TraceRay3D(xi, B, c, s, A, cos_gamma, zsource,
                                         x0, dx, Nx, y0, dy, Ny, z0, dz, Nz, Ima)

        return Projection

    @staticmethod
    def RayDrivenForwardProj3DStaticCTMultipleSequence(RF, RD, alpha0, dalpha, delta0, dDelta,
                                                       s0, ds, soffset, x0, dx, Nx, y0, dy, Ny,
                                                       z0, dz, Nz, Ima, MDelta, Ns, N, zn):
        Projections = Allocation.zeros(N * MDelta * Ns)
        block = MDelta * Ns

        for n in range(1, N + 1):
            alpha = alpha0 + n * dalpha
            Projections[(n - 1) * block:n * block] = (
                ForwardProjection.RayDrivenForwardProj3DStaticCTSingle(
                    RF, RD, alpha, delta0, dDelta, s0, ds, soffset,
                    x0, dx, Nx, y0, dy, Ny, z0, dz, Nz, Ima, MDelta, Ns, zn
                )
            )

        return Projections

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

        Projections = Allocation.zeros(N * MDelta * Ns)
        block = MDelta * Ns

        RFD = RF + RD
        for n in range(1, N + 1):
            alpha = alpha0 + n * dalpha
            # The static CT source ring advances with the magnified collimation RF/RFD.
            zn = z0focus + (alpha - alpha0) * pitch * Ns * ds * RF / RFD / (2 * np.pi)
            Projections[(n - 1) * block:n * block] = (
                ForwardProjection.RayDrivenForwardProj3DStaticCTSingle(
                    RF, RD, alpha, delta0, dDelta, s0, ds, soffset,
                    x0, dx, Nx, y0, dy, Ny, z0, dz, Nz, Ima_in, MDelta, Ns, zn
                )
            )

        return Projections

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
        rows = dFVecArray_in.shape[0]

        ArrayOfCollimationOfOneProjection = Allocation.zeros(rows, Ns * MDelta)
        for i in range(rows):
            ArrayOfCollimationOfOneProjection[i, :] = ForwardProjection.StaticCTCollimationSingle(
                StaticCTScanDescriptor_in, alpha_in, dFVecArray_in[i, :]
            )

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
        rows = dFVecArray_in.shape[0]

        ArrayOfCollimationOfOneProjection = Allocation.zeros(rows, Ns * MDelta)
        for i in range(rows):
            ArrayOfCollimationOfOneProjection[i, :] = ForwardProjection.StaticCTCollimationLeafSingle(
                StaticCTScanDescriptor_in, alpha_in, dFVecArray_in[i, :], LeafConfig
            )

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
