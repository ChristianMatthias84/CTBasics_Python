import numpy as np

from ..Memory.Allocation import Allocation
from ..MultiCore.Partition.SequenceStacks import SequenceStacks
from ..MultiCore.Partition.SpiralStacks import SpiralStacks


def _fix(value):
    """MATLAB fix(): truncate toward zero."""
    return int(value)




def _fix_array(values):
    """Vectorized MATLAB fix() returning integer indices."""
    return np.trunc(values).astype(np.int64)


def _InPlaneRay(theta, x0, dx, Nx, y0, dy, Ny, xi0, dxi, M):
    """Channel coordinate of every pixel for one parallel view.

    Returns (mreal, m, wm, valid, xi, cc, ss) with the (Ny, Nx) arrays of the MATLAB
    expression mreal = a*nx + b*ny + c for a = dx*cos/dxi, b = dy*sin/dxi.
    """
    cc = np.cos(theta)
    ss = np.sin(theta)

    a = dx * cc / dxi
    b = dy * ss / dxi
    c = (x0 * cc + y0 * ss - xi0) / dxi

    nx = np.arange(1, Nx + 1)
    ny = np.arange(1, Ny + 1)

    mreal = a * nx[None, :] + b * ny[:, None] + c
    m = _fix_array(mreal)

    return mreal, m, mreal - m, (m >= 1) & (m < M), xi0 + mreal * dxi, cc, ss


def _GatherSino3D(Sino, base, M, wm, wns, valid):
    """Bilinear sample of one sinogram in the channel and the detector row direction."""
    safe = np.where(valid, base, 0)

    Reproj_ns = (1 - wm) * Sino[safe] + wm * Sino[safe + 1]
    Reproj_ns_plus1 = (1 - wm) * Sino[safe + M] + wm * Sino[safe + M + 1]

    return np.where(valid, (1 - wns) * Reproj_ns + wns * Reproj_ns_plus1, 0.0)


def _HalfTurnGroups(N, Npi):
    """MATLAB loops nthetaBar = 1..Npi and then n = nthetaBar:Npi:N."""
    return [list(range(nthetaBar, N + 1, Npi)) for nthetaBar in range(1, Npi + 1)]


class Backprojection:
    """Pixel-driven backprojectors.

    The |dtheta| factor from the inverse Radon transform is applied once at the end,
    matching the MATLAB original.
    """

    @staticmethod
    def PixDrivenBackProjParCartesianSingleBase(M, Nx, Ny, a, b, c, Proj):
        Proj = np.asarray(Proj, dtype=float)

        nx = np.arange(1, Nx + 1)
        ny = np.arange(1, Ny + 1)
        mreal = a * nx[None, :] + b * ny[:, None] + c
        m = np.trunc(mreal).astype(np.int64)
        w = mreal - m
        valid = (m >= 1) & (m < M)

        safe = np.where(valid, m, 1)
        contribution = (1 - w) * Proj[safe - 1] + w * Proj[safe]

        Reprojection = Allocation.zeros(Nx * Ny)
        Reprojection[:] = np.where(valid, contribution, 0.0).ravel()
        return Reprojection

    @staticmethod
    def PixDrivenBackProjParCartesianSingle(theta, xi0, dxi, M, Proj, x0, dx, Nx, y0, dy, Ny):
        # xi = x*cos(theta) + y*sin(theta)
        cc = np.cos(theta)
        ss = np.sin(theta)
        a = dx * cc / dxi
        b = dy * ss / dxi
        c = (x0 * cc + y0 * ss - xi0) / dxi
        return Backprojection.PixDrivenBackProjParCartesianSingleBase(M, Nx, Ny, a, b, c, Proj)

    @staticmethod
    def PixDrivenBackProjParCartesian(theta0, dtheta, N, N2pi, xi0, dxi, M, Sino, x0, dx, Nx, y0, dy, Ny):
        Sino = np.asarray(Sino, dtype=float)

        # Rays a half turn apart are redundant, so each half-turn group shares one norm.
        # MATLAB buffers all N backprojections first; normalizing on the fly is equivalent
        # and avoids an (N, Nx*Ny) array.
        Npi = int(N2pi // 2)
        Norm = np.array([len(range(nbar, N + 1, Npi)) for nbar in range(1, Npi + 1)], dtype=float)

        Reconstruction = Allocation.zeros(Nx * Ny)
        for n in range(1, N + 1):
            norm = Norm[(n - 1) % Npi]
            if norm == 0:
                continue
            Projection = Sino[(n - 1) * M:n * M]
            Reconstruction += Backprojection.PixDrivenBackProjParCartesianSingle(
                theta0 + n * dtheta, xi0, dxi, M, Projection, x0, dx, Nx, y0, dy, Ny
            ) / norm

        return Reconstruction * dtheta

    @staticmethod
    def WFBP2DFullyPixelDriven(theta0, dtheta, N, N2pi, xi0, dxi, M, Sino, x0, dx, Nx, y0, dy, Ny):
        """Weighted FBP that stays correct beyond a pi range by normalizing per half turn.

        Unlike PixDrivenBackProjParCartesian the weighting is per voxel: each voxel counts
        only the projections that actually hit the detector, so truncated views do not bias it.
        """
        Sino = np.asarray(Sino, dtype=float)
        Npi = int(N2pi // 2)

        nx = np.arange(1, Nx + 1)
        ny = np.arange(1, Ny + 1)

        Reconstruction = Allocation.zeros(Nx * Ny)

        for nthetaBar in range(1, Npi + 1):
            Vxy = np.zeros((Ny, Nx))
            Hxy = np.zeros((Ny, Nx))

            for n in range(nthetaBar, N + 1, Npi):
                theta = theta0 + n * dtheta
                cc = np.cos(theta)
                ss = np.sin(theta)
                a = dx * cc / dxi
                b = dy * ss / dxi
                c = (x0 * cc + y0 * ss - xi0) / dxi

                mreal = a * nx[None, :] + b * ny[:, None] + c
                m = np.trunc(mreal).astype(np.int64)
                w = mreal - m
                valid = (m >= 1) & (m < M)

                safe = np.where(valid, m, 1)
                Projection = Sino[(n - 1) * M:n * M]
                contribution = (1 - w) * Projection[safe - 1] + w * Projection[safe]

                Vxy += np.where(valid, contribution, 0.0)
                Hxy += valid

            np.divide(Vxy, Hxy, out=Vxy, where=Hxy > 0)
            Reconstruction += np.where(Hxy > 0, Vxy, 0.0).ravel()

        return Reconstruction * dtheta

    # ------------------------------------------------------- 3D parallel beam

    @staticmethod
    def PixDriven3DBackProjParCartesianSingleBase(RF, M, Ns, ds, xi0, dxi, Nx, x0, dx,
                                                  Ny, y0, dy, Nz, z0, dz, a, b, c, ss, cc, Proj):
        Proj = np.asarray(Proj, dtype=float).ravel()

        A = Ns / 2.0 + 0.5
        RFsquare = RF * RF

        nx = np.arange(1, Nx + 1)
        ny = np.arange(1, Ny + 1)
        x = x0 + nx * dx
        y = y0 + ny * dy
        z = z0 + np.arange(1, Nz + 1) * dz

        mreal = a * nx[None, :] + b * ny[:, None] + c
        m = _fix_array(mreal)
        wm = mreal - m
        valid_m = (m >= 1) & (m < M)

        xi = xi0 + mreal * dxi

        # Distance from the source plane to the voxel, measured along the ray.
        denom = (np.sqrt(np.clip(RFsquare - xi * xi, 0.0, None))
                 - x[None, :] * ss + (y * cc)[:, None]) * ds

        with np.errstate(divide="ignore", invalid="ignore"):
            nsreal = (z[:, None, None] * RF) / denom[None, :, :] + A

        nsreal = np.nan_to_num(nsreal, nan=-1.0, posinf=-1.0, neginf=-1.0)
        ns = _fix_array(nsreal)
        wns = nsreal - ns

        valid = valid_m[None, :, :] & (ns >= 1) & (ns < Ns)
        base = (ns - 1) * M + np.broadcast_to(m, ns.shape) - 1

        samples = _GatherSino3D(Proj, base, M, np.broadcast_to(wm, ns.shape), wns, valid)

        Reprojection = Allocation.zeros(Nx * Ny * Nz)
        Reprojection[:] = samples.ravel()
        return Reprojection

    @staticmethod
    def PixDriven3DBackProjParCartesianSingle(theta, RF, xi0, dxi, M, Ns, ds, Proj,
                                              x0, dx, Nx, y0, dy, Ny, z0, dz, Nz):
        cc = np.cos(theta)
        ss = np.sin(theta)

        a = dx * cc / dxi
        b = dy * ss / dxi
        c = (x0 * cc + y0 * ss - xi0) / dxi

        return Backprojection.PixDriven3DBackProjParCartesianSingleBase(
            RF, M, Ns, ds, xi0, dxi, Nx, x0, dx, Ny, y0, dy, Nz, z0, dz, a, b, c, ss, cc, Proj
        )

    @staticmethod
    def PixDriven3DBackProjParCartesian(RF, theta0, dtheta, N, N2pi, xi0, dxi, M,
                                        s0, ds, Ns, Sino, x0, dx, Nx, y0, dy, Ny, z0, dz, Nz):
        Sino = np.array(Sino, dtype=float).ravel()

        # 1. Weight every detector row with the cosine of its cone angle.
        ns = np.arange(1, Ns + 1)
        cos_gamma = np.cos(np.abs(np.arctan((s0 + ns * ds) / RF)))
        Sino = (Sino.reshape(N, Ns, M) * cos_gamma[None, :, None]).ravel()

        # 2. Backproject each view, 3. average over the redundant half turns.
        Npi = int(N2pi // 2)
        block = M * Ns

        Reconstruction = Allocation.zeros(Nx * Ny * Nz)
        for group in _HalfTurnGroups(N, Npi):
            Partial = Allocation.zeros(Nx * Ny * Nz)
            for n in group:
                Partial += Backprojection.PixDriven3DBackProjParCartesianSingle(
                    theta0 + n * dtheta, RF, xi0, dxi, M, Ns, ds,
                    Sino[(n - 1) * block:n * block], x0, dx, Nx, y0, dy, Ny, z0, dz, Nz
                )
            Reconstruction += Partial / len(group)

        return Reconstruction * dtheta

    # ------------------------------------------------------------- 3D WFBP

    @staticmethod
    def _WFBP3D(N, N2pi, Nx, Ny, Nz, Sample):
        """Shared half-turn averaging of the weighted filtered backprojection.

        `Sample(n)` returns (contribution, valid) for projection n over the whole volume.
        Voxels that never see a contribution for some half-turn group are incomplete and
        are zeroed at the end, exactly like the MATLAB DataCompletenessVolume.
        """
        shape = (Nz, Ny, Nx)
        Npi = int(N2pi // 2)

        Reconstruction = np.zeros(shape)
        HxyzVolume = np.zeros(shape)
        Complete = np.ones(shape, dtype=bool)

        for group in _HalfTurnGroups(N, Npi):
            Vxyz = np.zeros(shape)
            Hxyz = np.zeros(shape, dtype=np.int64)

            for n in group:
                contribution, valid = Sample(n)
                Vxyz += contribution
                Hxyz += valid

            hit = Hxyz > 0
            Reconstruction[hit] += Vxyz[hit] / Hxyz[hit]
            Complete &= hit
            HxyzVolume += Hxyz

        Reconstruction[~Complete] = 0.0

        return Reconstruction.ravel(), HxyzVolume.ravel()

    @staticmethod
    def SpiralWFBP3DFullyPixelDriven(RF, theta0, dtheta, N, N2pi, xi0, dxi, M, s0, ds, Ns,
                                     Sino, x0, dx, Nx, y0, dy, Ny, z0, dz, Nz,
                                     z0focus, pitch, alpha0):
        Sino = np.asarray(Sino, dtype=float).ravel()

        A = Ns / 2.0 + 0.5
        RFsquare = RF * RF
        M_Ns = M * Ns
        zRot_2pi = pitch * Ns * ds / (2 * np.pi)

        x = x0 + np.arange(1, Nx + 1) * dx
        y = y0 + np.arange(1, Ny + 1) * dy
        z = z0 + np.arange(1, Nz + 1) * dz

        def Sample(n):
            theta = theta0 + n * dtheta
            mreal, m, wm, valid_m, xi, cc, ss = _InPlaneRay(
                theta, x0, dx, Nx, y0, dy, Ny, xi0, dxi, M
            )

            # The focus height follows the source angle alpha, not the parallel angle theta.
            alpha = theta + np.arcsin(np.clip(xi / RF, -1.0, 1.0))
            zn = z0focus + (alpha - alpha0) * zRot_2pi

            denom = (np.sqrt(np.clip(RFsquare - xi * xi, 0.0, None))
                     - x[None, :] * ss + (y * cc)[:, None]) * ds

            with np.errstate(divide="ignore", invalid="ignore"):
                nsreal = ((z[:, None, None] - zn[None, :, :]) * RF) / denom[None, :, :] + A

            nsreal = np.nan_to_num(nsreal, nan=-1.0, posinf=-1.0, neginf=-1.0)
            ns = _fix_array(nsreal)

            valid = valid_m[None, :, :] & (ns >= 1) & (ns < Ns)
            base = (n - 1) * M_Ns + (ns - 1) * M + np.broadcast_to(m, ns.shape) - 1

            contribution = _GatherSino3D(
                Sino, base, M, np.broadcast_to(wm, ns.shape), nsreal - ns, valid
            )
            return contribution, valid

        Reconstruction, HxyzVolume = Backprojection._WFBP3D(N, N2pi, Nx, Ny, Nz, Sample)
        return Reconstruction * dtheta, HxyzVolume

    @staticmethod
    def SequenceWFBP3DFullyPixelDriven(RF, theta0, dtheta, N, N2pi, xi0, dxi, M, s0, ds, Ns,
                                       Sino, x0, dx, Nx, y0, dy, Ny, z0, dz, Nz,
                                       z0focus, dzfocus, NzScans):
        Sino = np.asarray(Sino, dtype=float).ravel()
        theta0 = np.atleast_1d(np.asarray(theta0, dtype=float))
        NzScans = np.atleast_1d(np.asarray(NzScans))

        A = Ns / 2.0 + 0.5
        RFsquare = RF * RF
        M_Ns = M * Ns
        M_Ns_N = M * Ns * N

        x = x0 + np.arange(1, Nx + 1) * dx
        y = y0 + np.arange(1, Ny + 1) * dy
        z = z0 + np.arange(1, Nz + 1) * dz

        def Sample(n):
            contribution = np.zeros((Nz, Ny, Nx))
            valid_total = np.zeros((Nz, Ny, Nx), dtype=np.int64)

            # Every sequence scan position contributes to the same half-turn group.
            for nzScan in range(1, len(NzScans) + 1):
                ScanIdx = int(NzScans[nzScan - 1])
                zn = z0focus + (nzScan - 1) * dzfocus

                theta = theta0[ScanIdx - 1] + n * dtheta
                mreal, m, wm, valid_m, xi, cc, ss = _InPlaneRay(
                    theta, x0, dx, Nx, y0, dy, Ny, xi0, dxi, M
                )

                denom = (np.sqrt(np.clip(RFsquare - xi * xi, 0.0, None))
                         - x[None, :] * ss + (y * cc)[:, None]) * ds

                with np.errstate(divide="ignore", invalid="ignore"):
                    nsreal = ((z[:, None, None] - zn) * RF) / denom[None, :, :] + A

                nsreal = np.nan_to_num(nsreal, nan=-1.0, posinf=-1.0, neginf=-1.0)
                ns = _fix_array(nsreal)

                valid = valid_m[None, :, :] & (ns >= 1) & (ns < Ns)
                base = ((nzScan - 1) * M_Ns_N + (n - 1) * M_Ns + (ns - 1) * M
                        + np.broadcast_to(m, ns.shape) - 1)

                contribution += _GatherSino3D(
                    Sino, base, M, np.broadcast_to(wm, ns.shape), nsreal - ns, valid
                )
                valid_total += valid

            return contribution, valid_total

        Reconstruction, HxyzVolume = Backprojection._WFBP3D(N, N2pi, Nx, Ny, Nz, Sample)
        return Reconstruction * dtheta, HxyzVolume

    # --------------------------------------------------------- 3D WFBP static CT

    @staticmethod
    def _StaticCTRowCoordinate(RF, RD, xi, x, y, ss, cc, z, zn, ds, A):
        """Detector row coordinate of the static CT geometry.

        The source-to-element distance L depends on betaDash, which follows from the
        parallel coordinate xi. MATLAB wraps the whole bracket of `l` in the square root,
        unlike the fan variant, and that form is kept verbatim here.
        """
        RFsquare = RF * RF
        RFbyRD = RF / RD

        beta_real = -np.arcsin(np.clip(xi / RF, -1.0, 1.0))

        tan_beta_real_square = np.tan(beta_real) ** 2
        # Rays beyond the detector ring make the radicand negative; MATLAB continues with a
        # complex value there, the port lets the NaN mark the sample as invalid.
        with np.errstate(invalid="ignore"):
            argument = ((-tan_beta_real_square * RFbyRD
                         + np.sqrt(tan_beta_real_square * (1 - RFbyRD * RFbyRD) + 1))
                        / (tan_beta_real_square + 1))
        betaDash_real = np.sign(beta_real) * np.arccos(np.clip(argument, -1.0, 1.0))

        L = np.sqrt(RFsquare + RD * RD + 2 * RD * RF * np.sin(betaDash_real + np.pi / 2))
        l = np.sqrt(np.clip((RFsquare - xi * xi) - x[None, :] * ss + (y * cc)[:, None],
                            0.0, None))

        with np.errstate(divide="ignore", invalid="ignore"):
            nsreal = (z[:, None, None] - zn) * L[None, :, :] / (l[None, :, :] * ds) + A

        return np.nan_to_num(nsreal, nan=-1.0, posinf=-1.0, neginf=-1.0), beta_real

    @staticmethod
    def SequenceWFBP3DFullyPixelDriven_StaticCT(RF, RD, theta0, dtheta, N, N2pi, xi0, dxi,
                                                Mpar, s0, ds, Ns, soffset, Sino,
                                                x0, dx, Nx, y0, dy, Ny, z0, dz, Nz,
                                                z0focus, dzfocus, NzScans):
        Sino = np.asarray(Sino, dtype=float).ravel()
        NzScans = np.atleast_1d(np.asarray(NzScans))

        # The static CT detector is shifted by soffset instead of being centred on zero.
        A = -soffset / ds + Ns + 0.5
        M_Ns = Mpar * Ns
        M_Ns_N = Mpar * Ns * N

        x = x0 + np.arange(1, Nx + 1) * dx
        y = y0 + np.arange(1, Ny + 1) * dy
        z = z0 + np.arange(1, Nz + 1) * dz

        def Sample(n):
            contribution = np.zeros((Nz, Ny, Nx))
            valid_total = np.zeros((Nz, Ny, Nx), dtype=np.int64)

            theta = theta0 + n * dtheta
            mreal, m, wm, valid_m, xi, cc, ss = _InPlaneRay(
                theta, x0, dx, Nx, y0, dy, Ny, xi0, dxi, Mpar
            )

            for nzScan in range(1, len(NzScans) + 1):
                zn = z0focus + (nzScan - 1) * dzfocus

                nsreal, _ = Backprojection._StaticCTRowCoordinate(
                    RF, RD, xi, x, y, ss, cc, z, zn, ds, A
                )
                ns = _fix_array(nsreal)

                valid = valid_m[None, :, :] & (ns >= 1) & (ns < Ns)
                base = ((nzScan - 1) * M_Ns_N + (n - 1) * M_Ns + (ns - 1) * Mpar
                        + np.broadcast_to(m, ns.shape) - 1)

                contribution += _GatherSino3D(
                    Sino, base, Mpar, np.broadcast_to(wm, ns.shape), nsreal - ns, valid
                )
                valid_total += valid

            return contribution, valid_total

        Reconstruction, HxyzVolume = Backprojection._WFBP3D(N, N2pi, Nx, Ny, Nz, Sample)
        return Reconstruction * dtheta, HxyzVolume

    @staticmethod
    def SpiralWFBP3DFullyPixelDriven_StaticCT(RF, RD, theta0, dtheta, N, N2pi, xi0, dxi,
                                              Mpar, s0, ds, Ns, soffset, Sino,
                                              x0, dx, Nx, y0, dy, Ny, z0, dz, Nz,
                                              z0focus, pitch, alpha0):
        Sino = np.asarray(Sino, dtype=float).ravel()

        A = -soffset / ds + Ns + 0.5
        M_Ns = Mpar * Ns
        zRot_2pi = pitch * Ns * ds / (2 * np.pi)

        x = x0 + np.arange(1, Nx + 1) * dx
        y = y0 + np.arange(1, Ny + 1) * dy
        z = z0 + np.arange(1, Nz + 1) * dz

        def Sample(n):
            theta = theta0 + n * dtheta
            mreal, m, wm, valid_m, xi, cc, ss = _InPlaneRay(
                theta, x0, dx, Nx, y0, dy, Ny, xi0, dxi, Mpar
            )

            beta_real = -np.arcsin(np.clip(xi / RF, -1.0, 1.0))
            alpha = theta - beta_real
            zn = z0focus + (alpha - alpha0) * zRot_2pi

            nsreal, _ = Backprojection._StaticCTRowCoordinate(
                RF, RD, xi, x, y, ss, cc, z, zn[None, :, :], ds, A
            )
            ns = _fix_array(nsreal)

            valid = valid_m[None, :, :] & (ns >= 1) & (ns < Ns)
            base = (n - 1) * M_Ns + (ns - 1) * Mpar + np.broadcast_to(m, ns.shape) - 1

            contribution = _GatherSino3D(
                Sino, base, Mpar, np.broadcast_to(wm, ns.shape), nsreal - ns, valid
            )
            return contribution, valid

        Reconstruction, HxyzVolume = Backprojection._WFBP3D(N, N2pi, Nx, Ny, Nz, Sample)
        return Reconstruction * dtheta, HxyzVolume

    # ------------------------------------------------- stacked / parallel wrappers

    @staticmethod
    def _WeightConeAngle(Sino, N, Ns, M, s0, ds, RF, NzScans=1):
        """Step 1 of the WFBP: weight every detector row with the cosine of its cone angle."""
        Sino = np.array(Sino, dtype=float).ravel()

        ns = np.arange(1, Ns + 1)
        cos_gamma = np.cos(np.abs(np.arctan((s0 + ns * ds) / RF)))

        return (Sino.reshape(NzScans, N, Ns, M) * cos_gamma[None, None, :, None]).ravel()

    @staticmethod
    def _MergeStacks(VolumeStacks, HxyzStacks, StackSizes, Nx, Ny):
        """Concatenates the z-stacks back into one flat volume."""
        Nz = int(np.sum(StackSizes))

        Reconstruction = Allocation.zeros(Nx * Ny * Nz)
        HxyzVolume = Allocation.zeros(Nx * Ny * Nz)

        offset = 0
        for i, size in enumerate(StackSizes):
            span = int(size) * Nx * Ny

            if VolumeStacks[i] is not None:
                Reconstruction[offset:offset + span] = VolumeStacks[i]
                HxyzVolume[offset:offset + span] = HxyzStacks[i]

            offset += span

        return Reconstruction, HxyzVolume

    @staticmethod
    def SpiralWFBP3DFullyPixelDrivenParallelized(Sino, FanBeamGeometryDescriptor_in,
                                                 ParallelBeamGeometryDescriptor_in,
                                                 ReconVolumeGeometry_in):
        Geometry = FanBeamGeometryDescriptor_in.Geometry
        Parameter = FanBeamGeometryDescriptor_in.Parameter

        RF = Geometry.RF
        s0 = Geometry.s0
        ds = Geometry.ds
        Ns = Geometry.Ns

        z0focus = Parameter.z0focus
        pitch = Parameter.pitch
        alpha0 = Parameter.alpha0

        dtheta = ParallelBeamGeometryDescriptor_in.dtheta
        N = ParallelBeamGeometryDescriptor_in.Npar
        N2pi = ParallelBeamGeometryDescriptor_in.N2pi
        xi0 = ParallelBeamGeometryDescriptor_in.xi0
        dxi = ParallelBeamGeometryDescriptor_in.dxi
        M = ParallelBeamGeometryDescriptor_in.M

        x0 = ReconVolumeGeometry_in.x0
        dx = ReconVolumeGeometry_in.dx
        Nx = ReconVolumeGeometry_in.Nx
        y0 = ReconVolumeGeometry_in.y0
        dy = ReconVolumeGeometry_in.dy
        Ny = ReconVolumeGeometry_in.Ny
        dz = ReconVolumeGeometry_in.dz

        Sino = Backprojection._WeightConeAngle(Sino, N, Ns, M, s0, ds, RF)

        Stacks = SpiralStacks(FanBeamGeometryDescriptor_in,
                              ParallelBeamGeometryDescriptor_in,
                              ReconVolumeGeometry_in)

        StackSizes = Stacks.StackSizes
        TotalItems = len(StackSizes)

        SinoStacks, NStacks, thetaStacks = Stacks.Apply(Sino)
        z0Stack, FirstReadingStack, FinalReadingStack = Stacks.GetStacks()

        VolumeStacks = [None] * TotalItems
        HxyzStacks = [None] * TotalItems

        # MATLAB uses parfor here; the port runs the stacks serially.
        for i in range(TotalItems):
            if FirstReadingStack[i] < N and FinalReadingStack[i] > 1:
                VolumeStacks[i], HxyzStacks[i] = Backprojection.SpiralWFBP3DFullyPixelDriven(
                    RF, thetaStacks[i], dtheta, int(NStacks[i]), N2pi, xi0, dxi, M, s0, ds, Ns,
                    SinoStacks[i], x0, dx, Nx, y0, dy, Ny, z0Stack[i], dz, int(StackSizes[i]),
                    z0focus, pitch, alpha0
                )

        return Backprojection._MergeStacks(VolumeStacks, HxyzStacks, StackSizes, Nx, Ny)

    @staticmethod
    def SpiralWFBP3DFullyPixelDrivenParallelizedStacked(Sino, FanBeamGeometryDescriptor_in,
                                                        ParallelBeamGeometryDescriptor_in,
                                                        ReconVolumeGeometry_in):
        return Backprojection._Stacked(
            ReconVolumeGeometry_in,
            lambda: Backprojection.SpiralWFBP3DFullyPixelDrivenParallelized(
                Sino, FanBeamGeometryDescriptor_in,
                ParallelBeamGeometryDescriptor_in, ReconVolumeGeometry_in
            ),
        )

    @staticmethod
    def SequenceWFBP3DFullyPixelDrivenParallelized(Sino, FanBeamGeometryDescriptor_in,
                                                   ParallelBeamGeometryDescriptor_in,
                                                   ReconVolumeGeometry_in):
        Geometry = FanBeamGeometryDescriptor_in.Geometry
        Parameter = FanBeamGeometryDescriptor_in.Parameter

        RF = Geometry.RF
        s0 = Geometry.s0
        ds = Geometry.ds
        Ns = Geometry.Ns

        dzfocus = Parameter.dzfocus
        NzScans = Parameter.NzScans

        theta0 = ParallelBeamGeometryDescriptor_in.theta0
        dtheta = ParallelBeamGeometryDescriptor_in.dtheta
        N = ParallelBeamGeometryDescriptor_in.Npar
        N2pi = ParallelBeamGeometryDescriptor_in.N2pi
        xi0 = ParallelBeamGeometryDescriptor_in.xi0
        dxi = ParallelBeamGeometryDescriptor_in.dxi
        M = ParallelBeamGeometryDescriptor_in.M

        x0 = ReconVolumeGeometry_in.x0
        dx = ReconVolumeGeometry_in.dx
        Nx = ReconVolumeGeometry_in.Nx
        y0 = ReconVolumeGeometry_in.y0
        dy = ReconVolumeGeometry_in.dy
        Ny = ReconVolumeGeometry_in.Ny
        dz = ReconVolumeGeometry_in.dz

        Sino = Backprojection._WeightConeAngle(Sino, N, Ns, M, s0, ds, RF, NzScans=int(NzScans))

        Stacks = SequenceStacks(FanBeamGeometryDescriptor_in,
                                ParallelBeamGeometryDescriptor_in,
                                ReconVolumeGeometry_in)

        StackSizes = Stacks.StackSizes
        TotalItems = len(StackSizes)

        SinoStacks, z0focousStacks = Stacks.Apply(Sino)
        z0stack, Scans = Stacks.GetStacks()

        VolumeStacks = [None] * TotalItems
        HxyzStacks = [None] * TotalItems

        for i in range(TotalItems):
            selectedScans = Scans[i]

            if selectedScans is not None and len(selectedScans) > 0:
                VolumeStacks[i], HxyzStacks[i] = Backprojection.SequenceWFBP3DFullyPixelDriven(
                    RF, theta0, dtheta, N, N2pi, xi0, dxi, M, s0, ds, Ns, SinoStacks[i],
                    x0, dx, Nx, y0, dy, Ny, z0stack[i], dz, int(StackSizes[i]),
                    z0focousStacks[i], dzfocus, selectedScans
                )
            else:
                empty = Allocation.zeros(Nx * Ny * int(StackSizes[i]))
                VolumeStacks[i] = empty
                HxyzStacks[i] = empty

        return Backprojection._MergeStacks(VolumeStacks, HxyzStacks, StackSizes, Nx, Ny)

    @staticmethod
    def SequenceWFBP3DFullyPixelDrivenParallelizedStacked(Sino, FanBeamGeometryDescriptor_in,
                                                          ParallelBeamGeometryDescriptor_in,
                                                          ReconVolumeGeometry_in):
        return Backprojection._Stacked(
            ReconVolumeGeometry_in,
            lambda: Backprojection.SequenceWFBP3DFullyPixelDrivenParallelized(
                Sino, FanBeamGeometryDescriptor_in,
                ParallelBeamGeometryDescriptor_in, ReconVolumeGeometry_in
            ),
        )

    @staticmethod
    def _Stacked(ReconVolumeGeometry_in, Reconstruct):
        """Retries the reconstruction with more, smaller z-stacks until it fits into memory.

        MATLAB starts with a single stack and increases the stack count on any error, which
        is almost always an out-of-memory condition. The descriptor is restored afterwards.
        """
        Nx = ReconVolumeGeometry_in.Nx
        Ny = ReconVolumeGeometry_in.Ny
        Nz = ReconVolumeGeometry_in.Nz
        z0 = ReconVolumeGeometry_in.z0
        dz = ReconVolumeGeometry_in.dz

        Reconstruction = None
        HxyzVolume = None

        try:
            for Stacks in range(1, Nz + 1):
                FullStacks = _fix(Nz / Stacks)
                LastStacks = Nz - Stacks * FullStacks

                StackSizes = np.full(Stacks, FullStacks, dtype=np.int64)
                StackSizes[:LastStacks] += 1  # reduce tail effect

                try:
                    RecoStacks = [None] * Stacks
                    HxyzStacks = [None] * Stacks

                    for i in range(Stacks):
                        ReconVolumeGeometry_in.z0 = z0 + float(np.sum(StackSizes[:i])) * dz
                        ReconVolumeGeometry_in.Nz = int(StackSizes[i])

                        RecoStacks[i], HxyzStacks[i] = Reconstruct()

                    Reconstruction, HxyzVolume = Backprojection._MergeStacks(
                        RecoStacks, HxyzStacks, StackSizes, Nx, Ny
                    )
                    break
                except MemoryError:
                    continue
        finally:
            ReconVolumeGeometry_in.z0 = z0
            ReconVolumeGeometry_in.Nz = Nz

        if Reconstruction is None:
            raise MemoryError("No z-stack partitioning fitted into memory.")

        return Reconstruction, HxyzVolume

    @staticmethod
    def _StaticCTZLimits(RF, RD, Nx, dx, soffset, Ns, ds):
        """z range a slice may be away from the source to still be illuminated.

        Only valid for a detector shifted towards negative s, as flagged in the MATLAB source.
        """
        RFD = RF + RD

        zlimupper = (RF + Nx * dx / 2) * soffset / RFD
        zlimlower = (RF + Nx / 2 * dx) * (soffset - Ns * ds) / RFD

        return zlimupper, zlimlower

    @staticmethod
    def SequenceWFBP3DFullyPixelDrivenParallelized_StaticCT(Sino, StaticCTGeometryDescriptor_in,
                                                            ParallelBeamGeometryDescriptor_in,
                                                            ReconVolumeGeometry_in):
        Geometry = StaticCTGeometryDescriptor_in.Geometry
        Parameter = StaticCTGeometryDescriptor_in.Parameter

        RF = Geometry.RF
        RD = Geometry.RD
        s0 = Geometry.s0
        ds = Geometry.ds
        Ns = Geometry.Ns
        soffset = Geometry.soffset

        z0focus = Parameter.z0focus
        dzfocus = Parameter.dzfocus
        NzScans = int(Parameter.NzScans)

        theta0 = ParallelBeamGeometryDescriptor_in.theta0
        dtheta = ParallelBeamGeometryDescriptor_in.dtheta
        N = ParallelBeamGeometryDescriptor_in.Npar
        N2pi = ParallelBeamGeometryDescriptor_in.N2pi
        xi0 = ParallelBeamGeometryDescriptor_in.xi0
        dxi = ParallelBeamGeometryDescriptor_in.dxi
        Mpar = ParallelBeamGeometryDescriptor_in.M

        x0 = ReconVolumeGeometry_in.x0
        dx = ReconVolumeGeometry_in.dx
        Nx = ReconVolumeGeometry_in.Nx
        y0 = ReconVolumeGeometry_in.y0
        dy = ReconVolumeGeometry_in.dy
        Ny = ReconVolumeGeometry_in.Ny
        z0 = ReconVolumeGeometry_in.z0
        dz = ReconVolumeGeometry_in.dz
        Nz = ReconVolumeGeometry_in.Nz

        Sino = Backprojection._WeightConeAngle(Sino, N, Ns, Mpar, s0, ds, RF, NzScans=NzScans)

        Reconstruction = Allocation.zeros(Nx * Ny * Nz)
        HxyzVolume = Allocation.zeros(Nx * Ny * Nz)

        zStacksize = 2
        NrzStacks = _fix(Nz / zStacksize)
        zStacksizeLast = Nz - NrzStacks * zStacksize

        zlimupper, zlimlower = Backprojection._StaticCTZLimits(RF, RD, Nx, dx, soffset, Ns, ds)
        delta_zMax = (zlimupper - zlimlower) / 2
        zlimCenter = delta_zMax + zlimlower

        block = Mpar * Ns * N
        slab = Nx * Ny

        def SelectScans(zstartStack, zendStack):
            selected = []
            for nzScans in range(1, NzScans + 1):
                znScanCenter = zlimCenter + z0focus + (nzScans - 1) * dzfocus
                if (abs(znScanCenter - zstartStack) < delta_zMax
                        or abs(znScanCenter - zendStack) < delta_zMax):
                    selected.append(nzScans)
            return selected

        for i in range(1, NrzStacks + 1):
            z0stack = z0 + (i - 1) * zStacksize * dz
            selected = SelectScans(z0stack + dz, z0stack + zStacksize * dz)

            if not selected:
                continue

            SinoContributing = Sino[(selected[0] - 1) * block:selected[-1] * block]

            Volume, Hxyz = Backprojection.SequenceWFBP3DFullyPixelDriven_StaticCT(
                RF, RD, theta0, dtheta, N, N2pi, xi0, dxi, Mpar, s0, ds, Ns, soffset,
                SinoContributing, x0, dx, Nx, y0, dy, Ny, z0stack, dz, zStacksize,
                z0focus + (selected[0] - 1) * dzfocus, dzfocus, range(1, len(selected) + 1)
            )

            start = (i - 1) * slab * zStacksize
            Reconstruction[start:start + slab * zStacksize] = Volume
            HxyzVolume[start:start + slab * zStacksize] = Hxyz

        if zStacksizeLast != 0:
            z0stack = z0 + NrzStacks * zStacksize * dz
            selected = SelectScans(z0stack + dz, z0stack + zStacksizeLast * dz)

            if selected:
                # MATLAB deliberately calls the fan-beam sequence WFBP for the tail stack.
                Volume, Hxyz = Backprojection.SequenceWFBP3DFullyPixelDriven(
                    RF, np.atleast_1d(theta0), dtheta, N, N2pi, xi0, dxi, Mpar, s0, ds, Ns,
                    Sino, x0, dx, Nx, y0, dy, Ny, z0stack, dz, zStacksizeLast,
                    z0focus, dzfocus, selected
                )

                start = NrzStacks * slab * zStacksize
                Reconstruction[start:start + zStacksizeLast * slab] = Volume
                HxyzVolume[start:start + zStacksizeLast * slab] = Hxyz

        return Reconstruction, HxyzVolume

    @staticmethod
    def SpiralWFBP3DFullyPixelDrivenParallelized_StaticCT(Sino, StaticCTGeometryDescriptor_in,
                                                          ParallelBeamGeometryDescriptor_in,
                                                          ReconVolumeGeometry_in):
        Geometry = StaticCTGeometryDescriptor_in.Geometry
        Parameter = StaticCTGeometryDescriptor_in.Parameter

        RF = Geometry.RF
        RD = Geometry.RD
        RFD = RF + RD
        s0 = Geometry.s0
        ds = Geometry.ds
        Ns = Geometry.Ns
        soffset = Geometry.soffset

        z0focus = Parameter.z0focus
        pitch = Parameter.pitch
        alpha0 = Parameter.alpha0

        theta0 = ParallelBeamGeometryDescriptor_in.theta0
        dtheta = ParallelBeamGeometryDescriptor_in.dtheta
        N = ParallelBeamGeometryDescriptor_in.Npar
        N2pi = ParallelBeamGeometryDescriptor_in.N2pi
        xi0 = ParallelBeamGeometryDescriptor_in.xi0
        dxi = ParallelBeamGeometryDescriptor_in.dxi
        Mpar = ParallelBeamGeometryDescriptor_in.M

        x0 = ReconVolumeGeometry_in.x0
        dx = ReconVolumeGeometry_in.dx
        Nx = ReconVolumeGeometry_in.Nx
        y0 = ReconVolumeGeometry_in.y0
        dy = ReconVolumeGeometry_in.dy
        Ny = ReconVolumeGeometry_in.Ny
        z0 = ReconVolumeGeometry_in.z0
        dz = ReconVolumeGeometry_in.dz
        Nz = ReconVolumeGeometry_in.Nz

        # The table feed per rotation is reduced by RF/RFD because the static CT source
        # ring and the detector ring have different radii.
        zrot = pitch * Ns * ds * (RF / RFD)
        z0FocusParallelCenterRay = z0focus + (theta0 - alpha0) * zrot / (2 * np.pi)

        # The cone weighting uses RFD here, which is exact only for the centre beam.
        Sino = Backprojection._WeightConeAngle(Sino, N, Ns, Mpar, s0, ds, RFD)

        Reconstruction = Allocation.zeros(Nx * Ny * Nz)
        HxyzVolume = Allocation.zeros(Nx * Ny * Nz)

        zStacksize = 2
        NrzStacks = _fix(Nz / zStacksize)
        zStacksizeLast = Nz - NrzStacks * zStacksize

        zlimupper, zlimlower = Backprojection._StaticCTZLimits(RF, RD, Nx, dx, soffset, Ns, ds)
        zlimCenter = (zlimupper - zlimlower) / 2 + zlimlower
        Na = (zlimupper - zlimlower) / zrot * N2pi

        zRot = Ns * ds * pitch
        block = Mpar * Ns
        slab = Nx * Ny

        def SelectReadings(zstartStack, zendStack):
            rc_startslice = (zstartStack - z0FocusParallelCenterRay - zlimCenter) / zRot * N2pi
            rc_endslide = (zendStack - z0FocusParallelCenterRay - zlimCenter) / zRot * N2pi

            readingstart = _fix(rc_startslice - Na / 2)
            readingend = _fix(rc_endslide + Na / 2) + 1

            if readingstart >= N or readingend <= 1:
                return None

            return max(readingstart, 1), min(readingend, N)

        for i in range(1, NrzStacks + 1):
            z0stack = z0 + (i - 1) * zStacksize * dz
            readings = SelectReadings(z0stack + dz, z0stack + zStacksize * dz)

            if readings is None:
                continue

            readingstart, readingend = readings

            Volume, Hxyz = Backprojection.SpiralWFBP3DFullyPixelDriven_StaticCT(
                RF, RD, theta0 + (readingstart - 1) * dtheta, dtheta,
                readingend - readingstart + 1, N2pi, xi0, dxi, Mpar, s0, ds, Ns, soffset,
                Sino[(readingstart - 1) * block:readingend * block],
                x0, dx, Nx, y0, dy, Ny, z0stack, dz, zStacksize, z0focus, pitch, alpha0
            )

            start = (i - 1) * slab * zStacksize
            Reconstruction[start:start + slab * zStacksize] = Volume
            HxyzVolume[start:start + slab * zStacksize] = Hxyz

        if zStacksizeLast != 0:
            z0stack = z0 + NrzStacks * zStacksize * dz
            # MATLAB sizes the tail window with zStacksize, not zStacksizeLast.
            readings = SelectReadings(z0stack + dz, z0stack + zStacksize * dz)

            if readings is not None:
                readingstart, readingend = readings

                Volume, Hxyz = Backprojection.SpiralWFBP3DFullyPixelDriven_StaticCT(
                    RF, RD, theta0 + (readingstart - 1) * dtheta, dtheta,
                    readingend - readingstart + 1, N2pi, xi0, dxi, Mpar, s0, ds, Ns, soffset,
                    Sino[(readingstart - 1) * block:readingend * block],
                    x0, dx, Nx, y0, dy, Ny, z0stack, dz, zStacksizeLast, z0focus, pitch, alpha0
                )

                start = NrzStacks * slab * zStacksize
                Reconstruction[start:start + zStacksizeLast * slab] = Volume
                HxyzVolume[start:start + zStacksizeLast * slab] = Hxyz

        return Reconstruction, HxyzVolume
