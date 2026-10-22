import numba
import numpy as np

from CTBasics.Memory.Allocation import Allocation
from CTBasics.MultiCore.Parallelization import Parallelization
from CTBasics.MultiCore.Partition.SequenceStacks import SequenceStacks
from CTBasics.MultiCore.Partition.SpiralStacks import SpiralStacks


def _fix(value):
    """MATLAB fix(): truncate toward zero."""
    return int(value)


def _fix_array(values):
    """Vectorized MATLAB fix() returning integer indices."""
    return np.trunc(values).astype(np.int64)


# The numba kernels below give every image row its own thread and add the views in their
# serial order, so the result does not depend on the number of threads.

@numba.njit(parallel=True, cache=True)
def _BackprojectParallelViews(a, b, c, norm, Sino, M, Nx, Ny, Reconstruction):
    """Adds view n / norm[n] of every parallel view to Reconstruction.

    View n samples the channel mreal = a[n]*nx + b[n]*ny + c[n] of Sino[n*M:(n+1)*M].
    """
    for j in numba.prange(Ny):
        ny = j + 1
        row = j * Nx
        for n in range(a.size):
            base = n * M
            for nx in range(1, Nx + 1):
                mreal = a[n] * nx + b[n] * ny + c[n]
                m = int(mreal)
                if 1 <= m < M:
                    w = mreal - m
                    Reconstruction[row + nx - 1] += (
                        (1 - w) * Sino[base + m - 1] + w * Sino[base + m]) / norm[n]


@numba.njit(parallel=True, cache=True)
def _WFBP2DViews(a, b, c, Npi, Sino, M, Nx, Ny, Reconstruction):
    """Adds the per-voxel average of every half-turn group nthetaBar, nthetaBar + Npi, ..."""
    for j in numba.prange(Ny):
        ny = j + 1
        row = j * Nx
        for nthetaBar in range(Npi):
            for nx in range(1, Nx + 1):
                Vxy = 0.0
                Hxy = 0.0
                for n in range(nthetaBar, a.size, Npi):
                    mreal = a[n] * nx + b[n] * ny + c[n]
                    m = int(mreal)
                    if 1 <= m < M:
                        w = mreal - m
                        Vxy += (1 - w) * Sino[n * M + m - 1] + w * Sino[n * M + m]
                        Hxy += 1.0
                if Hxy > 0:
                    Reconstruction[row + nx - 1] += Vxy / Hxy


def _ParallelViewCoefficients(theta, x0, dx, y0, dy, xi0, dxi):
    """(a, b, c) of mreal = a*nx + b*ny + c, where xi = x*cos(theta) + y*sin(theta)."""
    cc = np.cos(theta)
    ss = np.sin(theta)
    a = dx * cc / dxi
    b = dy * ss / dxi
    c = (x0 * cc + y0 * ss - xi0) / dxi
    return a, b, c


@numba.njit(parallel=True, cache=True, error_model="numpy")
def _WFBP3DVoxels(theta, cc, ss, a, b, c, zn, Spiral, StaticCT, z0focus, alpha0, zRot_2pi,
                  RF, RD, A, Npi, xi0, dxi, M, Ns, ds, Sino,
                  x0, dx, Nx, y0, dy, Ny, z0, dz, Nz, Reconstruction, HxyzVolume):
    """Weighted filtered backprojection of the views theta[scan, n] into every voxel.

    Loops like the MATLAB original: per voxel the half-turn groups nthetaBar, and within a
    group all scans and then the views n = nthetaBar, nthetaBar + Npi, ... Voxels that
    miss a group are incomplete and end up 0. The focus height is zn[scan] for sequence
    scans and follows the source angle for spirals; StaticCT selects the ring geometry.
    """
    S, N = theta.shape
    RFsquare = RF * RF
    M_Ns = M * Ns
    M_Ns_N = M_Ns * N
    RFsquarePlusRDSquare = RFsquare + RD * RD
    TwoRDRF = 2 * RD * RF
    RFbyRD = RF / RD
    one_minus_RFbyRD_square = 1 - RFbyRD * RFbyRD
    pi_half = np.pi / 2
    for j in numba.prange(Nz * Ny):
        z = z0 + (j // Ny + 1) * dz
        z_RF = z * RF
        ny = j % Ny + 1
        y = y0 + ny * dy
        for nx in range(1, Nx + 1):
            x = x0 + nx * dx
            total = 0.0
            hits = 0.0
            complete = True
            for nthetaBar in range(Npi):
                Hxyz = 0
                Vxyz = 0.0
                for scan in range(S):
                    for n in range(nthetaBar, N, Npi):
                        mreal = a[scan, n] * nx + b[scan, n] * ny + c[scan, n]
                        # 1 <= fix(v) < M is the same as 1 <= v < M and also rejects NaN.
                        if not 1 <= mreal < M:
                            continue
                        xi = xi0 + mreal * dxi
                        y_cc = y * cc[scan, n]
                        if StaticCT:
                            beta_real = -np.arcsin(xi / RF)
                            zs = zn[scan]
                            if Spiral:
                                alpha = theta[scan, n] - beta_real
                                zs = z0focus + (alpha - alpha0) * zRot_2pi
                            tan_beta_real_square = np.tan(beta_real) ** 2
                            argument = ((-tan_beta_real_square * RFbyRD
                                         + np.sqrt(tan_beta_real_square
                                                   * one_minus_RFbyRD_square + 1))
                                        / (tan_beta_real_square + 1))
                            betaDash_real = np.sign(beta_real) * np.arccos(
                                min(max(argument, -1.0), 1.0))
                            L = np.sqrt(RFsquarePlusRDSquare
                                        + TwoRDRF * np.sin(betaDash_real + pi_half))
                            l = np.sqrt((RFsquare - xi * xi) - x * ss[scan, n] + y_cc)
                            nsreal = (z - zs) * L / (l * ds) + A
                        else:
                            zs = zn[scan]
                            if Spiral:
                                alpha = theta[scan, n] + np.arcsin(xi / RF)
                                zs = z0focus + (alpha - alpha0) * zRot_2pi
                            nsreal = ((z_RF - zs * RF)
                                      / ((np.sqrt(RFsquare - xi * xi) - x * ss[scan, n] + y_cc)
                                         * ds) + A)
                        if not 1 <= nsreal < Ns:
                            continue
                        m = int(mreal)
                        ns = int(nsreal)
                        wm = mreal - m
                        wns = nsreal - ns
                        k = scan * M_Ns_N + n * M_Ns + (ns - 1) * M + m - 1
                        Reproj_ns = (1 - wm) * Sino[k] + wm * Sino[k + 1]
                        Reproj_ns_plus1 = (1 - wm) * Sino[k + M] + wm * Sino[k + M + 1]
                        Vxyz = Vxyz + (1 - wns) * Reproj_ns + wns * Reproj_ns_plus1
                        Hxyz += 1
                if Hxyz > 0:
                    total += Vxyz / Hxyz
                else:
                    complete = False
                hits += Hxyz
            index = j * Nx + nx - 1
            Reconstruction[index] = total if complete else 0.0
            HxyzVolume[index] = hits


def _WFBP3D(theta, zn, A, N2pi, dtheta, RF, xi0, dxi, M, Ns, ds, Sino,
            x0, dx, Nx, y0, dy, Ny, z0, dz, Nz,
            Spiral=False, z0focus=0.0, alpha0=0.0, zRot_2pi=0.0, StaticCT=False, RD=1.0):
    """(Reconstruction, HxyzVolume) of the views theta with shape (scans, N)."""
    theta = np.ascontiguousarray(theta, dtype=float)
    a, b, c = _ParallelViewCoefficients(theta, x0, dx, y0, dy, xi0, dxi)

    Reconstruction = np.zeros(Nx * Ny * Nz)
    HxyzVolume = np.zeros(Nx * Ny * Nz)
    _WFBP3DVoxels(theta, np.cos(theta), np.sin(theta), a, b, c,
                  np.ascontiguousarray(zn, dtype=float), Spiral, StaticCT,
                  float(z0focus), float(alpha0), float(zRot_2pi), float(RF), float(RD),
                  float(A), int(N2pi // 2), float(xi0), float(dxi), int(M), int(Ns),
                  float(ds), _Sinogram(Sino), float(x0), float(dx), int(Nx),
                  float(y0), float(dy), int(Ny), float(z0), float(dz), int(Nz),
                  Reconstruction, HxyzVolume)

    return Reconstruction * dtheta, HxyzVolume


def _EachStack(func, stacks, **shared):
    """The MATLAB parfor over the z-stacks; the threaded kernel parallelizes each stack."""
    return [func(**stack, **shared) for stack in stacks]


def _Views(theta0, dtheta, N):
    return theta0 + np.arange(1, N + 1) * dtheta


def _Sinogram(Sino):
    return np.ascontiguousarray(Sino, dtype=float).ravel()


def _GatherSino3D(Sino, base, M, wm, wns, valid):
    """Bilinear sample of one sinogram in the channel and the detector row direction."""
    safe = np.where(valid, base, 0)

    Reproj_ns = (1 - wm) * Sino[safe] + wm * Sino[safe + 1]
    Reproj_ns_plus1 = (1 - wm) * Sino[safe + M] + wm * Sino[safe + M + 1]

    return np.where(valid, (1 - wns) * Reproj_ns + wns * Reproj_ns_plus1, 0.0)


def _HalfTurnGroups(N, Npi):
    """MATLAB loops nthetaBar = 1..Npi and then n = nthetaBar:Npi:N."""
    return [list(range(nthetaBar, N + 1, Npi)) for nthetaBar in range(1, Npi + 1)]


def _PixDriven3DGroups(groups, Sino, block, theta0, dtheta, RF, xi0, dxi, M, Ns, ds,
                       x0, dx, Nx, y0, dy, Ny, z0, dz, Nz):
    """parfor body: sum of the half-turn averaged backprojections of some groups."""
    Reconstruction = Allocation.zeros(Nx * Ny * Nz)
    for group in groups:
        Partial = Allocation.zeros(Nx * Ny * Nz)
        for n in group:
            Partial += Backprojection.PixDriven3DBackProjParCartesianSingle(
                theta0 + n * dtheta, RF, xi0, dxi, M, Ns, ds,
                Sino[(n - 1) * block:n * block], x0, dx, Nx, y0, dy, Ny, z0, dz, Nz
            )
        Reconstruction += Partial / len(group)
    return Reconstruction


class Backprojection:
    """Pixel-driven backprojectors.

    The |dtheta| factor from the inverse Radon transform is applied once at the end,
    matching the MATLAB original.
    """

    @staticmethod
    def PixDrivenBackProjParCartesianSingleBase(M, Nx, Ny, a, b, c, Proj):
        Reprojection = Allocation.zeros(Nx * Ny)
        _BackprojectParallelViews(np.array([a], dtype=float), np.array([b], dtype=float),
                                  np.array([c], dtype=float), np.ones(1), _Sinogram(Proj),
                                  int(M), int(Nx), int(Ny), Reprojection)
        return Reprojection

    @staticmethod
    def PixDrivenBackProjParCartesianSingle(theta, xi0, dxi, M, Proj, x0, dx, Nx, y0, dy, Ny):
        a, b, c = _ParallelViewCoefficients(theta, x0, dx, y0, dy, xi0, dxi)
        return Backprojection.PixDrivenBackProjParCartesianSingleBase(M, Nx, Ny, a, b, c, Proj)

    @staticmethod
    def PixDrivenBackProjParCartesian(theta0, dtheta, N, N2pi, xi0, dxi, M, Sino, x0, dx, Nx, y0, dy, Ny):
        # Rays a half turn apart are redundant, so each half-turn group shares one norm.
        # MATLAB buffers all N backprojections first; normalizing on the fly is equivalent
        # and avoids an (N, Nx*Ny) array.
        Npi = int(N2pi // 2)
        Norm = np.array([len(range(nbar, N + 1, Npi)) for nbar in range(1, Npi + 1)], dtype=float)

        a, b, c = _ParallelViewCoefficients(_Views(theta0, dtheta, N), x0, dx, y0, dy, xi0, dxi)

        Reconstruction = Allocation.zeros(Nx * Ny)
        _BackprojectParallelViews(a, b, c, Norm[np.arange(N) % Npi], _Sinogram(Sino),
                                  int(M), int(Nx), int(Ny), Reconstruction)
        return Reconstruction * dtheta

    @staticmethod
    def WFBP2DFullyPixelDriven(theta0, dtheta, N, N2pi, xi0, dxi, M, Sino, x0, dx, Nx, y0, dy, Ny):
        """Weighted FBP that stays correct beyond a pi range by normalizing per half turn.

        Unlike PixDrivenBackProjParCartesian the weighting is per voxel: each voxel counts
        only the projections that actually hit the detector, so truncated views do not bias it.
        """
        a, b, c = _ParallelViewCoefficients(_Views(theta0, dtheta, N), x0, dx, y0, dy, xi0, dxi)

        Reconstruction = Allocation.zeros(Nx * Ny)
        _WFBP2DViews(a, b, c, int(N2pi // 2), _Sinogram(Sino), int(M), int(Nx), int(Ny),
                     Reconstruction)
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

        # 2. Backproject each view, 3. average over the redundant half turns. Each worker
        # sums the groups of its chunk; without a pool there is one chunk, i.e. serial order.
        Npi = int(N2pi // 2)

        Partials = Parallelization.ParFor(
            _PixDriven3DGroups,
            [{"groups": chunk} for chunk in Parallelization.Chunks(_HalfTurnGroups(N, Npi))],
            Sino=Sino, block=M * Ns, theta0=theta0, dtheta=dtheta, RF=RF, xi0=xi0, dxi=dxi,
            M=M, Ns=Ns, ds=ds, x0=x0, dx=dx, Nx=Nx, y0=y0, dy=dy, Ny=Ny, z0=z0, dz=dz, Nz=Nz
        )

        Reconstruction = Allocation.zeros(Nx * Ny * Nz)
        for Partial in Partials:
            Reconstruction += Partial

        return Reconstruction * dtheta

    # ------------------------------------------------------------- 3D WFBP

    @staticmethod
    def SpiralWFBP3DFullyPixelDriven(RF, theta0, dtheta, N, N2pi, xi0, dxi, M, s0, ds, Ns,
                                     Sino, x0, dx, Nx, y0, dy, Ny, z0, dz, Nz,
                                     z0focus, pitch, alpha0):
        theta = theta0 + np.arange(1, N + 1)[None, :] * dtheta

        # The focus height follows the source angle alpha, not the parallel angle theta.
        return _WFBP3D(theta, [0.0], Ns / 2.0 + 0.5, N2pi, dtheta, RF, xi0, dxi, M, Ns, ds,
                       Sino, x0, dx, Nx, y0, dy, Ny, z0, dz, Nz, Spiral=True,
                       z0focus=z0focus, alpha0=alpha0, zRot_2pi=pitch * Ns * ds / (2 * np.pi))

    @staticmethod
    def SequenceWFBP3DFullyPixelDriven(RF, theta0, dtheta, N, N2pi, xi0, dxi, M, s0, ds, Ns,
                                       Sino, x0, dx, Nx, y0, dy, Ny, z0, dz, Nz,
                                       z0focus, dzfocus, NzScans):
        theta0 = np.atleast_1d(np.asarray(theta0, dtype=float))
        ScanIdx = np.atleast_1d(np.asarray(NzScans, dtype=np.int64))

        # Every sequence scan position contributes to the same half-turn group.
        theta = theta0[ScanIdx - 1][:, None] + np.arange(1, N + 1)[None, :] * dtheta
        zn = z0focus + np.arange(ScanIdx.size) * dzfocus

        return _WFBP3D(theta, zn, Ns / 2.0 + 0.5, N2pi, dtheta, RF, xi0, dxi, M, Ns, ds,
                       Sino, x0, dx, Nx, y0, dy, Ny, z0, dz, Nz)

    # --------------------------------------------------------- 3D WFBP static CT

    @staticmethod
    def SequenceWFBP3DFullyPixelDriven_StaticCT(RF, RD, theta0, dtheta, N, N2pi, xi0, dxi,
                                                Mpar, s0, ds, Ns, soffset, Sino,
                                                x0, dx, Nx, y0, dy, Ny, z0, dz, Nz,
                                                z0focus, dzfocus, NzScans):
        NzScans = len(np.atleast_1d(np.asarray(NzScans)))
        theta = np.broadcast_to(theta0 + np.arange(1, N + 1) * dtheta, (NzScans, N))
        zn = z0focus + np.arange(NzScans) * dzfocus

        # The static CT detector is shifted by soffset instead of being centred on zero.
        return _WFBP3D(theta, zn, -soffset / ds + Ns + 0.5, N2pi, dtheta, RF, xi0, dxi, Mpar,
                       Ns, ds, Sino, x0, dx, Nx, y0, dy, Ny, z0, dz, Nz, StaticCT=True, RD=RD)

    @staticmethod
    def SpiralWFBP3DFullyPixelDriven_StaticCT(RF, RD, theta0, dtheta, N, N2pi, xi0, dxi,
                                              Mpar, s0, ds, Ns, soffset, Sino,
                                              x0, dx, Nx, y0, dy, Ny, z0, dz, Nz,
                                              z0focus, pitch, alpha0):
        theta = theta0 + np.arange(1, N + 1)[None, :] * dtheta

        # MATLAB marks zRot_2pi as still to be adapted to the static CT geometry.
        return _WFBP3D(theta, [0.0], -soffset / ds + Ns + 0.5, N2pi, dtheta, RF, xi0, dxi,
                       Mpar, Ns, ds, Sino, x0, dx, Nx, y0, dy, Ny, z0, dz, Nz, Spiral=True,
                       z0focus=z0focus, alpha0=alpha0, zRot_2pi=pitch * Ns * ds / (2 * np.pi),
                       StaticCT=True, RD=RD)

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

        Items = [i for i in range(TotalItems)
                 if FirstReadingStack[i] < N and FinalReadingStack[i] > 1]

        Results = _EachStack(
            Backprojection.SpiralWFBP3DFullyPixelDriven,
            [{"theta0": thetaStacks[i], "N": int(NStacks[i]), "Sino": SinoStacks[i],
              "z0": z0Stack[i], "Nz": int(StackSizes[i])} for i in Items],
            RF=RF, dtheta=dtheta, N2pi=N2pi, xi0=xi0, dxi=dxi, M=M, s0=s0, ds=ds, Ns=Ns,
            x0=x0, dx=dx, Nx=Nx, y0=y0, dy=dy, Ny=Ny, dz=dz,
            z0focus=z0focus, pitch=pitch, alpha0=alpha0
        )

        for i, (Volume, Hxyz) in zip(Items, Results):
            VolumeStacks[i] = Volume
            HxyzStacks[i] = Hxyz

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

        Items = [i for i in range(TotalItems) if Scans[i] is not None and len(Scans[i]) > 0]

        Results = _EachStack(
            Backprojection.SequenceWFBP3DFullyPixelDriven,
            [{"Sino": SinoStacks[i], "z0": z0stack[i], "Nz": int(StackSizes[i]),
              "z0focus": z0focousStacks[i], "NzScans": Scans[i]} for i in Items],
            RF=RF, theta0=theta0, dtheta=dtheta, N=N, N2pi=N2pi, xi0=xi0, dxi=dxi, M=M,
            s0=s0, ds=ds, Ns=Ns, x0=x0, dx=dx, Nx=Nx, y0=y0, dy=dy, Ny=Ny, dz=dz,
            dzfocus=dzfocus
        )

        for i in range(TotalItems):
            if i not in Items:
                VolumeStacks[i] = HxyzStacks[i] = Allocation.zeros(Nx * Ny * int(StackSizes[i]))

        for i, (Volume, Hxyz) in zip(Items, Results):
            VolumeStacks[i] = Volume
            HxyzStacks[i] = Hxyz

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

        Stacks = []
        for i in range(1, NrzStacks + 1):
            z0stack = z0 + (i - 1) * zStacksize * dz
            selected = SelectScans(z0stack + dz, z0stack + zStacksize * dz)

            if selected:
                Stacks.append((i, {
                    "Sino": Sino[(selected[0] - 1) * block:selected[-1] * block],
                    "z0": z0stack,
                    "z0focus": z0focus + (selected[0] - 1) * dzfocus,
                    "NzScans": range(1, len(selected) + 1),
                }))

        Results = _EachStack(
            Backprojection.SequenceWFBP3DFullyPixelDriven_StaticCT,
            [sliced for _, sliced in Stacks],
            RF=RF, RD=RD, theta0=theta0, dtheta=dtheta, N=N, N2pi=N2pi, xi0=xi0, dxi=dxi,
            Mpar=Mpar, s0=s0, ds=ds, Ns=Ns, soffset=soffset,
            x0=x0, dx=dx, Nx=Nx, y0=y0, dy=dy, Ny=Ny, dz=dz, Nz=zStacksize, dzfocus=dzfocus
        )

        for (i, _), (Volume, Hxyz) in zip(Stacks, Results):
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

        Stacks = []
        for i in range(1, NrzStacks + 1):
            z0stack = z0 + (i - 1) * zStacksize * dz
            readings = SelectReadings(z0stack + dz, z0stack + zStacksize * dz)

            if readings is not None:
                readingstart, readingend = readings
                Stacks.append((i, {
                    "theta0": theta0 + (readingstart - 1) * dtheta,
                    "N": readingend - readingstart + 1,
                    "Sino": Sino[(readingstart - 1) * block:readingend * block],
                    "z0": z0stack,
                }))

        Results = _EachStack(
            Backprojection.SpiralWFBP3DFullyPixelDriven_StaticCT,
            [sliced for _, sliced in Stacks],
            RF=RF, RD=RD, dtheta=dtheta, N2pi=N2pi, xi0=xi0, dxi=dxi, Mpar=Mpar,
            s0=s0, ds=ds, Ns=Ns, soffset=soffset, x0=x0, dx=dx, Nx=Nx, y0=y0, dy=dy, Ny=Ny,
            dz=dz, Nz=zStacksize, z0focus=z0focus, pitch=pitch, alpha0=alpha0
        )

        for (i, _), (Volume, Hxyz) in zip(Stacks, Results):
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
