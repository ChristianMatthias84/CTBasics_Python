from pathlib import Path

import numba
import numpy as np

from CTBasics.Descriptors.ScannerGeometryTypes import ScannerGeometryTypes
from CTBasics.Descriptors.ScanTypes import ScanTypes
from CTBasics.Memory.Allocation import Allocation
from CTBasics.Memory.Precision import Precision
from CTBasics.ReadAndWrite.IOHelperFunctions import IOHelperFunctions


def _Regroup(Projections, M, Ns, N):
    """MATLAB reshape(P, M, Ns*N)' -- reading order (n, ns, m) becomes rows of length M."""
    return np.asarray(Projections, dtype=float).reshape(Ns * N, M)


def _Linearize(RebinnedProjections, Ns, Npar, Mpar):
    """Interleaves the per-row results back into the (n, ns, m) sinogram order."""
    return RebinnedProjections.reshape(Ns, Npar, Mpar).transpose(1, 0, 2).ravel()


@numba.njit(parallel=True, cache=True)
def _AzimuthalRebinning(p_alpha_beta, N, theta, beta, alpha0, dalpha):
    """p(alpha, beta) -> p(theta, beta); views outside the scan stay 0."""
    MBeta = p_alpha_beta.shape[1]
    p_theta_beta = np.zeros((theta.size, MBeta))

    for n in numba.prange(theta.size):
        for m in range(MBeta):
            n_alpha_real = (-beta[m] + theta[n] - alpha0) / dalpha
            n_alpha = np.floor(n_alpha_real)
            if 1 <= n_alpha < N:
                w = n_alpha_real - n_alpha
                k = int(n_alpha)
                p_theta_beta[n, m] = (1 - w) * p_alpha_beta[k - 1, m] + w * p_alpha_beta[k, m]

    return p_theta_beta


@numba.njit(parallel=True, cache=True)
def _RadialRebinning(p_theta_beta, m_beta, w, valid):
    """p(theta, beta) -> p(theta, xi) with per-channel indices and weights; invalid channels stay 0."""
    NPar = p_theta_beta.shape[0]
    p_theta_xi = np.zeros((NPar, m_beta.size))

    for n in numba.prange(NPar):
        for m in range(m_beta.size):
            if valid[m]:
                k = m_beta[m]
                p_theta_xi[n, m] = (1 - w[m]) * p_theta_beta[n, k - 1] + w[m] * p_theta_beta[n, k]

    return p_theta_xi


@numba.njit(parallel=True, cache=True)
def _RingResampling(p_alpha_delta, N, alpha, betaDash, delta0, dDelta):
    """Resamples the closed source ring onto the betaDash raster; the delta index wraps around."""
    MDelta = p_alpha_delta.shape[1]
    p_alpha_deltaDash = np.empty((N, betaDash.size))

    for n in numba.prange(N):
        for m in range(betaDash.size):
            m_delta_real = (alpha[n] + np.pi / 2 + betaDash[m] - delta0) / dDelta
            m_delta_floor = np.floor(m_delta_real)
            w = m_delta_real - m_delta_floor

            m_delta = int(m_delta_floor)
            if m_delta <= 0:
                m_delta = MDelta - abs(m_delta) % MDelta
            else:
                m_delta = (m_delta - 1) % MDelta + 1
            m_delta_plus1 = (m_delta + 1) % MDelta
            if m_delta_plus1 == 0:
                m_delta_plus1 = MDelta

            p_alpha_deltaDash[n, m] = ((1 - w) * p_alpha_delta[n, m_delta - 1]
                                       + w * p_alpha_delta[n, m_delta_plus1 - 1])

    return p_alpha_deltaDash


def _CachedFilename(ScanDescriptor, suffix):
    """Builds the cache filename; MATLAB names the bit count 'bytes'."""
    file = Path(ScanDescriptor.Filename)

    if file.suffix != ".raw":
        raise ValueError("Invalid File-Extension")

    bits = {"double": 64, "single": 32}[Precision.Value()]
    FFS = ScanDescriptor.Geometry.FFS.ToString()

    return f"{file.stem}_{FFS}_{bits}_bytes_{suffix}.bin"


class Rebinning:
    # ==============================
    # Caching Mechanism
    # ==============================
    @staticmethod
    def Apply(Projections, ScanDescriptor, ParallelGeometry):
        RebinnedFilename = Rebinning.RebinnedFilename(ScanDescriptor)

        try:
            result = IOHelperFunctions.Load(RebinnedFilename)
            print(f"Using cached rebinned data [{RebinnedFilename}]")
            return result
        except OSError:
            pass

        BeamType = ScanDescriptor.Geometry.BeamType
        ScanType = ScanDescriptor.Parameter.ScanType

        if BeamType == ScannerGeometryTypes.Fan:
            Rebin = {ScanTypes.Spiral: Rebinning.RebinningFanToParallelSpiral,
                     ScanTypes.Sequence: Rebinning.RebinningFanToParallelSequence}
        elif BeamType == ScannerGeometryTypes.StaticCT:
            Rebin = {ScanTypes.Spiral: Rebinning.RebinningStaticCTToParallelSpiral,
                     ScanTypes.Sequence: Rebinning.RebinningStaticCTToParallelSequence}
        else:
            raise ValueError("Unknown BeamType!")

        if ScanType not in Rebin:
            raise ValueError("Unknown ScanType!")

        result = Rebin[ScanType](Projections, ScanDescriptor, ParallelGeometry)

        IOHelperFunctions.Save(result, RebinnedFilename)
        return result

    # ==============================
    # Fan Beam Geometry
    # ==============================
    @staticmethod
    def RebinningFanToParallel(ProjectiondataFanBeam, RF, MFan, NFan, alpha0, dalpha, beta, MPar, NPar, theta0, dtheta, xi0, dxi):
        p_alpha_beta = np.asarray(ProjectiondataFanBeam, dtype=float).reshape(-1, MFan)
        beta = np.array(beta, dtype=float, copy=True)

        # Azimuthal rebinning: p(alpha, beta) -> p(theta, beta)
        theta = theta0 + np.arange(1, NPar + 1) * dtheta
        p_theta_beta = Allocation.zeros(NPar, MFan)
        p_theta_beta[:] = _AzimuthalRebinning(p_alpha_beta, NFan, theta, beta, alpha0, dalpha)

        # Radial rebinning: p(theta, beta) -> p(theta, xi)
        xi = xi0 + np.arange(1, MPar + 1) * dxi
        beta_real = -np.arcsin(xi / RF)

        IsAscending = beta[0] < beta[-1]
        if IsAscending:
            beta = -beta

        # MATLAB takes the largest index whose beta still exceeds beta_real.
        above = beta[None, :] > beta_real[:, None]
        m_beta = MFan - np.argmax(above[:, ::-1], axis=1)

        valid = above.any(axis=1) & (m_beta >= 1) & (m_beta < MFan)
        safe = np.where(valid, m_beta, 1)

        with np.errstate(divide="ignore", invalid="ignore"):
            w = (beta[safe - 1] - beta_real) / (beta[safe - 1] - beta[safe])

        p_theta_xi = Allocation.zeros(NPar * MPar)
        p_theta_xi[:] = _RadialRebinning(p_theta_beta, safe, w, valid).ravel()

        return p_theta_xi

    @staticmethod
    def RebinningFanToParallelSequence(ProjectionsSequence, FanBeamGeometry_in, ParallelBeamGeometry_in):
        Geometry = FanBeamGeometry_in.Geometry
        Parameter = FanBeamGeometry_in.Parameter

        RF = Geometry.RF
        MFan = Geometry.MFan
        beta = Geometry.beta
        Ns = Geometry.Ns
        dalpha = Geometry.Sampling.dalpha

        NFan = Parameter.N
        alpha0 = np.atleast_1d(np.asarray(Parameter.alpha0, dtype=float))
        NzScans = int(Parameter.NzScans)

        Mpar = ParallelBeamGeometry_in.M
        Npar = ParallelBeamGeometry_in.Npar
        theta0 = np.atleast_1d(np.asarray(ParallelBeamGeometry_in.theta0, dtype=float))
        dtheta = ParallelBeamGeometry_in.dtheta
        xi0 = ParallelBeamGeometry_in.xi0
        dxi = ParallelBeamGeometry_in.dxi

        ProjectionsSequence = np.asarray(ProjectionsSequence, dtype=float).ravel()
        ProjectiondataParallelBeam = Allocation.zeros(Mpar * Ns * Npar * NzScans)

        for nz in range(1, NzScans + 1):
            Projections = ProjectionsSequence[(nz - 1) * MFan * Ns * NFan:nz * MFan * Ns * NFan]
            RegroupedProjections = _Regroup(Projections, MFan, Ns, NFan)

            RebinnedProjections = Allocation.zeros(Ns, Mpar * Npar)
            for ns in range(1, Ns + 1):
                subgroup = RegroupedProjections[ns - 1::Ns, :].ravel()
                RebinnedProjections[ns - 1, :] = Rebinning.RebinningFanToParallel(
                    subgroup, RF, MFan, NFan, alpha0[nz - 1], dalpha, beta,
                    Mpar, Npar, theta0[nz - 1], dtheta, xi0, dxi
                )

            block = Mpar * Ns * Npar
            ProjectiondataParallelBeam[(nz - 1) * block:nz * block] = _Linearize(
                RebinnedProjections, Ns, Npar, Mpar
            )

        return ProjectiondataParallelBeam

    @staticmethod
    def RebinningFanToParallelSpiral(Projections, FanBeamGeometry_in, ParallelBeamGeometry_in):
        Geometry = FanBeamGeometry_in.Geometry
        Parameter = FanBeamGeometry_in.Parameter

        RF = Geometry.RF
        MFan = Geometry.MFan
        beta = Geometry.beta
        Ns = Geometry.Ns
        dalpha = Geometry.Sampling.dalpha

        NFan = Parameter.N
        alpha0 = Parameter.alpha0

        Mpar = ParallelBeamGeometry_in.M
        Npar = ParallelBeamGeometry_in.Npar
        theta0 = ParallelBeamGeometry_in.theta0
        dtheta = ParallelBeamGeometry_in.dtheta
        xi0 = ParallelBeamGeometry_in.xi0
        dxi = ParallelBeamGeometry_in.dxi

        RegroupedProjections = _Regroup(Projections, MFan, Ns, NFan)

        RebinnedProjections = Allocation.zeros(Ns, Mpar * Npar)
        for ns in range(1, Ns + 1):
            subgroup = RegroupedProjections[ns - 1::Ns, :].ravel()
            RebinnedProjections[ns - 1, :] = Rebinning.RebinningFanToParallel(
                subgroup, RF, MFan, NFan, alpha0, dalpha, beta,
                Mpar, Npar, theta0, dtheta, xi0, dxi
            )

        return _Linearize(RebinnedProjections, Ns, Npar, Mpar)

    # ==============================
    # Static CT Geometry
    # ==============================
    @staticmethod
    def RebinningStaticCTToParallel(ProjectiondataStaticCT, RF, RD, MDelta, N, alpha0, dalpha,
                                    delta0, dDelta, MPar, NPar, theta0, dtheta, xi0, dxi,
                                    dbetaDash, betaDash0, MbetaDash):
        p_alpha_delta = np.asarray(ProjectiondataStaticCT, dtype=float).reshape(-1, MDelta)

        RFbyRD = RF / RD
        one_minus_RFbyRD_square = 1 - RFbyRD * RFbyRD

        # 1. Resample the closed source ring onto an equidistant betaDash raster. The ring
        #    wraps around, so the delta index is taken modulo MDelta instead of being clipped.
        alpha = alpha0 + np.arange(1, N + 1) * dalpha
        betaDash = betaDash0 + np.arange(1, MbetaDash + 1) * dbetaDash

        p_alpha_deltaDash = _RingResampling(p_alpha_delta, N, alpha, betaDash, delta0, dDelta)

        # 2. Azimuthal rebinning: p(alpha, betaDash) -> p(theta, betaDash)
        theta = theta0 + np.arange(1, NPar + 1) * dtheta
        beta = np.arctan(np.sin(betaDash) * RD / (np.cos(betaDash) * RD + RF))

        p_theta_beta = _AzimuthalRebinning(p_alpha_deltaDash, N, theta, beta, alpha0, dalpha)

        # 3. Radial rebinning: p(theta, betaDash) -> p(theta, xi)
        xi = xi0 + np.arange(1, MPar + 1) * dxi
        beta_real = -np.arcsin(xi / RF)

        tan_beta_real_square = np.tan(beta_real) ** 2
        betaDash_real = np.sign(beta_real) * np.arccos(
            (-tan_beta_real_square * RFbyRD
             + np.sqrt(tan_beta_real_square * one_minus_RFbyRD_square + 1))
            / (tan_beta_real_square + 1)
        )

        m_betaDash_real = (betaDash_real - betaDash0) / dbetaDash
        m_betaDash = np.floor(m_betaDash_real).astype(np.int64)
        w = m_betaDash_real - m_betaDash

        valid = (m_betaDash >= 1) & (m_betaDash < MbetaDash)

        p_theta_xi = Allocation.zeros(NPar * MPar)
        p_theta_xi[:] = _RadialRebinning(p_theta_beta, m_betaDash, w, valid).ravel()

        return p_theta_xi

    @staticmethod
    def RebinningStaticCTToParallelSequence(ProjectiondataStaticCT, StaticCTGeometry_in, ParallelBeamGeometry_in):
        Geometry = StaticCTGeometry_in.Geometry
        Parameter = StaticCTGeometry_in.Parameter

        RF = Geometry.RF
        RD = Geometry.RD
        MDelta = Geometry.MDelta
        delta0 = Geometry.delta0
        dDelta = Geometry.dDelta
        dbetaDash = Geometry.dbetaDash
        betaDash0 = Geometry.betaDash0
        MbetaDash = Geometry.MbetaDash
        Ns = Geometry.Ns
        dalpha = Geometry.Sampling.dalpha

        N = Parameter.N
        alpha0 = np.atleast_1d(np.asarray(Parameter.alpha0, dtype=float))
        NzScans = int(Parameter.NzScans)

        Mpar = ParallelBeamGeometry_in.M
        Npar = ParallelBeamGeometry_in.Npar
        theta0 = np.atleast_1d(np.asarray(ParallelBeamGeometry_in.theta0, dtype=float))
        dtheta = ParallelBeamGeometry_in.dtheta
        xi0 = ParallelBeamGeometry_in.xi0
        dxi = ParallelBeamGeometry_in.dxi

        ProjectiondataStaticCT = np.asarray(ProjectiondataStaticCT, dtype=float).ravel()
        ProjectiondataParallelBeam = Allocation.zeros(Mpar * Ns * Npar * NzScans)

        for nz in range(1, NzScans + 1):
            Projections = ProjectiondataStaticCT[(nz - 1) * MDelta * Ns * N:nz * MDelta * Ns * N]
            RegroupedProjections = _Regroup(Projections, MDelta, Ns, N)

            RebinnedProjections = Allocation.zeros(Ns, Mpar * Npar)
            for ns in range(1, Ns + 1):
                subgroup = RegroupedProjections[ns - 1::Ns, :].ravel()
                RebinnedProjections[ns - 1, :] = Rebinning.RebinningStaticCTToParallel(
                    subgroup, RF, RD, MDelta, N, alpha0[nz - 1], dalpha, delta0, dDelta,
                    Mpar, Npar, theta0[nz - 1], dtheta, xi0, dxi, dbetaDash, betaDash0, MbetaDash
                )

            block = Mpar * Ns * Npar
            ProjectiondataParallelBeam[(nz - 1) * block:nz * block] = _Linearize(
                RebinnedProjections, Ns, Npar, Mpar
            )

        return ProjectiondataParallelBeam

    @staticmethod
    def RebinningStaticCTToParallelSpiral(ProjectiondataStaticCT, StaticCTGeometry_in, ParallelBeamGeometry_in):
        Geometry = StaticCTGeometry_in.Geometry
        Parameter = StaticCTGeometry_in.Parameter

        RF = Geometry.RF
        RD = Geometry.RD
        MDelta = Geometry.MDelta
        delta0 = Geometry.delta0
        dDelta = Geometry.dDelta
        dbetaDash = Geometry.dbetaDash
        betaDash0 = Geometry.betaDash0
        MbetaDash = Geometry.MbetaDash
        Ns = Geometry.Ns
        dalpha = Geometry.Sampling.dalpha

        N = Parameter.N
        alpha0 = Parameter.alpha0

        Mpar = ParallelBeamGeometry_in.M
        Npar = ParallelBeamGeometry_in.Npar
        theta0 = ParallelBeamGeometry_in.theta0
        dtheta = ParallelBeamGeometry_in.dtheta
        xi0 = ParallelBeamGeometry_in.xi0
        dxi = ParallelBeamGeometry_in.dxi

        RegroupedProjections = _Regroup(ProjectiondataStaticCT, MDelta, Ns, N)

        RebinnedProjections = Allocation.zeros(Ns, Mpar * Npar)
        for ns in range(1, Ns + 1):
            subgroup = RegroupedProjections[ns - 1::Ns, :].ravel()
            RebinnedProjections[ns - 1, :] = Rebinning.RebinningStaticCTToParallel(
                subgroup, RF, RD, MDelta, N, alpha0, dalpha, delta0, dDelta,
                Mpar, Npar, theta0, dtheta, xi0, dxi, dbetaDash, betaDash0, MbetaDash
            )

        return _Linearize(RebinnedProjections, Ns, Npar, Mpar)

    @staticmethod
    def RebinnedFilename(ScanDescriptor):
        return _CachedFilename(ScanDescriptor, "rebinned")
