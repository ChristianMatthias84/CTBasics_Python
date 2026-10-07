"""Rebinning from fan and static CT geometry to the virtual parallel geometry.

The reference is always a directly simulated parallel-beam sinogram of the same phantom,
so the checks compare two independent implementations rather than a stored result.
"""
import math
from types import SimpleNamespace

import numpy as np
import pytest

from CTBasics.Descriptors.Geometries.StaticCTBeam import StaticCTBeam
from CTBasics.Descriptors.Sampling.AzimuthalSampling import AzimuthalSampling
from CTBasics.Reconstruction.ForwardProjection import ForwardProjection
from CTBasics.Reconstruction.Rebinning import Rebinning

MU = 0.02
R = 80.0
XC, YC = 30.0, -20.0

NX = NY = 64


def _phantom(RM):
    dx = 2 * RM / NX
    x0 = -RM - dx / 2
    axis = x0 + np.arange(1, NX + 1) * dx
    X, Y = np.meshgrid(axis, axis)
    return np.where((X - XC) ** 2 + (Y - YC) ** 2 <= R * R, MU, 0.0).ravel(), x0, dx


def _parallel_reference(Ima, x0, dx, theta0, dtheta, Npar, xi0, dxi, Mpar):
    return np.asarray(ForwardProjection.RayDrivenForwProjParCartesianMultiple(
        x0, dx, NX, x0, dx, NY, Ima, theta0, dtheta, Npar, xi0, dxi, Mpar
    )).reshape(Npar, Mpar)


# --------------------------------------------------------------------- fan beam

def _fan_geometry():
    RF = 595.0
    MFan = 201
    betaMax = 24.8 / 180.0 * math.pi
    dbeta = -2 * betaMax / (MFan - 1)

    Geometry = SimpleNamespace(
        RF=RF,
        MFan=MFan,
        Ns=1,
        beta=betaMax + dbeta * np.arange(MFan),
        RM=math.sin(betaMax) * RF,
        Sampling=AzimuthalSampling(256),
    )

    dalpha = Geometry.Sampling.dalpha
    alphaN = 2 * math.pi + 2 * betaMax

    Parameter = SimpleNamespace(
        alpha0=-betaMax,
        alphaN=alphaN,
        N=int(alphaN / dalpha),
    )

    dxi = 2 * Geometry.RM / (MFan - 1)
    Parallel = SimpleNamespace(
        M=MFan,
        Npar=int((alphaN - 2 * betaMax) / dalpha),
        theta0=Parameter.alpha0 + betaMax,
        dtheta=dalpha,
        dxi=dxi,
        xi0=-Geometry.RM - dxi,
    )

    return SimpleNamespace(Geometry=Geometry, Parameter=Parameter), Parallel


@pytest.fixture(scope="module")
def fan_case():
    Descriptor, Parallel = _fan_geometry()
    Geometry, Parameter = Descriptor.Geometry, Descriptor.Parameter

    Ima, x0, dx = _phantom(Geometry.RM)

    MFan, N = Geometry.MFan, Parameter.N
    FanSinogram = np.zeros(N * MFan)
    for n in range(1, N + 1):
        FanSinogram[(n - 1) * MFan:n * MFan] = ForwardProjection.RayDrivenForwardProj2DFanSingle(
            Geometry.RF, Parameter.alpha0 + n * Geometry.Sampling.dalpha, Geometry.beta,
            x0, dx, NX, x0, dx, NY, Ima, MFan
        )

    Reference = _parallel_reference(Ima, x0, dx, Parallel.theta0, Parallel.dtheta,
                                   Parallel.Npar, Parallel.xi0, Parallel.dxi, Parallel.M)

    return Descriptor, Parallel, FanSinogram, Reference


def test_fan_rebinning_reproduces_the_parallel_sinogram(fan_case):
    Descriptor, Parallel, FanSinogram, Reference = fan_case

    Rebinned = Rebinning.RebinningFanToParallelSpiral(
        FanSinogram, Descriptor, Parallel
    ).reshape(Parallel.Npar, Parallel.M)

    inside = Reference > 0.5 * Reference.max()
    deviation = np.abs(Rebinned - Reference)[inside]

    assert Rebinned.max() == pytest.approx(Reference.max(), rel=0.01)
    assert deviation.mean() < 0.01 * Reference.max()
    assert np.median(deviation) < 0.002 * Reference.max()


def test_fan_rebinning_keeps_the_sinogram_layout(fan_case):
    Descriptor, Parallel, FanSinogram, _ = fan_case

    Rebinned = Rebinning.RebinningFanToParallelSpiral(FanSinogram, Descriptor, Parallel)

    assert Rebinned.shape == (Parallel.Npar * Parallel.M * Descriptor.Geometry.Ns,)
    assert np.all(np.isfinite(Rebinned))
    assert np.all(Rebinned >= 0.0)


def test_fan_sequence_rebinning_repeats_every_scan(fan_case):
    """Two identical scan positions must rebin to two identical blocks."""
    Descriptor, Parallel, FanSinogram, _ = fan_case

    Geometry = Descriptor.Geometry
    Parameter = SimpleNamespace(
        alpha0=[Descriptor.Parameter.alpha0] * 2,
        alphaN=Descriptor.Parameter.alphaN,
        N=Descriptor.Parameter.N,
        NzScans=2,
    )
    Sequence = SimpleNamespace(Geometry=Geometry, Parameter=Parameter)
    ParallelSequence = SimpleNamespace(**{**vars(Parallel),
                                         "theta0": np.array([Parallel.theta0] * 2)})

    Rebinned = Rebinning.RebinningFanToParallelSequence(
        np.concatenate([FanSinogram, FanSinogram]), Sequence, ParallelSequence
    )

    block = Parallel.M * Geometry.Ns * Parallel.Npar
    assert np.array_equal(Rebinned[:block], Rebinned[block:])

    Single = Rebinning.RebinningFanToParallelSpiral(FanSinogram, Descriptor, Parallel)
    assert np.allclose(Rebinned[:block], Single)


# -------------------------------------------------------------------- static CT

def _static_ct_geometry():
    Full = StaticCTBeam()
    Full.Init()

    # A coarser source ring keeps the reference forward projection affordable.
    MDelta = 240
    dDelta = 2 * math.pi / MDelta

    Geometry = SimpleNamespace(
        RF=Full.RF,
        RD=Full.RD,
        RM=Full.RM,
        Ns=1,
        MDelta=MDelta,
        delta0=0.0,
        dDelta=dDelta,
        MbetaDash=MDelta,
        dbetaDash=dDelta,
        betaDash0=-math.pi,
        betaMax=Full.betaMax,
        Sampling=AzimuthalSampling(192),
    )

    dalpha = Geometry.Sampling.dalpha
    alphaN = 2 * math.pi + 2 * Geometry.betaMax

    Parameter = SimpleNamespace(
        alpha0=-Geometry.betaMax,
        alphaN=alphaN,
        N=int(alphaN / dalpha),
    )

    dxi = dDelta * Geometry.RF
    Parallel = SimpleNamespace(
        M=int(2 * Geometry.RD / dxi + 1),
        Npar=int((alphaN - 2 * Geometry.betaMax) / dalpha),
        theta0=Parameter.alpha0 + Geometry.betaMax,
        dtheta=dalpha,
        dxi=dxi,
        xi0=-Geometry.RD - dxi,
    )

    return SimpleNamespace(Geometry=Geometry, Parameter=Parameter), Parallel


@pytest.fixture(scope="module")
def static_ct_case():
    Descriptor, Parallel = _static_ct_geometry()
    Geometry, Parameter = Descriptor.Geometry, Descriptor.Parameter

    Ima, x0, dx = _phantom(Geometry.RM)

    Sinogram = ForwardProjection.RayDrivenForwardProj2DStaticCTMultiple(
        Geometry.RF, Geometry.RD, Parameter.alpha0, Geometry.Sampling.dalpha,
        Geometry.delta0, Geometry.dDelta, x0, dx, NX, x0, dx, NY, Ima,
        Geometry.MDelta, Parameter.N
    )

    Reference = _parallel_reference(Ima, x0, dx, Parallel.theta0, Parallel.dtheta,
                                    Parallel.Npar, Parallel.xi0, Parallel.dxi, Parallel.M)

    return Descriptor, Parallel, np.asarray(Sinogram), Reference


def test_static_ct_rebinning_reproduces_the_parallel_sinogram(static_ct_case):
    Descriptor, Parallel, Sinogram, Reference = static_ct_case
    Geometry, Parameter = Descriptor.Geometry, Descriptor.Parameter

    Rebinned = Rebinning.RebinningStaticCTToParallel(
        Sinogram, Geometry.RF, Geometry.RD, Geometry.MDelta, Parameter.N,
        Parameter.alpha0, Geometry.Sampling.dalpha, Geometry.delta0, Geometry.dDelta,
        Parallel.M, Parallel.Npar, Parallel.theta0, Parallel.dtheta,
        Parallel.xi0, Parallel.dxi, Geometry.dbetaDash, Geometry.betaDash0, Geometry.MbetaDash
    ).reshape(Parallel.Npar, Parallel.M)

    inside = Reference > 0.5 * Reference.max()
    deviation = np.abs(Rebinned - Reference)[inside]

    assert Rebinned.max() == pytest.approx(Reference.max(), rel=0.02)
    assert deviation.mean() < 0.02 * Reference.max()
    assert np.median(deviation) < 0.005 * Reference.max()


def test_static_ct_spiral_wrapper_matches_the_core_routine(static_ct_case):
    Descriptor, Parallel, Sinogram, _ = static_ct_case
    Geometry, Parameter = Descriptor.Geometry, Descriptor.Parameter

    Wrapped = Rebinning.RebinningStaticCTToParallelSpiral(Sinogram, Descriptor, Parallel)

    Direct = Rebinning.RebinningStaticCTToParallel(
        Sinogram, Geometry.RF, Geometry.RD, Geometry.MDelta, Parameter.N,
        Parameter.alpha0, Geometry.Sampling.dalpha, Geometry.delta0, Geometry.dDelta,
        Parallel.M, Parallel.Npar, Parallel.theta0, Parallel.dtheta,
        Parallel.xi0, Parallel.dxi, Geometry.dbetaDash, Geometry.betaDash0, Geometry.MbetaDash
    )

    # With a single detector row the regrouping is the identity.
    assert np.array_equal(Wrapped, Direct)


def test_static_ct_ring_wraps_around(static_ct_case):
    """Source positions beyond the ring must wrap instead of producing empty channels."""
    Descriptor, Parallel, Sinogram, Reference = static_ct_case

    Rebinned = Rebinning.RebinningStaticCTToParallelSpiral(
        Sinogram, Descriptor, Parallel
    ).reshape(Parallel.Npar, Parallel.M)

    illuminated = Reference.max(axis=1) > 0
    assert np.all(Rebinned[illuminated].max(axis=1) > 0)


# ------------------------------------------------------------------- caching

def test_cache_filenames_encode_focal_spot_and_precision():
    from CTBasics.Descriptors.Sampling.FlyingFocalSpot import FlyingFocalSpot
    from CTBasics.Reconstruction.Convolution import Convolution

    Descriptor = SimpleNamespace(
        Filename="scan.raw",
        Geometry=SimpleNamespace(FFS=FlyingFocalSpot("None")),
    )

    assert Rebinning.RebinnedFilename(Descriptor) == "scan_FFS_[1x1]_64_bytes_rebinned.bin"
    assert Convolution.ConvolutedFilename(Descriptor) == "scan_FFS_[1x1]_64_bytes_convoluted.bin"


def test_cache_filename_rejects_foreign_extensions():
    from CTBasics.Descriptors.Sampling.FlyingFocalSpot import FlyingFocalSpot

    Descriptor = SimpleNamespace(
        Filename="scan.dcm",
        Geometry=SimpleNamespace(FFS=FlyingFocalSpot("None")),
    )

    with pytest.raises(ValueError):
        Rebinning.RebinnedFilename(Descriptor)
