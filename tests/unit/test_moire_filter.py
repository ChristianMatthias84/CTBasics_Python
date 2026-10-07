"""Row-wise Moire filtering of a 3D parallel sinogram."""
from types import SimpleNamespace

import numpy as np

from CTBasics.Descriptors.ScanTypes import ScanTypes
from CTBasics.Reconstruction.MoireFilter import MoireFilter

M = 64
NS = 4
NTOTAL = 8
DXI = 1.0


def _geometries(ScanType):
    ParGeo = SimpleNamespace(dxi=DXI, M=M, Npar_spi=NTOTAL, Npar_seq=NTOTAL // 2)
    FanGeo = SimpleNamespace(Ns=NS, ScanType=ScanType, NzScans_seq=2)
    return ParGeo, FanGeo


def _sinogram():
    rng = np.random.default_rng(0)
    return rng.standard_normal(NTOTAL * NS * M)


def test_moire_filter_3d_keeps_the_sinogram_layout():
    ParGeo, FanGeo = _geometries(ScanTypes.Spiral)
    Projections = _sinogram()

    Filtered = MoireFilter.ApplyMoireFilterOnProjections3D(Projections, ParGeo, FanGeo)

    assert Filtered.shape == Projections.shape
    assert np.all(np.isfinite(Filtered))


def test_moire_filter_3d_filters_each_detector_row_independently():
    ParGeo, FanGeo = _geometries(ScanTypes.Spiral)
    Projections = _sinogram()

    Filtered = MoireFilter.ApplyMoireFilterOnProjections3D(
        Projections, ParGeo, FanGeo
    ).reshape(NTOTAL, NS, M)

    Rows = np.asarray(Projections).reshape(NTOTAL, NS, M)
    for ns in range(NS):
        Expected = MoireFilter.ApplyMoireFilterOnProjections(
            Rows[:, ns, :].ravel(), NTOTAL, M, DXI, 0.2 / DXI, 0.5 / DXI
        ).reshape(NTOTAL, M)
        assert np.allclose(Filtered[:, ns, :], Expected)


def test_moire_filter_3d_uses_the_sequence_view_count():
    ParGeo, FanGeo = _geometries(ScanTypes.Sequence)
    Projections = _sinogram()

    Filtered = MoireFilter.ApplyMoireFilterOnProjections3D(Projections, ParGeo, FanGeo)

    # Npar_seq * NzScans_seq must reproduce the same total view count.
    assert Filtered.shape == Projections.shape


def test_moire_filter_passes_frequencies_outside_the_stop_band():
    """A constant projection has only a DC component and must survive unchanged."""
    ParGeo, FanGeo = _geometries(ScanTypes.Spiral)
    Projections = np.full(NTOTAL * NS * M, 3.0)

    Filtered = MoireFilter.ApplyMoireFilterOnProjections3D(Projections, ParGeo, FanGeo)

    assert np.allclose(Filtered, 3.0)
