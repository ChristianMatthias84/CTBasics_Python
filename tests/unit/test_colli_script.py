import numpy as np

from ctbasics.descriptors import ScanDescriptor, ScanTypes, ScannerGeometryTypes
from ctbasics.apps import run_colli_script


def test_run_colli_script_standard_mode_returns_projection_arrays():
    descriptor = ScanDescriptor.create(ScannerGeometryTypes.StaticCT, ScanTypes.Spiral)
    descriptor.Geometry.MDelta = 8
    descriptor.Geometry.Ns = 4
    descriptor.Geometry.dDelta = 2.0 * np.pi / descriptor.Geometry.MDelta
    descriptor.Geometry.ds = 1.0
    descriptor.Geometry.soffset = -2.5
    descriptor.Geometry.s0 = -2.5 - descriptor.Geometry.Ns * descriptor.Geometry.ds - descriptor.Geometry.ds / 2.0

    collimation, array_of_colli = run_colli_script("standard", descriptor=descriptor)

    assert collimation.ndim == 1
    assert array_of_colli.ndim == 2
    assert np.isfinite(collimation).all()
    assert np.isfinite(array_of_colli).all()