import numpy as np
import pytest

from CTBasics.Descriptors import ScanDescriptor, ScanTypes, ScannerGeometryTypes
from CTBasics.Scripts.ColliScript import (
    CreateExampleLeafCollimatorConfig,
    RunCollimatorTestByMode,
)


def _SmallStaticCTDescriptor():
    descriptor = ScanDescriptor.Create(ScannerGeometryTypes.StaticCT, ScanTypes.Spiral)
    Geometry = descriptor.Geometry
    Geometry.MDelta = 8
    Geometry.Ns = 4
    Geometry.dDelta = 2.0 * np.pi / Geometry.MDelta
    Geometry.ds = 1.0
    Geometry.soffset = -2.5
    Geometry.s0 = -2.5 - Geometry.Ns * Geometry.ds - Geometry.ds / 2.0
    return descriptor


def test_run_colli_script_standard_mode_returns_projection_arrays():
    descriptor = _SmallStaticCTDescriptor()

    collimation, array_of_colli = RunCollimatorTestByMode(
        "standard", Descriptor=descriptor
    )

    assert collimation.ndim == 1
    assert collimation.size == descriptor.Geometry.MDelta * descriptor.Geometry.Ns
    assert array_of_colli.ndim == 2
    assert np.isfinite(collimation).all()
    assert np.isfinite(array_of_colli).all()


def test_run_colli_script_leaf_mode_returns_projection_arrays():
    descriptor = _SmallStaticCTDescriptor()

    collimation, array_of_colli = RunCollimatorTestByMode(
        "LeafTest", Descriptor=descriptor
    )

    assert collimation.size == descriptor.Geometry.MDelta * descriptor.Geometry.Ns
    assert np.isfinite(collimation).all()
    assert np.isfinite(array_of_colli).all()


def test_all_mode_aliases_of_the_matlab_switch_are_accepted():
    descriptor = _SmallStaticCTDescriptor()

    Reference, _ = RunCollimatorTestByMode("standard", Descriptor=descriptor)
    for alias in ("StandardCollimator", "Ring", "old"):
        collimation, _ = RunCollimatorTestByMode(alias, Descriptor=descriptor)
        assert np.array_equal(collimation, Reference)


def test_unknown_mode_is_rejected():
    descriptor = _SmallStaticCTDescriptor()

    with pytest.raises(ValueError):
        RunCollimatorTestByMode("DoesNotExist", Descriptor=descriptor)


def test_example_leaf_configuration_closes_the_second_half_of_the_leafs():
    descriptor = _SmallStaticCTDescriptor()
    Geometry = descriptor.Geometry

    LeafConfig = CreateExampleLeafCollimatorConfig(Geometry)
    Height = Geometry.zCollimatorUpper - Geometry.zCollimatorLower

    assert LeafConfig["N_collimatorLeafs"] == 100
    assert np.allclose(LeafConfig["zUpperPerLeaf"][:50], Geometry.zCollimatorUpper)
    assert np.allclose(LeafConfig["zLowerPerLeaf"][:50], Geometry.zCollimatorLower)
    assert np.allclose(LeafConfig["zUpperPerLeaf"][50:], Geometry.zCollimatorUpper - 0.5 * Height)
    assert np.allclose(LeafConfig["zLowerPerLeaf"][50:], Geometry.zCollimatorLower + 0.5 * Height)
