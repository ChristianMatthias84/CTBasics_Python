import numpy as np

from ctbasics.descriptors import ScannerGeometryTypes, ScanTypes, ScanDescriptor, FocalSpotSimulationStaticCT
from ctbasics.reconstruction import ForwardProjection
from ctbasics.visualization import VisualizationTests


def test_static_ct_focal_spot_sampling_positions_shape_and_symmetry():
    samples = FocalSpotSimulationStaticCT.calculate_focal_spot_sampling_positions()

    assert samples.shape == (105, 3)
    assert np.isclose(samples[0, 2], -samples[-1, 2])
    assert np.isclose(samples[0, 0], -samples[-1, 0])


def test_default_static_ct_descriptor_and_leaf_config():
    descriptor = VisualizationTests.create_default_static_ct_descriptor()
    geometry = VisualizationTests.resolve_static_ct_geometry(descriptor)
    leaf_config = VisualizationTests.create_example_leaf_collimator_config(geometry)

    assert descriptor.Geometry.BeamType == ScannerGeometryTypes.StaticCT
    assert descriptor.Parameter.__class__.__name__ == "Spiral"
    assert leaf_config["N_collimatorLeafs"] == 100
    assert len(leaf_config["zUpperPerLeaf"]) == 100
    assert len(leaf_config["zLowerPerLeaf"]) == 100


def test_static_ct_collimation_helpers_return_expected_shapes():
    descriptor = VisualizationTests.create_default_static_ct_descriptor()
    descriptor.Geometry.MDelta = 8
    descriptor.Geometry.Ns = 4
    descriptor.Geometry.dDelta = 2.0 * np.pi / descriptor.Geometry.MDelta
    descriptor.Geometry.ds = 1.0
    descriptor.Geometry.soffset = -2.5
    descriptor.Geometry.s0 = -2.5 - descriptor.Geometry.Ns * descriptor.Geometry.ds - descriptor.Geometry.ds / 2.0
    focal_spots = FocalSpotSimulationStaticCT.calculate_focal_spot_sampling_positions(nr_of_width_samples=1, nr_of_length_samples=3)

    averaged, array_of_collimation = ForwardProjection.static_ct_collimation_single_source_sampled(descriptor, np.pi / 2.0, focal_spots)
    leaf_config = VisualizationTests.create_example_leaf_collimator_config(descriptor.Geometry)
    leaf_averaged, leaf_array = ForwardProjection.static_ct_collimation_leaf_single_source_sampled(descriptor, np.pi / 2.0, focal_spots, leaf_config)

    assert averaged.shape == (descriptor.Geometry.MDelta * descriptor.Geometry.Ns,)
    assert array_of_collimation.shape == (focal_spots.shape[0], descriptor.Geometry.MDelta * descriptor.Geometry.Ns)
    assert leaf_averaged.shape == averaged.shape
    assert leaf_array.shape == array_of_collimation.shape
