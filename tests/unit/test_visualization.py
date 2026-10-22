import numpy as np

from CTBasics.Descriptors import FocalSpotSimulationStaticCT, ScannerGeometryTypes
from CTBasics.Reconstruction import ForwardProjection
from CTBasics.Scripts.ColliScript import CreateExampleLeafCollimatorConfig
from CTBasics.Visualization import StaticCTVisualizationFunctionalitites


def test_static_ct_focal_spot_sampling_positions_shape_and_symmetry():
    samples = FocalSpotSimulationStaticCT.CalculateFocalSpotSamplingPositions()

    assert samples.shape == (105, 3)
    assert np.isclose(samples[0, 2], -samples[-1, 2])
    assert np.isclose(samples[0, 0], -samples[-1, 0])


def test_default_static_ct_descriptor_and_leaf_config():
    descriptor = StaticCTVisualizationFunctionalitites.CreateDefaultStaticCTDescriptor()
    geometry = StaticCTVisualizationFunctionalitites.ResolveStaticCTGeometry(descriptor)
    leaf_config = CreateExampleLeafCollimatorConfig(geometry)

    assert descriptor.Geometry.BeamType == ScannerGeometryTypes.StaticCT
    assert descriptor.Parameter.__class__.__name__ == "Spiral"
    assert leaf_config["N_collimatorLeafs"] == 100
    assert len(leaf_config["zUpperPerLeaf"]) == 100
    assert len(leaf_config["zLowerPerLeaf"]) == 100


def _small_static_ct_descriptor():
    descriptor = StaticCTVisualizationFunctionalitites.CreateDefaultStaticCTDescriptor()
    descriptor.Geometry.MDelta = 8
    descriptor.Geometry.Ns = 4
    descriptor.Geometry.dDelta = 2.0 * np.pi / descriptor.Geometry.MDelta
    descriptor.Geometry.ds = 1.0
    descriptor.Geometry.soffset = -2.5
    descriptor.Geometry.s0 = -2.5 - descriptor.Geometry.Ns * descriptor.Geometry.ds - descriptor.Geometry.ds / 2.0
    return descriptor


def test_static_ct_collimation_mask_is_binary_and_partially_open():
    descriptor = _small_static_ct_descriptor()

    mask = ForwardProjection.StaticCTCollimationSingle(descriptor, np.pi / 2.0, [0.0, 0.0, 0.0])

    assert mask.shape == (descriptor.Geometry.MDelta * descriptor.Geometry.Ns,)
    assert set(np.unique(mask)).issubset({0.0, 1.0})
    # A correctly placed collimator blocks some rays and passes others.
    assert 0 < mask.sum() < mask.size


def test_widening_the_collimator_opens_at_least_as_many_rays():
    descriptor = _small_static_ct_descriptor()
    narrow = ForwardProjection.StaticCTCollimationSingle(descriptor, np.pi / 2.0, [0.0, 0.0, 0.0])

    descriptor.Geometry.zCollimatorUpper += 2.0
    descriptor.Geometry.zCollimatorLower -= 2.0
    wide = ForwardProjection.StaticCTCollimationSingle(descriptor, np.pi / 2.0, [0.0, 0.0, 0.0])

    assert wide.sum() >= narrow.sum()
    assert np.all(wide >= narrow)


def test_leaf_collimation_with_uniform_leaves_matches_the_plain_collimator():
    descriptor = _small_static_ct_descriptor()
    geometry = descriptor.Geometry

    plain = ForwardProjection.StaticCTCollimationSingle(descriptor, np.pi / 2.0, [0.0, 0.0, 0.0])

    # Every leaf carries the global collimator bounds, so both paths must agree exactly.
    uniform_config = {
        "N_collimatorLeafs": 4,
        "zUpperPerLeaf": np.full(4, geometry.zCollimatorUpper),
        "zLowerPerLeaf": np.full(4, geometry.zCollimatorLower),
    }
    leaf = ForwardProjection.StaticCTCollimationLeafSingle(descriptor, np.pi / 2.0, [0.0, 0.0, 0.0], uniform_config)

    assert np.array_equal(plain, leaf)


def test_leaf_config_validation_rejects_inconsistent_input():
    import pytest

    with pytest.raises(ValueError, match="missing required field"):
        ForwardProjection.ValidateLeafCollimatorConfig({"N_collimatorLeafs": 2}, 8)

    with pytest.raises(ValueError, match="length N_collimatorLeafs"):
        ForwardProjection.ValidateLeafCollimatorConfig(
            {"N_collimatorLeafs": 2, "zUpperPerLeaf": [1.0], "zLowerPerLeaf": [0.0]}, 8
        )

    with pytest.raises(ValueError, match="length MDelta"):
        ForwardProjection.ValidateLeafCollimatorConfig(
            {
                "N_collimatorLeafs": 2,
                "zUpperPerLeaf": [1.0, 1.0],
                "zLowerPerLeaf": [0.0, 0.0],
                "leafIndexPerDetectorElement": [1, 2, 1],
            },
            8,
        )


def test_default_leaf_index_mapping_segments_the_detector_uniformly():
    config = {"N_collimatorLeafs": 4, "zUpperPerLeaf": np.ones(4), "zLowerPerLeaf": np.zeros(4)}

    indices = [ForwardProjection.GetLeafIndexFromDetectorElement(m, 8, config) for m in range(1, 9)]

    assert indices == [1, 1, 2, 2, 3, 3, 4, 4]


def test_static_ct_collimation_source_sampling_sums_the_single_results():
    descriptor = _small_static_ct_descriptor()
    focal_spots = FocalSpotSimulationStaticCT.CalculateFocalSpotSamplingPositions(
        nrOfWidthSamples=1, nrOfLengthSamples=3
    )

    averaged, array_of_collimation = ForwardProjection.StaticCTCollimationSingleSourceSampled(
        descriptor, np.pi / 2.0, focal_spots
    )

    assert array_of_collimation.shape == (focal_spots.shape[0], descriptor.Geometry.MDelta * descriptor.Geometry.Ns)
    for i in range(focal_spots.shape[0]):
        expected = ForwardProjection.StaticCTCollimationSingle(descriptor, np.pi / 2.0, focal_spots[i])
        assert np.array_equal(array_of_collimation[i], expected)

    # MATLAB accumulates the samples without dividing by their count.
    assert np.array_equal(averaged, array_of_collimation.sum(axis=0))


def test_collimator_circle_intersection_lies_on_circle_and_line():
    RF, RD, RC = 586.3, 425.0, 413.5
    alpha, delta = np.pi / 2.0, 1.1

    x1, x2, y1, y2 = ForwardProjection.CalculateIntersectionPointsCircleAndLineIn2D_StaticCTGeo(
        RF, RD, RC, alpha, delta
    )

    assert np.isclose(np.hypot(x1, y1), RC)
    assert np.isclose(np.hypot(x2, y2), RC)

    # Both points must lie on the straight line through source and detector element.
    sx, sy = RF * np.sin(alpha), -RF * np.cos(alpha)
    dx_, dy_ = RD * np.cos(delta), RD * np.sin(delta)
    cross1 = (x1 - sx) * (dy_ - sy) - (y1 - sy) * (dx_ - sx)
    cross2 = (x2 - sx) * (dy_ - sy) - (y2 - sy) * (dx_ - sx)
    assert abs(cross1) < 1e-6 * RF * RD
    assert abs(cross2) < 1e-6 * RF * RD


# --------------------------------------------------------------------- figures

def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.close("all")
    return plt


def test_rebin_variables_plot_shows_the_fan_angle_relation():
    _plt()
    axes = StaticCTVisualizationFunctionalitites.ShowRebinVariables2D()

    beta_curve, identity = axes.get_lines()
    betaDash, beta = beta_curve.get_data()

    assert np.array_equal(identity.get_ydata(), betaDash)
    # beta(betaDash) stays below the identity because the detector radius is finite.
    assert np.all(beta[1:] < betaDash[1:])
    assert np.isclose(beta[0], 0.0)


def test_ring_sketch_draws_source_and_detector_rings():
    _plt()
    axes = StaticCTVisualizationFunctionalitites.ShowRingSketch3D()

    radii = [float(np.hypot(*line.get_data()).max()) for line in axes.get_lines()]

    assert np.isclose(max(r for r in radii if r < 450), 400.0, atol=1.0)
    assert np.isclose(max(radii), 500.0, atol=1.0)


def test_rebinned_geometry_lines_are_equidistant_and_widest_in_the_centre():
    _plt()
    axes, surface = StaticCTVisualizationFunctionalitites.ShowRebinnedGeometry3D()

    xi = axes.get_lines()[0].get_xdata()
    z = np.array([line.get_ydata() for line in axes.get_lines()])

    # 17 row boundaries of a 16-row detector.
    assert z.shape[0] == 17

    # At every channel the rows are equidistant; the spacing follows the ray length L.
    spacing = np.diff(z, axis=0)
    assert np.allclose(spacing, spacing[0], atol=1e-9)

    centre = np.argmin(np.abs(xi))
    assert spacing[0, centre] == spacing[0].max()
    assert surface.get_ylabel() == "z [mm]"


def test_static_ct_geometry_plot_uses_the_descriptor_radii():
    _plt()
    descriptor = _small_static_ct_descriptor()

    axes = StaticCTVisualizationFunctionalitites.ShowStaticCTGeometry3D(descriptor)

    radii = {round(float(np.hypot(*line.get_data()).max()), 1) for line in axes.get_lines()}

    assert round(descriptor.Geometry.RF, 1) in radii
    assert round(descriptor.Geometry.RD, 1) in radii
    assert round(descriptor.Geometry.RC, 1) in radii


def test_static_ct_geometry_rejects_a_fan_beam():
    import pytest

    from CTBasics.Descriptors.Geometries.FanBeam import FanBeam

    with pytest.raises(ValueError):
        StaticCTVisualizationFunctionalitites.ShowStaticCTGeometry3D(FanBeam())


def test_collimation_view_returns_the_detector_illumination():
    _plt()
    descriptor = _small_static_ct_descriptor()
    focal_spots = FocalSpotSimulationStaticCT.CalculateFocalSpotSamplingPositions(
        nrOfWidthSamples=1, nrOfLengthSamples=3
    )

    collimation, detectorImage = StaticCTVisualizationFunctionalitites.ShowStaticCTCollimation3D(
        descriptor, focalSpotSamples=focal_spots
    )

    assert detectorImage.shape == (descriptor.Geometry.Ns, descriptor.Geometry.MDelta)
    assert np.array_equal(detectorImage.ravel(), np.asarray(collimation))
    assert detectorImage.max() > 0


def test_collimation_view_can_skip_the_illumination():
    _plt()
    collimation, detectorImage = StaticCTVisualizationFunctionalitites.ShowStaticCTCollimation3D(
        _small_static_ct_descriptor(), computeIllumination=False
    )

    assert collimation is None
    assert detectorImage is None


def test_multi_leaf_view_requires_a_leaf_configuration():
    import pytest

    _plt()
    with pytest.raises(ValueError):
        StaticCTVisualizationFunctionalitites.ShowStaticCTCollimation3DFromComputedDataLeaf(
            _small_static_ct_descriptor()
        )


def test_multi_leaf_view_draws_two_edges_per_leaf():
    from types import SimpleNamespace

    plt = _plt()
    NLeafs = 4
    LeafConfig = SimpleNamespace(
        N_collimatorLeafs=NLeafs,
        zUpperPerLeaf=np.linspace(-1.0, -2.0, NLeafs),
        zLowerPerLeaf=np.linspace(-7.0, -8.0, NLeafs),
    )

    StaticCTVisualizationFunctionalitites.ShowStaticCTCollimation3DFromComputedDataLeaf(
        _small_static_ct_descriptor(),
        focalSpotSamples=np.zeros((1, 3)),
        LeafConfig=LeafConfig,
    )

    axes = plt.gcf().axes[0]
    leaf_edges = [line for line in axes.get_lines()
                  if len(np.atleast_1d(line.get_xdata())) == 2]

    assert len(leaf_edges) >= 2 * NLeafs
