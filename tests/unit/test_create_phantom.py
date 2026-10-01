import numpy as np

from ctbasics.simulation import Constants, CreatePhantom, Materials


def test_water_phantom_creates_center_fill():
    phantom, is_spectral = CreatePhantom.create_voxel_phantom_2d(301, 1.0, -151.0, 301, 1.0, -151.0, "WaterPhantom")
    image = phantom.reshape(301, 301)

    assert not is_spectral
    assert phantom.shape == (301 * 301,)
    assert image[150, 150] == Constants.MuW
    assert image[0, 0] == 0.0


def test_simple_ellipse_with_hips_overwrites_inner_regions():
    phantom, is_spectral = CreatePhantom.create_voxel_phantom_2d(301, 1.0, -151.0, 301, 1.0, -151.0, "SimpleEllipseWithHips")
    image = phantom.reshape(301, 301)

    assert not is_spectral
    assert image[150, 150] == Constants.MuW
    assert image[150, 50] == 1.8 * Constants.MuW
    assert image[150, 250] == 1.8 * Constants.MuW


def test_spectral_phantom_uses_material_codes():
    phantom, is_spectral = CreatePhantom.create_voxel_phantom_2d(301, 1.0, -151.0, 301, 1.0, -151.0, "SpectralHipWithMetal")
    image = phantom.reshape(301, 301)

    assert is_spectral
    assert image[150, 150] == Materials.Nyst_water
    assert image[150, 50] == Materials.Nyst_aluminum
    assert image[150, 250] == Materials.Nyst_aluminum


def test_simple_ellipse_with_hips_3d_varies_with_slice_index():
    phantom, is_spectral = CreatePhantom.create_voxel_phantom_3d("SimpleEllipseWithHips3D", 301, 1.0, -151.0, 301, 1.0, -151.0, 10, 1.0, 0.0)
    volume = phantom.reshape(10, 301, 301)

    assert not is_spectral
    assert volume[0, 150, 150] == 1.8 * Constants.MuW
    assert volume[6, 150, 150] == Constants.MuW
    assert volume[0, 100, 150] == 1.8 * Constants.MuW
    assert volume[6, 100, 150] == Constants.MuW


def test_legacy_cardiac_wrapper_maps_plaque_type():
    phantom, is_spectral, x_point, y_point = CreatePhantom.create_voxel_phantom_cardiac_2d_with_motion(
        301,
        1.0,
        -151.0,
        301,
        1.0,
        -151.0,
        0.0,
        0.1,
        0,
        ContrastMedium=True,
        Stent=False,
        PlaqueType="half",
    )

    assert phantom.shape == (301 * 301,)
    assert not is_spectral
    assert np.isfinite(phantom).all()
    assert x_point == 0.0
    assert np.isfinite(y_point)