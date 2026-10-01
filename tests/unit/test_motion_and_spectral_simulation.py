import numpy as np

from ctbasics.simulation import Constants, CreatePhantom, Materials
from ctbasics.simulation.material_properties import MaterialProperties


def test_motion_profile_full_and_rest_phase():
    on_peak = CreatePhantom.motion_profile(0.0, 60, True, "full", 0.0)
    rest = CreatePhantom.motion_profile(1.0, 60, True, "full", 0.25)

    assert on_peak == 0.0
    assert rest == 0.0


def test_cardiac_phantom_with_plaque_sets_motion_outputs():
    phantom, is_spectral, x_disp, y_disp = CreatePhantom.create_voxel_phantom_2d_with_motion_respiratory(
        301, 1.0, -151.0, 301, 1.0, -151.0, 0.0, 0.1, 0, "CardiacPhantomWithPlaque"
    )

    image = phantom.reshape(301, 301)
    assert not is_spectral
    assert image[150, 150] > Constants.MuW
    assert x_disp == 0.0
    assert y_disp == 0.0


def test_material_properties_interpolates_spectral_coefficients():
    props = MaterialProperties()
    value = props.get_spectral_attenuation_coefficient(Materials.Nyst_water, 70.0)

    assert np.isfinite(value)
    assert value > 0.0