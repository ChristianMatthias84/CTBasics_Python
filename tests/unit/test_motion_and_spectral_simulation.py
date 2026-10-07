import numpy as np
import pytest

from CTBasics.Reconstruction import Constants
from CTBasics.Simulation import CreatePhantom, MaterialProperties, Materials


def test_motion_profile_full_and_rest_phase():
    on_peak = CreatePhantom.MotionProfile(0.0, 60, True, "full", 0.0)
    rest = CreatePhantom.MotionProfile(1.0, 60, True, "full", 0.25)

    assert on_peak == 0.0
    assert rest == 0.0


def test_motion_profile_full_cycle_is_a_sine_over_the_motion_phase():
    # HeartRate 60 bpm gives Tcycle = 1 s; without rest the whole cycle is one sine period.
    assert np.isclose(CreatePhantom.MotionProfile(0.25, 60, True, "full", 0.0), 1.0)
    assert np.isclose(CreatePhantom.MotionProfile(0.5, 60, True, "full", 0.0), 0.0, atol=1e-12)
    assert np.isclose(CreatePhantom.MotionProfile(0.75, 60, True, "full", 0.0), -1.0)


def test_motion_profile_half_cycle_never_goes_negative():
    values = [CreatePhantom.MotionProfile(t, 60, True, "half", 0.0) for t in np.linspace(0, 0.99, 50)]
    assert min(values) >= 0.0
    assert np.isclose(max(values), 1.0, atol=1e-3)


def test_motion_profile_is_periodic_and_rests_after_the_motion_phase():
    # TRest = 0.5 s leaves TMotion = 0.5 s of motion followed by 0.5 s of rest.
    assert np.isclose(CreatePhantom.MotionProfile(0.125, 60, True, "full", 0.5), 1.0)
    assert CreatePhantom.MotionProfile(0.75, 60, True, "full", 0.5) == 0.0
    assert np.isclose(
        CreatePhantom.MotionProfile(0.125, 60, True, "full", 0.5),
        CreatePhantom.MotionProfile(3.125, 60, True, "full", 0.5),
    )


def test_motion_profile_returns_zero_when_motion_is_disabled():
    assert CreatePhantom.MotionProfile(0.25, 60, False, "full", 0.0) == 0.0


def test_motion_profile_rejects_invalid_configuration():
    with pytest.raises(ValueError, match="must be smaller than the cycle duration"):
        CreatePhantom.MotionProfile(0.1, 60, True, "full", 2.0)

    with pytest.raises(ValueError, match="Unknown MotionType"):
        CreatePhantom.MotionProfile(0.1, 60, True, "quarter", 0.0)


def test_material_properties_interpolates_spectral_coefficients():
    props = MaterialProperties()
    value = props.GetSpectralAttenuationCoefficient(Materials.Nyst_water, 70.0)

    assert np.isfinite(value)
    assert value > 0.0
