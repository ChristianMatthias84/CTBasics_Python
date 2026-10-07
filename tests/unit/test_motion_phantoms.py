"""Geometric and temporal checks of the dynamic (motion) phantoms."""
import numpy as np
import pytest

from CTBasics.Reconstruction.Constants import Constants
from CTBasics.Simulation.CreatePhantom import CreatePhantom

NX = NY = 256
RM = 250.0
D = 2 * RM / NX
ORIGIN = -RM - D / 2

MU_W = Constants.MuW
MU_ART = 0.019968
MU_BLOOD = 0.020256
MU_BLOOD_CM = 0.0250
MU_PLAQUE = 0.0240
MU_STENT = 0.02592


def _grid():
    axis = ORIGIN + np.arange(1, NX + 1) * D
    return np.meshgrid(axis, axis)


def _cardiac(nT, **options):
    return CreatePhantom.CreateVoxelPhantom2DWithMotionCardiac(
        NX, D, ORIGIN, NY, D, ORIGIN, 0.0, 0.025, nT, **options
    )


def _unified(SelectedPhantom, nT=0, dT=0.01, **options):
    return CreatePhantom.CreateVoxelPhantom2DWithMotionUnified(
        NX, D, ORIGIN, NY, D, ORIGIN, 0.0, dT, nT, SelectedPhantom, **options
    )


def _area(Phantom, value):
    return np.count_nonzero(np.isclose(Phantom, value)) * D * D


def test_cardiac_water_disc_has_the_analytic_area():
    Phantom, _, _, _ = _cardiac(0, Motion=False)

    # Water, artery wall and lumen together fill the 50 mm disc.
    covered = np.count_nonzero(Phantom > 0) * D * D
    assert covered == pytest.approx(np.pi * 50.0 ** 2, rel=0.02)


def test_cardiac_lumen_and_wall_have_the_analytic_areas():
    # The wall is a 0.5 mm shell, which the 1.95 mm clinical grid cannot resolve,
    # so it is measured on a fine grid over a small field of view instead.
    N = 512
    fov = 15.0
    d = 2 * fov / N
    origin = -fov - d / 2

    Phantom, _, _, _ = CreatePhantom.CreateVoxelPhantom2DWithMotionCardiac(
        N, d, origin, N, d, origin, 0.0, 0.025, 0, Motion=False
    )

    lumen = np.count_nonzero(np.isclose(Phantom, MU_BLOOD)) * d * d
    wall = np.count_nonzero(np.isclose(Phantom, MU_ART)) * d * d

    assert lumen == pytest.approx(np.pi * 4.5 ** 2, rel=0.01)
    assert wall == pytest.approx(np.pi * (5.0 ** 2 - 4.5 ** 2), rel=0.02)


def test_cardiac_contrast_medium_replaces_the_lumen_value():
    Phantom, _, _, _ = _cardiac(0, Motion=False, CM=True)

    assert _area(Phantom, MU_BLOOD) == 0.0
    assert _area(Phantom, MU_BLOOD_CM) == pytest.approx(np.pi * 4.5 ** 2, rel=0.05)


def test_cardiac_displacement_amplitude_is_two_outer_radii():
    Y = _grid()[1]

    centers = []
    for nT in range(40):
        Phantom, _, _, _ = _cardiac(nT)
        mask = Phantom.reshape(NY, NX) > MU_W + 1e-6
        centers.append((Y * mask).sum() / mask.sum())

    centers = np.array(centers)

    # Default direction is 0, so the artery moves along y by 2*r = 10 mm.
    assert centers.max() == pytest.approx(10.0, abs=1.0)
    assert centers.min() == pytest.approx(-10.0, abs=1.0)


def test_cardiac_direction_switches_the_motion_axis():
    _, _, xd, yd = _cardiac(5, Direction=np.pi / 2)

    assert abs(yd) < 1e-12
    assert abs(xd) > 1.0


def test_cardiac_motion_flag_freezes_the_phantom():
    a = _cardiac(3, Motion=False)[0]
    b = _cardiac(17, Motion=False)[0]

    assert np.array_equal(a, b)


@pytest.mark.parametrize("Plaque, fraction", [(1, 0.5), (2, 0.25)])
def test_cardiac_plaque_covers_the_expected_lumen_fraction(Plaque, fraction):
    Phantom, _, _, _ = _cardiac(0, Motion=False, Plaque=Plaque)

    lumen = np.pi * 4.5 ** 2
    assert _area(Phantom, MU_PLAQUE) == pytest.approx(fraction * lumen, rel=0.1)


def test_cardiac_stents_add_eight_struts():
    Phantom, _, _, _ = _cardiac(0, Motion=False, Stent=True)

    assert _area(Phantom, MU_STENT) == pytest.approx(8 * np.pi * 0.8 ** 2, rel=0.25)


def test_water_only_phantom_is_static_and_homogeneous():
    a = _unified("WaterOnly", nT=0)[0]
    b = _unified("WaterOnly", nT=37)[0]

    assert np.array_equal(a, b)
    assert set(np.unique(a)) == {0.0, MU_W}


def test_unified_motion_amplitude_is_ten_millimetres():
    displacements = [_unified("Cardiac", nT=n, dT=0.02)[3] for n in range(50)]

    assert max(displacements) == pytest.approx(10.0, abs=0.5)
    assert min(displacements) == pytest.approx(-10.0, abs=0.5)


def test_unified_rejects_unknown_phantoms():
    with pytest.raises(ValueError, match="Unknown phantom type"):
        _unified("DoesNotExist")


def test_four_artery_phantom_reports_all_artery_centres():
    _, _, _, _, RoI = _unified("WaterWith4Arteries", Motion=False)

    assert RoI.shape == (2, 4)
    # The arteries sit on a circle of radius 25 mm.
    assert np.hypot(RoI[0], RoI[1]) == pytest.approx(25.0, rel=1e-9)


def test_four_artery_phantom_contains_stents_and_all_plaque_variants():
    Phantom, _, _, _, _ = _unified("WaterWith4Arteries", Motion=False)

    assert _area(Phantom, MU_STENT) > 0.0
    assert _area(Phantom, MU_PLAQUE) > 0.0
    assert _area(Phantom, MU_ART) == pytest.approx(4 * np.pi * (5.0 ** 2 - 4.5 ** 2), rel=0.15)


def test_line_pattern_has_four_stripes():
    Phantom, _, _, _, _ = _unified("LinePattern", LineWidth=6.0, Motion=False)

    row = Phantom.reshape(NY, NX)[NY // 2] > MU_W + 1e-6
    starts = np.diff(np.concatenate(([0], row.astype(np.int8), [0])))

    assert np.count_nonzero(starts == 1) == 4


def test_line_pattern_rotation_turns_the_stripes():
    upright = _unified("LinePattern", LineWidth=6.0, Motion=False)[0]
    turned = _unified("LinePattern", LineWidth=6.0, Rotation=np.pi / 2, Motion=False)[0]

    assert not np.array_equal(upright, turned)
    # A quarter turn preserves the painted area.
    assert np.count_nonzero(np.isclose(upright, MU_PLAQUE)) == pytest.approx(
        np.count_nonzero(np.isclose(turned, MU_PLAQUE)), rel=0.05
    )


def test_iodine_concentration_follows_the_cardiac_cycle():
    peaks = [_unified("TemporalIodinePhantom", nT=n, dT=0.05)[0].max() for n in range(20)]

    mu_blood = (1 + 55 / 1000) * MU_W

    # The lumen peaks at mu_blood + IodinePeakFactor * mu_w and never falls below blood.
    assert max(peaks) == pytest.approx(mu_blood + 3.0 * MU_W, rel=0.05)
    assert min(peaks) > mu_blood


def test_iodine_vessels_do_not_move():
    a = _unified("TemporalIodinePhantom", nT=0)
    assert a[2] == 0 and a[3] == 0


def test_respiratory_structures_move_with_the_breathing_cycle():
    a = _unified("Respiratory", nT=0, dT=0.5, Heartrate=15)[0]
    b = _unified("Respiratory", nT=2, dT=0.5, Heartrate=15)[0]

    assert not np.array_equal(a, b)


def test_respiratory_lungs_are_air_filled_and_reported_as_roi():
    Phantom, _, _, _, RoI = _unified("Respiratory", Motion=False)

    assert RoI.shape == (2, 2)
    assert list(RoI[0]) == [-70.0, 70.0]

    X, Y = _grid()
    Slice = Phantom.reshape(NY, NX)

    # A point well inside the left lung but away from any bronchus must be air.
    assert Slice[np.argmin(np.abs(Y[:, 0] - (-35.0))), np.argmin(np.abs(X[0] - (-70.0)))] == 0.0


def test_thorax_ellipse_has_the_analytic_area():
    Phantom, _, _, _, _ = _unified("Respiratory", Motion=False)

    body = np.count_nonzero(Phantom > 0) * D * D
    lungs = 2 * np.pi * 55 * 45

    # The thorax ellipse minus the two air-filled lungs, plus the solid bronchi.
    assert body > np.pi * 150 * 100 - lungs
    assert body < np.pi * 150 * 100


@pytest.mark.parametrize("SelectedPhantom", [
    "CardiacPhantom", "CardiacPhantomWithPlaque",
    "CardiacPhantomMultipleArteries", "SimpleLungWithTumor4D",
])
def test_respiratory_variants_produce_physical_attenuation(SelectedPhantom):
    Phantom, isSpectral, _, _ = CreatePhantom.CreateVoxelPhantom2DWithMotionRespiratory(
        NX, D, ORIGIN, NY, D, ORIGIN, 0.0, 0.01, 25, SelectedPhantom
    )

    assert isSpectral is False
    assert Phantom.shape == (NX * NY,)
    assert Phantom.min() == 0.0
    assert Phantom.max() <= MU_PLAQUE


def test_respiratory_variant_rejects_unknown_phantoms():
    with pytest.raises(ValueError, match="Unknown phantom"):
        CreatePhantom.CreateVoxelPhantom2DWithMotionRespiratory(
            NX, D, ORIGIN, NY, D, ORIGIN, 0.0, 0.01, 0, "Nope"
        )


def test_lung_tumor_phantom_moves_both_tumors():
    def phantom(nT):
        return CreatePhantom.CreateVoxelPhantom2DWithMotionRespiratory(
            NX, D, ORIGIN, NY, D, ORIGIN, 0.0, 1.0, nT, "SimpleLungWithTumor4D"
        )[0]

    # The breathing cycle is 4 s, so t = 1 s is the extreme of the sine.
    assert not np.array_equal(phantom(0), phantom(1))
    # One full cycle later the phantom repeats.
    assert np.allclose(phantom(0), phantom(4))


def test_cardiac_2d_with_motion_uses_the_contrast_option():
    def phantom(ContrastMedium):
        return CreatePhantom.CreateVoxelPhantomCardiac2DWithMotion(
            NX, D, ORIGIN, NY, D, ORIGIN, 0.0, 0.01, 25, ContrastMedium=ContrastMedium
        )[0]

    assert _area(phantom(False), MU_BLOOD) > 0.0
    assert _area(phantom(True), MU_BLOOD_CM) > 0.0
    assert _area(phantom(True), MU_BLOOD) == 0.0


def test_cardiac_2d_with_motion_validates_the_plaque_type():
    with pytest.raises(ValueError, match="Invalid PlaqueType"):
        CreatePhantom.CreateVoxelPhantomCardiac2DWithMotion(
            NX, D, ORIGIN, NY, D, ORIGIN, 0.0, 0.01, 0, PlaqueType="banana"
        )
