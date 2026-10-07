"""Phantom geometry validated against analytic ellipse areas and MATLAB material codes."""
import numpy as np
import pytest

from CTBasics.Reconstruction import Constants
from CTBasics.Simulation import CreatePhantom, Materials

# Centred 500 mm field of view, matching the CTRecon101 simulation geometry.
RM = 250.0
N = 256
D = 2 * RM / N
ORIGIN = -RM - D / 2.0


def _phantom(name, n=N, d=D, origin=ORIGIN):
    phantom, is_spectral = CreatePhantom.CreateVoxelPhantom2D(n, d, origin, n, d, origin, name)
    return phantom.reshape(n, n), is_spectral


def test_water_phantom_is_a_150_mm_disk_of_water():
    image, is_spectral = _phantom("WaterPhantom")

    assert not is_spectral
    assert set(np.unique(image)) == {0.0, Constants.MuW}

    area = np.count_nonzero(image) * D * D
    assert np.isclose(area, np.pi * 150.0 * 150.0, rtol=0.01)


def test_simple_ellipse_has_the_expected_semi_axes():
    image, _ = _phantom("SimpleEllipse")

    area = np.count_nonzero(image) * D * D
    assert np.isclose(area, np.pi * 150.0 * 100.0, rtol=0.01)

    # Semi-axis a = 150 mm along x and b = 100 mm along y through the centre.
    row = image[np.argmin(np.abs(ORIGIN + np.arange(1, N + 1) * D)), :]
    column = image[:, np.argmin(np.abs(ORIGIN + np.arange(1, N + 1) * D))]
    assert np.isclose(np.count_nonzero(row) * D, 300.0, atol=2 * D)
    assert np.isclose(np.count_nonzero(column) * D, 200.0, atol=2 * D)


def test_simple_ellipse_with_hips_paints_bone_over_water():
    image, is_spectral = _phantom("SimpleEllipseWithHips")

    assert not is_spectral
    assert set(np.unique(image)) == {0.0, Constants.MuW, 1.8 * Constants.MuW}

    bone_area = np.count_nonzero(image == 1.8 * Constants.MuW) * D * D
    assert np.isclose(bone_area, 2 * np.pi * 25.0 * 20.0, rtol=0.02)

    total_area = np.count_nonzero(image) * D * D
    assert np.isclose(total_area, np.pi * 150.0 * 100.0, rtol=0.01)


def test_hip_with_metal_layers_water_bone_and_metal():
    image, _ = _phantom("HipWithMetal")

    assert set(np.unique(image)) == {0.0, Constants.MuW, 1.8 * Constants.MuW, 5 * Constants.MuW}

    metal_area = np.count_nonzero(image == 5 * Constants.MuW) * D * D
    assert np.isclose(metal_area, 2 * np.pi * 10.0 * 10.0, rtol=0.03)

    # Metal sits inside the bone inserts, so bone area is reduced by exactly the metal area.
    bone_area = np.count_nonzero(image == 1.8 * Constants.MuW) * D * D
    assert np.isclose(bone_area + metal_area, 2 * np.pi * 25.0 * 20.0, rtol=0.02)


def test_spectral_phantoms_store_material_codes_not_attenuation():
    image, is_spectral = _phantom("SpectralHipWithMetal")

    assert is_spectral
    assert set(np.unique(image)) == {
        0.0,
        float(Materials.Nyst_water),
        float(Materials.Nyst_CorticalBone),
        float(Materials.Nyst_aluminum),
    }

    centre = N // 2
    assert image[centre, centre] == Materials.Nyst_water


def test_spectral_metal_only_has_no_surrounding_water():
    image, is_spectral = _phantom("SpectralMetalOnly")

    assert is_spectral
    assert set(np.unique(image)) == {0.0, float(Materials.Nyst_Iron)}

    area = np.count_nonzero(image) * D * D
    assert np.isclose(area, 2 * np.pi * 15.0 * 15.0, rtol=0.02)


@pytest.mark.parametrize(
    "name,radius",
    [("WaterPhantomSpectral30", 150.0), ("WaterPhantomSpectral40", 200.0)],
)
def test_spectral_water_disks_have_the_documented_radius(name, radius):
    image, is_spectral = _phantom(name)

    assert is_spectral
    area = np.count_nonzero(image) * D * D
    assert np.isclose(area, np.pi * radius * radius, rtol=0.01)


def test_cortical_bone_insert_is_centred_and_18_mm():
    image, _ = _phantom("WaterPhantomSpectral40_CBinsert")

    insert = image == float(Materials.Nyst_CorticalBone)
    area = np.count_nonzero(insert) * D * D
    assert np.isclose(area, np.pi * 18.0 * 18.0, rtol=0.05)

    rows, cols = np.nonzero(insert)
    assert abs(rows.mean() - (N - 1) / 2.0) < 1.0
    assert abs(cols.mean() - (N - 1) / 2.0) < 1.0


def test_unknown_phantom_name_is_rejected():
    with pytest.raises(ValueError, match="Unknown phantom"):
        CreatePhantom.CreateVoxelPhantom2D(16, 1.0, -8.0, 16, 1.0, -8.0, "DoesNotExist")


def test_3d_phantom_must_not_be_requested_through_the_2d_entry_point():
    with pytest.raises(ValueError, match="use CreateVoxelPhantom3D"):
        CreatePhantom.CreateVoxelPhantom2D(16, 1.0, -8.0, 16, 1.0, -8.0, "SimpleEllipseWithHips3D")


def test_simple_ellipse_with_hips_3d_tapers_the_right_hip_with_slice_index():
    Nz, dz = 12, 2.0
    phantom, is_spectral = CreatePhantom.CreateVoxelPhantom3D(
        "SimpleEllipseWithHips3D", None,
        N, D, ORIGIN, N, D, ORIGIN, Nz, dz, -Nz * dz / 2.0, 1, Nz,
    )
    volume = phantom.reshape(Nz, N, N)

    assert not is_spectral

    # The right hip shrinks as a = 25 - nz*0.5, so its area must decrease monotonically.
    bone = 1.8 * Constants.MuW
    right_half = volume[:, :, N // 2:]
    areas = [np.count_nonzero(right_half[nz] == bone) for nz in range(Nz)]
    assert areas[0] > areas[-1]

    # Slices outside [zstart, zstop] stay empty.
    empty, _ = CreatePhantom.CreateVoxelPhantom3D(
        "SimpleEllipseWithHips3D", None,
        N, D, ORIGIN, N, D, ORIGIN, Nz, dz, -Nz * dz / 2.0, 1, 3,
    )
    assert np.count_nonzero(empty.reshape(Nz, N, N)[5]) == 0


def test_3d_spine_alternates_between_vertebra_and_gap():
    Nz, dz = 10, 2.0
    phantom, _ = CreatePhantom.CreateVoxelPhantom3D(
        "SimpleEllipseWithHips3D", None,
        N, D, ORIGIN, N, D, ORIGIN, Nz, dz, -Nz * dz / 2.0, 1, Nz,
    )
    volume = phantom.reshape(Nz, N, N)

    # The spine sits at y = -50 mm; count bone voxels in that band per slice.
    y = ORIGIN + np.arange(1, N + 1) * D
    band = (y > -62.0) & (y < -38.0)
    counts = [np.count_nonzero(volume[nz][band] == 1.8 * Constants.MuW) for nz in range(Nz)]

    # height = 5 mm with dz = 2 mm toggles the segment every third slice.
    assert max(counts) > 0
    assert min(counts) == 0
