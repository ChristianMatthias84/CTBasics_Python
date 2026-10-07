"""End-to-end check of the CTRecon101 lecture script against expected HU values."""
import numpy as np
import pytest

from CTBasics.LecturesAndTestScripts import CTRecon101
from CTBasics.Reconstruction import Constants, Kernels


@pytest.fixture(scope="module")
def recon():
    # Npar = 180 keeps the runtime low while still covering the full half turn.
    return CTRecon101.main(Npar=180)


def _roi_mean(values, phantom, mu):
    """Mean over voxels of one material, eroded to avoid partial volume at the edges."""
    mask = (phantom == mu).reshape(256, 256)
    interior = mask.copy()
    for _ in range(3):
        interior &= np.roll(mask, 1, 0) & np.roll(mask, -1, 0) & np.roll(mask, 1, 1) & np.roll(mask, -1, 1)
        mask = interior.copy()
    return values.reshape(256, 256)[interior].mean(), interior.sum()


def test_ctrecon101_reconstructs_water_at_zero_hu(recon):
    mean, count = _roi_mean(recon["ReconstructedImageHU"], recon["Voxelphantom"], Constants.MuW)

    assert count > 1000
    assert abs(mean) < 5.0


def test_ctrecon101_reconstructs_bone_at_800_hu(recon):
    mean, count = _roi_mean(recon["ReconstructedImageHU"], recon["Voxelphantom"], 1.8 * Constants.MuW)

    # The phantom defines bone as 1.8 * MuW, which is +800 HU by construction.
    assert count > 100
    assert abs(mean - 800.0) < 25.0


def test_ctrecon101_reconstructs_air_near_minus_1000_hu(recon):
    mean, _ = _roi_mean(recon["ReconstructedImageHU"], recon["Voxelphantom"], 0.0)

    # Residual cupping from the finite field of measurement keeps this slightly above -1000.
    assert -1000.0 <= mean < -950.0


def test_ctrecon101_pipeline_shapes_and_kernel_choice(recon):
    geometry = recon["Geometry"]
    M, Npar, Nx, Ny = geometry["M"], geometry["Npar"], geometry["Nx"], geometry["Ny"]

    assert recon["Projections"].shape == (M * Npar,)
    assert recon["FilteredProjections"].shape == (M * Npar,)
    assert recon["ReconstructedImage"].shape == (Nx * Ny,)
    assert not recon["isSpectral"]

    # Filtering must remove the DC offset that the raw line integrals carry.
    assert recon["Projections"].mean() > 0.0
    assert abs(recon["FilteredProjections"].mean()) < 0.05 * recon["Projections"].mean()


def test_ctrecon101_shepp_logan_kernel_gives_a_comparable_reconstruction():
    result = CTRecon101.main(Npar=180, Reconkernel=Kernels.SheppLogan)

    mean, _ = _roi_mean(result["ReconstructedImageHU"], result["Voxelphantom"], Constants.MuW)
    assert abs(mean) < 10.0


def test_ctrecon101_rejects_an_unknown_kernel():
    with pytest.raises(ValueError, match="Unknown reconstruction kernel"):
        CTRecon101.main(Npar=8, Reconkernel="NoSuchKernel")


def test_ctrecon101_writes_matlab_compatible_binaries(tmp_path, recon):
    CTRecon101.main(Npar=8, Outputdirectory=tmp_path)

    for name in ("Phantom", "Projections2DParallelBeam", "Projections2DFiltered",
                 "Reconstruction", "ReconstructionHU"):
        assert (tmp_path / name).is_file()

    # MATLAB reads these with fread(..., 'double').
    phantom = np.fromfile(tmp_path / "Phantom", dtype=np.float64)
    assert phantom.shape == (256 * 256,)
