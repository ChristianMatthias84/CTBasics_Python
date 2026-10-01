import numpy as np

from ctbasics.reconstruction import Backprojection, ForwardProjection
from ctbasics.simulation import CreatePhantom


phantom, _ = CreatePhantom.create_voxel_phantom_2d(31, 1.0, -16.0, 31, 1.0, -16.0, "WaterPhantom")
beta = np.linspace(-0.2, 0.2, 9)
projections = ForwardProjection.ray_driven_forward_proj_2d_fan_multiple(
    20.0,
    -0.1,
    0.05,
    beta,
    -16.0,
    1.0,
    31,
    -16.0,
    1.0,
    31,
    phantom,
    9,
    4,
)
recon = Backprojection.wfbp2d_fully_pixel_driven(
    -0.5,
    0.5,
    2,
    4,
    -1.0,
    0.2,
    3,
    projections,
    -1.0,
    1.0,
    2,
    -1.0,
    1.0,
    2,
)

print(f"projection_count={projections.shape[0]}")
print(f"recon_shape={recon.shape}")
print(f"finite={np.isfinite(recon).all()}")
