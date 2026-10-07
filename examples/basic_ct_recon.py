import numpy as np

from CTBasics.Reconstruction import Backprojection, ForwardProjection
from CTBasics.Simulation import CreatePhantom


phantom, _ = CreatePhantom.CreateVoxelPhantom2D(31, 1.0, -16.0, 31, 1.0, -16.0, "WaterPhantom")
beta = np.linspace(-0.2, 0.2, 9)
projections = ForwardProjection.RayDrivenForwardProj2DFanMultiple(
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
recon = Backprojection.WFBP2DFullyPixelDriven(
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

print({"projections": projections.shape, "recon": recon.shape})
