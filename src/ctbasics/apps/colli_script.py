import numpy as np

from ctbasics.reconstruction.forward_projection import ForwardProjection


def run_colli_script(mode, descriptor=None):
    if descriptor is None:
        from ctbasics.descriptors import ScanDescriptor, ScannerGeometryTypes, ScanTypes
        descriptor = ScanDescriptor.create(ScannerGeometryTypes.StaticCT, ScanTypes.Spiral)
    focal_spots = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    return ForwardProjection.static_ct_collimation_single_source_sampled(descriptor, 0.0, focal_spots)[0], ForwardProjection.static_ct_collimation_single_source_sampled(descriptor, 0.0, focal_spots)[1]
