import numpy as np

from ctbasics.descriptors import ScanDescriptor, ScanTypes, ScannerGeometryTypes, FocalSpotSimulationStaticCT


class VisualizationTests:
    @staticmethod
    def create_default_static_ct_descriptor():
        descriptor = ScanDescriptor.create(ScannerGeometryTypes.StaticCT, ScanTypes.Spiral)
        descriptor.Geometry.MDelta = 8
        descriptor.Geometry.Ns = 4
        descriptor.Geometry.dDelta = 2.0 * np.pi / descriptor.Geometry.MDelta
        descriptor.Geometry.ds = 1.0
        descriptor.Geometry.soffset = -2.5
        descriptor.Geometry.s0 = -2.5 - descriptor.Geometry.Ns * descriptor.Geometry.ds - descriptor.Geometry.ds / 2.0
        return descriptor

    @staticmethod
    def resolve_static_ct_geometry(descriptor):
        return descriptor.Geometry

    @staticmethod
    def create_example_leaf_collimator_config(geometry):
        values = np.linspace(-2.5, 2.5, 100)
        return {
            "N_collimatorLeafs": 100,
            "zUpperPerLeaf": values.tolist(),
            "zLowerPerLeaf": values.tolist(),
        }
