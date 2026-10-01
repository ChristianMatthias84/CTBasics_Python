from ctbasics.descriptors import ScanDescriptor, ScanTypes, ScannerGeometryTypes


def test_scan_descriptor_factory_creates_fan_spiral():
    descriptor = ScanDescriptor.create(ScannerGeometryTypes.Fan, ScanTypes.Spiral)

    assert descriptor.Geometry.BeamType == ScannerGeometryTypes.Fan
    assert descriptor.Parameter.ScanType == ScanTypes.Spiral
    assert descriptor.Geometry.Sampling.N2pi == 512
    assert descriptor.Parameter.Geometry is descriptor.Geometry


def test_scan_descriptor_copy_detaches_nested_objects():
    descriptor = ScanDescriptor.create(ScannerGeometryTypes.Fan, ScanTypes.Spiral)
    copied = descriptor.copy()

    assert copied is not descriptor
    assert copied.Geometry is not descriptor.Geometry
    assert copied.Parameter is not descriptor.Parameter
    assert copied.Parameter.Geometry is copied.Geometry
