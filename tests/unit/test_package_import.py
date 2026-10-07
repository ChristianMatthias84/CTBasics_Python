def test_ctbasics_package_exposes_matlab_domains():
    import CTBasics

    for name in ("ScanDescriptor", "CreatePhantom", "ForwardProjection", "Backprojection", "Constants"):
        assert name in CTBasics.__all__
        assert hasattr(CTBasics, name)
