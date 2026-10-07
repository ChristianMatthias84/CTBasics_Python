"""Static visualization helper collection.

The figures are created with matplotlib; every method returns its axes so the callers
and the tests can inspect the plotted geometry instead of only looking at a window.
"""
import numpy as np

from ..Descriptors.Geometries.StaticCTBeam import StaticCTBeam
from ..Descriptors.Sampling.FocalSpotSimulationStaticCT import FocalSpotSimulationStaticCT
from ..Descriptors.ScanDescriptor import ScanDescriptor
from ..Descriptors.ScannerGeometryTypes import ScannerGeometryTypes
from ..Descriptors.ScanTypes import ScanTypes
from ..Reconstruction.ForwardProjection import ForwardProjection


def _Figure(name, projection=None):
    import matplotlib.pyplot as plt

    figure = plt.figure(name)
    axes = figure.add_subplot(projection=projection)
    axes.grid(True)
    return figure, axes


def _Axes3D(name):
    figure, axes = _Figure(name, projection="3d")
    axes.set_xlabel("x [mm]")
    axes.set_ylabel("y [mm]")
    axes.set_zlabel("z [mm]")
    axes.set_box_aspect((1, 1, 1))
    return figure, axes


class VisualizationTests:
    @staticmethod
    def DemoAll():
        """Runs every demo figure in the order of the MATLAB script."""
        VisualizationTests.ShowRebinVariables2D()
        VisualizationTests.ShowRebinnedGeometry3D()
        VisualizationTests.ShowStaticCTGeometry3D()
        VisualizationTests.ShowStaticCTCollimation3D()

    @staticmethod
    def ShowRingSketch3D():
        """Legacy sketch of a source ring, a detector cylinder and one fan ray."""
        alpha = np.arange(0, 2 * np.pi + 0.01, 0.01)
        Rdetector = 400
        xdetector = Rdetector * np.cos(alpha)
        ydetector = Rdetector * np.sin(alpha)

        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(-10, -50, 2)
        U, V = np.meshgrid(u, v)

        Rsource = 500
        xsource = Rsource * np.cos(alpha)
        ysource = Rsource * np.sin(alpha)

        _, axes = _Axes3D("Ring Sketch 3D")
        axes.set_xlim(-600, 600)
        axes.set_ylim(-600, 600)
        axes.set_zlim(-600, 600)

        axes.plot(xdetector, ydetector, np.full_like(xdetector, -10.0), "b", linewidth=1.2)
        axes.plot(xdetector, ydetector, np.full_like(xdetector, -50.0), "b", linewidth=1.2)
        axes.plot(xsource, ysource, np.zeros_like(xsource), "r", linewidth=1.2)

        axes.plot_surface(Rdetector * np.cos(U), Rdetector * np.sin(U), V,
                          alpha=0.2, linewidth=0, cmap="gray")

        ray = np.array([[-500.0, 0.0, 0.0], [400.0, 0.0, -10.0], [400.0, 0.0, -50.0]])
        # The ray triangle lies in the y=0 plane, so it needs an explicit polygon.
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        axes.add_collection3d(Poly3DCollection([ray], facecolor="r", edgecolor="r", alpha=0.25))

        axes.set_title("Legacy Ring Geometry Sketch")
        return axes

    @staticmethod
    def ShowRebinVariables2D():
        """Fan angle beta against the static CT source angle betaDash."""
        rs = 500
        rd = 400
        betaDash = np.arange(0, np.pi / 2 + 0.01, 0.01)
        beta = np.arctan(np.sin(betaDash) * rd / (np.cos(betaDash) * rd + rs))

        _, axes = _Figure("Rebin Variables 2D")
        axes.plot(betaDash, beta, linewidth=1.4, label=r"$\beta(\beta')$")
        axes.plot(betaDash, betaDash, "r--", linewidth=1.1, label=r"$\beta=\beta'$")
        axes.set_xlabel(r"$\beta'$ [rad]")
        axes.set_ylabel(r"$\beta$ [rad]")
        axes.legend(loc="upper left")
        axes.set_title("Geometry Variable Visualization")
        return axes

    @staticmethod
    def ShowRebinnedGeometry3D():
        """z position of every rebinned ray of a spiral scan over the channel coordinate."""
        zstart = 0
        pitch = 1.0
        colli = 64 * 0.6
        zrot = pitch * colli
        theta = 0
        betamax = 30 / 180 * np.pi
        RF = 600
        ximin = -RF * np.sin(betamax)
        ximax = +RF * np.sin(betamax)
        Nxi = 30
        xi = np.linspace(ximin, ximax, Nxi)
        alpha = theta - np.arcsin(xi / RF)
        y = 0
        x = 0
        ns = np.arange(0, 17)[:, None]
        Ns = 16
        deltas = 1

        L = np.sqrt(RF * RF - xi * xi) - x * np.sin(theta) + y * np.cos(theta)
        z = zstart + zrot / (2 * np.pi) * alpha + L * deltas / (2 * RF) * (2 * ns - Ns - 1)

        _, axes = _Figure("Rebinned Geometry 2D Lines")
        axes.plot(xi, z.T, linewidth=1.0)
        axes.set_ylim(-20, 20)
        axes.set_xlim(-500, 500)
        axes.set_xlabel(r"$\xi$ [mm]")
        axes.set_ylabel("z [mm]")
        axes.set_title("Rebinned Geometry Lines")

        xiSurf = np.broadcast_to(xi, z.shape)

        _, surface = _Figure("Rebinned Geometry Surface")
        surface.pcolormesh(xiSurf, z, np.zeros_like(xiSurf), edgecolors="face", alpha=0.4)
        surface.set_ylim(-20, 20)
        surface.set_xlim(-500, 500)
        surface.set_xlabel(r"$\xi$ [mm]")
        surface.set_ylabel("z [mm]")
        surface.set_title("Rebinned Geometry Surface (Top View)")
        return axes, surface

    @staticmethod
    def ShowStaticCTGeometry3D(scanDescriptorOrGeometry=None):
        """Source ring, detector surface and the two collimator rings of a static CT."""
        if scanDescriptorOrGeometry is None:
            geo = VisualizationTests.CreateDefaultStaticCTDescriptor().Geometry
        else:
            geo = VisualizationTests.ResolveStaticCTGeometry(scanDescriptorOrGeometry)

        delta = geo.delta0 + np.arange(1, geo.MDelta + 1) * geo.dDelta
        zRows = geo.s0 + np.arange(1, geo.Ns + 1) * geo.ds

        xSourceRing = geo.RF * np.sin(delta)
        ySourceRing = -geo.RF * np.cos(delta)

        xDetectorRing = geo.RD * np.cos(delta)
        yDetectorRing = geo.RD * np.sin(delta)

        DeltaGrid, ZGrid = np.meshgrid(delta, zRows)
        XDetector = geo.RD * np.cos(DeltaGrid)
        YDetector = geo.RD * np.sin(DeltaGrid)

        _, axes = _Axes3D("StaticCT Geometry 3D")

        axes.plot(xSourceRing, ySourceRing, np.zeros_like(delta), "r", linewidth=1.4,
                  label="Source ring")
        axes.plot(xDetectorRing, yDetectorRing, np.full_like(delta, geo.zCollimatorUpper),
                  "b--", linewidth=0.9, label="Detector envelope (upper z)")
        axes.plot(xDetectorRing, yDetectorRing, np.full_like(delta, geo.zCollimatorLower),
                  "b--", linewidth=0.9, label="Detector envelope (lower z)")

        axes.plot_surface(XDetector, YDetector, ZGrid, alpha=0.2, linewidth=0,
                          color=(0.2, 0.4, 0.9))

        xCol = geo.RC * np.cos(delta)
        yCol = geo.RC * np.sin(delta)
        axes.plot(xCol, yCol, np.full_like(delta, geo.zCollimatorUpper), "k", linewidth=1.6,
                  label="Upper collimator ring")
        axes.plot(xCol, yCol, np.full_like(delta, geo.zCollimatorLower), "k", linewidth=1.6,
                  label="Lower collimator ring")

        axes.plot([0, 0], [0, 0], [zRows.min(), max(0.0, zRows.max())], "k:", linewidth=1.0)

        axes.legend(loc="upper left")
        axes.set_title("StaticCT Beam Geometry with Ring Collimators")
        return axes

    @staticmethod
    def ShowStaticCTCollimation3D(scanDescriptor=None, projectionAngle=None,
                                  focalSpotSamples=None, computeIllumination=True):
        """Collimator rings plus the illuminated detector of one projection.

        Returns (collimationOfOneProjection, detectorImage).
        """
        if scanDescriptor is None:
            scanDescriptor = VisualizationTests.CreateDefaultStaticCTDescriptor()
        if projectionAngle is None:
            projectionAngle = np.pi / 2
        if focalSpotSamples is None:
            focalSpotSamples = FocalSpotSimulationStaticCT.CalculateFocalSpotSamplingPositions()

        geo = VisualizationTests.ResolveStaticCTGeometry(scanDescriptor)
        collimationOfOneProjection = None
        detectorImage = None

        if computeIllumination:
            collimationOfOneProjection, _ = ForwardProjection.StaticCTCollimationSingleSourceSampled(
                scanDescriptor, projectionAngle, focalSpotSamples
            )
            detectorImage = np.asarray(collimationOfOneProjection).reshape(geo.Ns, geo.MDelta)

        VisualizationTests.ShowStaticCTCollimation3DFromComputedData(
            scanDescriptor, projectionAngle, focalSpotSamples, collimationOfOneProjection,
            "StaticCT Collimator and Detector Illumination (3D)"
        )

        return collimationOfOneProjection, detectorImage

    @staticmethod
    def ShowStaticCTCollimation3DFromComputedData(scanDescriptor=None, projectionAngle=None,
                                                  focalSpotSamples=None,
                                                  collimationOfOneProjection=None,
                                                  plotTitle=None):
        """Draws the single-ring collimator and colours the detector by the illumination."""
        if scanDescriptor is None:
            scanDescriptor = VisualizationTests.CreateDefaultStaticCTDescriptor()
        if projectionAngle is None:
            projectionAngle = np.pi / 2
        if focalSpotSamples is None:
            focalSpotSamples = FocalSpotSimulationStaticCT.CalculateFocalSpotSamplingPositions()
        if plotTitle is None:
            plotTitle = "StaticCT Collimator and Detector Illumination (3D)"

        geo = VisualizationTests.ResolveStaticCTGeometry(scanDescriptor)
        focalSpotSamples = np.atleast_2d(np.asarray(focalSpotSamples, dtype=float))

        hasIllumination = collimationOfOneProjection is not None
        detectorImage = (np.asarray(collimationOfOneProjection).reshape(geo.Ns, geo.MDelta)
                         if hasIllumination else None)

        delta = geo.delta0 + np.arange(1, geo.MDelta + 1) * geo.dDelta
        zRows = geo.s0 + np.arange(1, geo.Ns + 1) * geo.ds

        _, axes = _Axes3D("StaticCT Collimation 3D")

        xCol = geo.RC * np.cos(delta)
        yCol = geo.RC * np.sin(delta)
        axes.plot(xCol, yCol, np.full_like(delta, geo.zCollimatorUpper), "k", linewidth=1.8,
                  label="Upper collimator ring")
        axes.plot(xCol, yCol, np.full_like(delta, geo.zCollimatorLower), "k", linewidth=1.8,
                  label="Lower collimator ring")

        sourceCenter, er, et = _SourceFrame(geo, projectionAngle)
        axes.plot([sourceCenter[0]], [sourceCenter[1]], [sourceCenter[2]], "ro",
                  markersize=7, label="Source center")

        sampleIdx = _SampleIndices(focalSpotSamples.shape[0], 35)
        samples = np.array([_SourceSample(sourceCenter, er, et, focalSpotSamples[i])
                            for i in sampleIdx])
        axes.plot(samples[:, 0], samples[:, 1], samples[:, 2], ".", color=(0.9, 0.3, 0.3),
                  markersize=10, linestyle="none", label="Focal spot samples")

        _PlotRayBundle(axes, geo, delta, zRows, samples[:min(len(samples), 8)])
        _PlotDetector(axes, geo, delta, zRows, detectorImage)

        axes.set_title(plotTitle)
        axes.legend(loc="upper left")
        return detectorImage

    @staticmethod
    def ShowStaticCTCollimation3DFromComputedDataLeaf(scanDescriptor=None, projectionAngle=None,
                                                      focalSpotSamples=None,
                                                      collimationOfOneProjection=None,
                                                      LeafConfig=None, plotTitle=None):
        """Same view as the single-ring variant but with one collimator leaf per segment."""
        if scanDescriptor is None:
            scanDescriptor = VisualizationTests.CreateDefaultStaticCTDescriptor()
        if projectionAngle is None:
            projectionAngle = np.pi / 2
        if focalSpotSamples is None:
            focalSpotSamples = FocalSpotSimulationStaticCT.CalculateFocalSpotSamplingPositions()
        if LeafConfig is None:
            raise ValueError("LeafConfig is required for multi-leaf visualization.")
        if plotTitle is None:
            plotTitle = "MultiLeaf static CT collimation"

        geo = VisualizationTests.ResolveStaticCTGeometry(scanDescriptor)
        focalSpotSamples = np.atleast_2d(np.asarray(focalSpotSamples, dtype=float))

        hasIllumination = collimationOfOneProjection is not None
        detectorImage = (np.asarray(collimationOfOneProjection).reshape(geo.Ns, geo.MDelta)
                         if hasIllumination else None)

        delta = geo.delta0 + np.arange(1, geo.MDelta + 1) * geo.dDelta
        zRows = geo.s0 + np.arange(1, geo.Ns + 1) * geo.ds

        NLeafs = LeafConfig.N_collimatorLeafs
        zUpperPerLeaf = np.asarray(LeafConfig.zUpperPerLeaf, dtype=float).ravel()
        zLowerPerLeaf = np.asarray(LeafConfig.zLowerPerLeaf, dtype=float).ravel()

        _, axes = _Axes3D("StaticCT Multi-Leaf Collimation 3D")

        # Angular bounds consistent with the detector-element to leaf mapping.
        for k in range(1, NLeafs + 1):
            mStart = int((k - 1) * geo.MDelta // NLeafs) + 1
            mEnd = max(int(k * geo.MDelta // NLeafs), mStart)

            deltaSegment = np.array([delta[mStart - 1], delta[mEnd - 1]])
            xSeg = geo.RC * np.cos(deltaSegment)
            ySeg = geo.RC * np.sin(deltaSegment)

            axes.plot(xSeg, ySeg, np.full(2, zUpperPerLeaf[k - 1]), "k", linewidth=1.4,
                      label="Leaf upper edges" if k == 1 else None)
            axes.plot(xSeg, ySeg, np.full(2, zLowerPerLeaf[k - 1]), "k", linewidth=1.0,
                      label="Leaf lower edges" if k == 1 else None)

        sourceCenter, er, et = _SourceFrame(geo, projectionAngle)
        axes.plot([sourceCenter[0]], [sourceCenter[1]], [sourceCenter[2]], "ro",
                  markersize=7, label="Source center")

        sampleIdx = _SampleIndices(focalSpotSamples.shape[0], 35)
        samples = np.array([_SourceSample(sourceCenter, er, et, focalSpotSamples[i])
                            for i in sampleIdx])
        axes.plot(samples[:, 0], samples[:, 1], samples[:, 2], ".", color=(0.9, 0.3, 0.3),
                  markersize=10, linestyle="none", label="Focal spot samples")

        _PlotRayBundle(axes, geo, delta, zRows, samples[:min(len(samples), 8)])
        _PlotDetector(axes, geo, delta, zRows, detectorImage)

        axes.set_title(plotTitle)
        axes.legend(loc="upper left")
        return detectorImage

    @staticmethod
    def CreateDefaultStaticCTDescriptor():
        return ScanDescriptor.Create(ScannerGeometryTypes.StaticCT, ScanTypes.Spiral)

    @staticmethod
    def ResolveStaticCTGeometry(scanDescriptorOrGeometry):
        geo = (scanDescriptorOrGeometry.Geometry
               if isinstance(scanDescriptorOrGeometry, ScanDescriptor)
               else scanDescriptorOrGeometry)

        if not isinstance(geo, StaticCTBeam):
            raise ValueError("Expected `StaticCTBeam` geometry.")

        return geo


def _SourceFrame(geo, projectionAngle):
    """Source centre plus the radial and the tangential unit vector of one projection."""
    sourceCenter = np.array([geo.RF * np.sin(projectionAngle),
                             -geo.RF * np.cos(projectionAngle), 0.0])
    er = np.array([np.sin(projectionAngle), -np.cos(projectionAngle), 0.0])
    et = np.array([np.cos(projectionAngle), np.sin(projectionAngle), 0.0])
    return sourceCenter, er, et


def _SourceSample(sourceCenter, er, et, dF):
    """Focal spot sample offset: dF is (z, radial, tangential)."""
    return sourceCenter + dF[1] * er + dF[2] * et + np.array([0.0, 0.0, dF[0]])


def _SampleIndices(count, maximum):
    return np.unique(np.round(np.linspace(1, count, min(count, maximum))).astype(int) - 1)


def _PlotRayBundle(axes, geo, delta, zRows, samples):
    """Draws rays from a few focal spot samples to the first and the last detector row."""
    nDeltaRay = min(80, geo.MDelta)
    deltaIdx = _SampleIndices(geo.MDelta, nDeltaRay)

    labelled = False
    for sample in samples:
        for d in delta[deltaIdx]:
            for row in (zRows[0], zRows[-1]):
                target = (geo.RD * np.cos(d), geo.RD * np.sin(d), row)
                axes.plot([sample[0], target[0]], [sample[1], target[1]], [sample[2], target[2]],
                          color=(0.2, 0.6, 0.2, 0.15),
                          label=None if labelled else "Ray bundle")
                labelled = True


def _PlotDetector(axes, geo, delta, zRows, detectorImage):
    import matplotlib.pyplot as plt

    DeltaGrid, ZGrid = np.meshgrid(delta, zRows)
    XDetector = geo.RD * np.cos(DeltaGrid)
    YDetector = geo.RD * np.sin(DeltaGrid)

    if detectorImage is None:
        axes.plot_surface(XDetector, YDetector, ZGrid, linewidth=0, alpha=0.2,
                          color=(0.2, 0.4, 0.9))
        return

    span = detectorImage.max() - detectorImage.min()
    colors = plt.get_cmap("turbo")(
        (detectorImage - detectorImage.min()) / span if span > 0 else np.zeros_like(detectorImage)
    )
    axes.plot_surface(XDetector, YDetector, ZGrid, facecolors=colors, linewidth=0, alpha=0.82)
