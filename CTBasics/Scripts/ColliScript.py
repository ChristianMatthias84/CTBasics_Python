"""Collimation test driver for the Static CT geometry.

Port of Scripts/ColliScript.m. The MATLAB file is a function file, so the local
functions map to module level functions here.
"""
import numpy as np

from CTBasics.Descriptors.Sampling.FocalSpotSimulationStaticCT import FocalSpotSimulationStaticCT
from CTBasics.Descriptors.ScanDescriptor import ScanDescriptor
from CTBasics.Descriptors.ScannerGeometryTypes import ScannerGeometryTypes
from CTBasics.Descriptors.ScanTypes import ScanTypes
from CTBasics.LecturesAndTestScripts.CTRecon101 import _Save
from CTBasics.Reconstruction.ForwardProjection import ForwardProjection
from CTBasics.Visualization.StaticCTVisualizationFunctionalitites import (
    StaticCTVisualizationFunctionalitites,
)


def ColliScript(mode="StandardCollimator", Outputdirectory=None, ShowFigures=False,
                Descriptor=None):
    return RunCollimatorTestByMode(mode, Outputdirectory, ShowFigures, Descriptor)


def RunCollimatorTestByMode(mode, Outputdirectory=None, ShowFigures=False, Descriptor=None):
    if Descriptor is None:
        Descriptor = ScanDescriptor.Create(ScannerGeometryTypes.StaticCT, ScanTypes.Spiral)

    key = str(mode).lower()

    if key in ("standardcollimator", "standard", "ring", "old"):
        return RunLegacyCollimatorSingleDirectionTest(Descriptor, Outputdirectory, ShowFigures)
    if key in ("leaf", "leaftest", "new"):
        return RunLeafCollimatorSingleDirectionTest(Descriptor, Outputdirectory, ShowFigures)

    raise ValueError(f"Unknown mode: {mode}. Use 'StandardCollimator' or 'LeafTest'.")


def RunLegacyCollimatorSingleDirectionTest(aScanDescriptor, Outputdirectory=None,
                                           ShowFigures=False):
    focalspotsamples = FocalSpotSimulationStaticCT.CalculateFocalSpotSamplingPositions()

    CollimationOfOneProjection, ArrayColliForDifferentFSSamplePoints = (
        ForwardProjection.StaticCTCollimationSingleSourceSampled(
            aScanDescriptor, np.pi / 2, focalspotsamples
        )
    )

    aStaticCTGeo = aScanDescriptor.Geometry
    aImage = np.asarray(CollimationOfOneProjection).reshape(aStaticCTGeo.Ns, aStaticCTGeo.MDelta)

    if Outputdirectory is not None:
        _Save(Outputdirectory, "CollimationOfOneProjection", CollimationOfOneProjection)
        # MATLAB flattens the sample-point array column by column before writing.
        _Save(Outputdirectory, "ArrayColliForDifferentFSSamplePoints",
              np.asarray(ArrayColliForDifferentFSSamplePoints).T.ravel())

    if ShowFigures:
        _ShowCollimation(aImage, "Legacy ring-based static CT collimation")
        StaticCTVisualizationFunctionalitites.ShowStaticCTCollimation3DFromComputedData(
            aScanDescriptor, np.pi / 2, focalspotsamples, CollimationOfOneProjection,
            "Legacy ring-based static CT collimation"
        )

    return CollimationOfOneProjection, ArrayColliForDifferentFSSamplePoints


def RunLeafCollimatorSingleDirectionTest(aScanDescriptor, Outputdirectory=None,
                                         ShowFigures=False):
    focalspotsamples = FocalSpotSimulationStaticCT.CalculateFocalSpotSamplingPositions()
    Geometry = aScanDescriptor.Geometry
    alpha = np.pi / 2
    LeafConfig = CreateExampleLeafCollimatorConfig(Geometry)

    CollimationOfOneProjection, ArrayColliForDifferentFSSamplePoints = (
        ForwardProjection.StaticCTCollimationLeafSingleSourceSampled(
            aScanDescriptor, alpha, focalspotsamples, LeafConfig
        )
    )

    aImage = np.asarray(CollimationOfOneProjection).reshape(Geometry.Ns, Geometry.MDelta)

    if Outputdirectory is not None:
        _Save(Outputdirectory, "CollimationOfOneProjection_Leaf", CollimationOfOneProjection)
        _Save(Outputdirectory, "ArrayColliForDifferentFSSamplePoints_Leaf",
              np.asarray(ArrayColliForDifferentFSSamplePoints).T.ravel())

    if ShowFigures:
        _ShowCollimation(aImage, "Leaf-based static CT collimation")
        StaticCTVisualizationFunctionalitites.ShowStaticCTCollimation3DFromComputedDataLeaf(
            aScanDescriptor, alpha, focalspotsamples, CollimationOfOneProjection,
            LeafConfig, "MultiLeaf static CT collimation"
        )

    return CollimationOfOneProjection, ArrayColliForDifferentFSSamplePoints


def CreateExampleLeafCollimatorConfig(Geometry):
    NLeafs = 100
    zUpperBase = Geometry.zCollimatorUpper
    zLowerBase = Geometry.zCollimatorLower

    # The first half of the leafs stays at the nominal opening, the second half is closed
    # by half the collimator height from both sides.
    upperOffsets = -0.5 * (zUpperBase - zLowerBase) * np.ones(NLeafs)
    lowerOffsets = +0.5 * (zUpperBase - zLowerBase) * np.ones(NLeafs)
    upperOffsets[:NLeafs // 2] = 0
    lowerOffsets[:NLeafs // 2] = 0

    return {
        "N_collimatorLeafs": NLeafs,
        "zUpperPerLeaf": zUpperBase + upperOffsets,
        "zLowerPerLeaf": zLowerBase + lowerOffsets,
    }


def _ShowCollimation(aImage, title):
    import matplotlib.pyplot as plt

    plt.figure()
    plt.imshow(aImage, cmap="gray", aspect="auto")
    plt.title(title)
    plt.show()


if __name__ == "__main__":
    Collimation, _ = ColliScript()
    print(f"Collimation sum {np.sum(Collimation):.0f}")
