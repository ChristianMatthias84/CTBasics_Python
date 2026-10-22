"""Synchronized segmented single source experiment ("to dual").

Port of Projects/Static CT/Projects AH/DualSource/
DualSource_CardiacMotion4DCT_SegmentedSingleSource_Experiment_to_dual.m.

The MATLAB script is a copy of DualSource_CardiacMotion4DCT_SegmentedSingleSource.m with
two changes: Nsources = 20, and every segment is centred on Shift_GT (window
Shift_GT +- Shift_Multi) instead of on its own slot of the single source read window; the
timing plot is drawn in its synchronized mode. Output folder, file names and metrics are
identical, so the port delegates to the segmented single source module.
"""
from CTBasics.Projects.StaticCT.ProjectsAH.DualSource.DualSource_CardiacMotion4DCT_SegmentedSingleSource import main as _SegmentedSingleSource


def main(Outputdirectory=None, *, Nsources=20, **kwargs):
    """Runs the synchronized variant; all other keywords as in the segmented main."""
    return _SegmentedSingleSource(Outputdirectory, Nsources=Nsources,
                                  isSynchronizedSegments=True, **kwargs)

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    main()
    plt.show()
