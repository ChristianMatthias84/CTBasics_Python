"""Real time imaging study: many short scan segments per reconstructed phase.

Port of Projects/Static CT/Projects CH/StaticCT_4_RealTimeImaging_Script_CH.m. The MATLAB
file is a copy of Projects JP/test_4DCT_prosp.m with a Poisson noise stage added and
different defaults (WaterWith4Arteries, 5 scan segments, 0.1 s rotation time). The port
therefore reuses the JP implementation and only supplies the CH defaults, which keeps a
single implementation of the shared pipeline.
"""
from ..ProjectsJP import test_4DCT_prosp
from ..ProjectsJP.test_4DCT_prosp import NSubsets_array  # noqa: F401  (kept for parity)


def main(Outputdirectory_base=None, reconType="180", ApplyMoireFilter=False,
         SimNoise=True, NSubsets=64, NRecons=8, N_I0=100000 / 5,
         SelectedPhantom="WaterWith4Arteries", Heartrate=60, Motion=True,
         Direction=0, MotionType="full", TRest=0, CM=False, Stent=False, Plaque=1,
         Rotation=0, isplot=True, phasesToPlot=(1, 3), isGIF=False, writeData=True,
         NScansPerRecon=5, TRot=0.1, RM=None, M=500, N2pi=1024, Nx=512, Ny=512,
         UseLinearCache=True, **options):
    return test_4DCT_prosp.main(
        Outputdirectory_base=Outputdirectory_base, reconType=reconType,
        ApplyMoireFilter=ApplyMoireFilter, SimNoise=SimNoise, N_I0=N_I0,
        NSubsets=NSubsets, NRecons=NRecons, SelectedPhantom=SelectedPhantom,
        Heartrate=Heartrate, Motion=Motion, Direction=Direction,
        MotionType=MotionType, TRest=TRest, CM=CM, Stent=Stent, Plaque=Plaque,
        Rotation=Rotation, isplot=isplot, phasesToPlot=phasesToPlot, isGIF=isGIF,
        writeData=writeData, NScansPerRecon=NScansPerRecon, TRot=TRot, RM=RM, M=M,
        N2pi=N2pi, Nx=Nx, Ny=Ny, UseLinearCache=UseLinearCache, **options)
