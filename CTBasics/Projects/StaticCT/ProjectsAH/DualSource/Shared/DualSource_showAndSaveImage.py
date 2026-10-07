"""Shows one reconstruction and optionally writes it as HU raw data.

Port of Projects/Static CT/Projects AH/DualSource/Shared/DualSource_showAndSaveImage.m.
"""
from pathlib import Path

import numpy as np

from ......Reconstruction.Constants import Constants


def DualSource_showAndSaveImage(dataVector, dims, Outputdirectory, name, doSave, params,
                                roiMask=None, ShowFigures=False):
    dataVector = np.asarray(dataVector, dtype=float).ravel()
    aImage = dataVector.reshape(dims[1], dims[0])

    if ShowFigures:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.imshow(aImage, cmap="gray")
        if roiMask is not None:
            plt.contour(np.asarray(roiMask), [1], colors="r", linewidths=2)
        plt.title(name)
        plt.show()

    filename = None
    if doSave:
        phantomTag = params.get("PhantomTag") or "NA"

        filename = Path(Outputdirectory) / (
            f"{name}_HR{int(params['Heartrate'])}"
            f"_TRot{round(params['TRot'] * 1000)}ms"
            f"_N{int(params['Npar'])}"
            f"_PS{round(params['Shift_GT'] * 1000)}"
            f"_Ph_{phantomTag}"
        )
        filename.parent.mkdir(parents=True, exist_ok=True)

        dataVectorHU = (dataVector - Constants.MuW) / Constants.MuW * 1000
        filename.write_bytes(np.asarray(dataVectorHU, dtype=np.float64).tobytes())

    return aImage, (str(filename) if filename is not None else None)
