"""Set of helper functions for IO and data manipulation.

Port of ReadAndWrite/IOHelperFunctions.m (Author: Jonas Hajek). MATLAB method names are
kept verbatim, including the mixed snake_case/PascalCase naming of the original.
"""
import os
from pathlib import Path

import numpy as np

from ..Memory.Allocation import Allocation
from ..Memory.Precision import Precision
from ..Memory.TypeCast import TypeCast

_CONFIG_ENV = "CTBASICS_CONFIG"


def _UserPath() -> Path:
    """MATLAB's `userpath`; overridable so the port is not tied to a MATLAB installation."""
    configured = os.environ.get(_CONFIG_ENV)
    if configured:
        return Path(configured)
    return Path.home() / "Documents" / "MATLAB"


class IOHelperFunctions:
    def __init__(self, ConfigFile=None):
        if ConfigFile is None:
            ConfigFile = _UserPath() / "CTBasicsConfig.txt"
        ConfigFile = Path(ConfigFile)

        if not ConfigFile.is_file():
            raise FileNotFoundError(
                f"The folder {ConfigFile.parent.as_posix()} must contain a config file "
                f"CTBasicsConfig.txt which at least contains valid global path and output directory. "
                f"Set {_CONFIG_ENV} to point at the folder holding it."
            )

        CTBasicsConfig = ConfigFile.read_text(encoding="utf-8").splitlines()

        self.GlobalPath = Path(CTBasicsConfig[1].strip())
        self.GlobalDataPath = self.GlobalPath / "Data"

        if not self.GlobalDataPath.is_dir():
            raise FileNotFoundError(
                f"The folder {self.GlobalDataPath.as_posix()} which was set as data folder does not exist. "
                f"Can it be that the folder structure of the project has changed? "
                f"Adapt the config file accordingly"
            )

        self.GlobalOutputFolderPath = Path(CTBasicsConfig[3].strip())

        if not self.GlobalOutputFolderPath.is_dir():
            raise FileNotFoundError(
                f"The folder {self.GlobalOutputFolderPath.as_posix()} which was set as output folder "
                f"does not exist. you need to create it. Adapt the config file accordingly"
            )

        self.TfsPath = Path(CTBasicsConfig[5].strip()) if len(CTBasicsConfig) > 5 and CTBasicsConfig[5].strip() else None

    # ---------------------------------------------------------------- folders

    @staticmethod
    def get_folder_content(m_path):
        path = Path(m_path)
        dirNames = sorted(entry.name for entry in path.iterdir() if entry.is_dir())
        fileNames = sorted(entry.name for entry in path.iterdir() if not entry.is_dir())
        return fileNames, dirNames

    @staticmethod
    def get_all_dicom_folders(root, folder_list=None):
        """Returns every subfolder containing at least one .ima or .dcm file."""
        if folder_list is None:
            folder_list = []

        fileNames, dirNames = IOHelperFunctions.get_folder_content(root)

        if (IOHelperFunctions.contains(fileNames, ".ima", "IgnoreCase", True)
                or IOHelperFunctions.contains(fileNames, ".dcm", "IgnoreCase", True)):
            folder_list.append(str(root))

        for name in dirNames:
            folder_list = IOHelperFunctions.get_all_dicom_folders(Path(root) / name, folder_list)

        return folder_list

    @staticmethod
    def contains(str1, str2, stringIgnoreCase="", IgnoreCase=False):
        """Checks whether `str1` (a string or a list of strings) contains `str2`."""
        if IgnoreCase and stringIgnoreCase == "IgnoreCase":
            str2 = str2.lower()
            if isinstance(str1, str):
                str1 = str1.lower()
            else:
                str1 = [str(item).lower() for item in str1]

        if isinstance(str1, str):
            return str2 in str1
        return any(str2 in str(item) for item in str1)

    # ------------------------------------------------------------------ dicom

    @staticmethod
    def read_Volume_with_Dicom(folder_path):
        """Returns (HU volume, headers, stored values); slices are ordered by InstanceNumber."""
        import pydicom

        fileNames, _ = IOHelperFunctions.get_folder_content(folder_path)
        z = len(fileNames)

        info = pydicom.dcmread(Path(folder_path) / fileNames[0])

        V_HU = Allocation.zeros(int(info.Rows), int(info.Columns), z)
        V_mapped = Allocation.zeros(int(info.Rows), int(info.Columns), z)
        V_info = [None] * z

        for i in range(1, z + 1):
            info = pydicom.dcmread(Path(folder_path) / fileNames[i - 1])
            idx = int(info.InstanceNumber)

            V_mapped[:, :, idx - 1] = info.pixel_array.astype(np.int16)
            V_HU[:, :, idx - 1] = V_mapped[:, :, idx - 1] * float(info.RescaleSlope) + float(info.RescaleIntercept)
            V_info[idx - 1] = info

        return V_HU, V_info, V_mapped

    @staticmethod
    def read_Dicom_Slice(filename):
        import pydicom

        info = pydicom.dcmread(filename)

        V_mapped = info.pixel_array.astype(np.int16)
        V_HU = V_mapped * float(info.RescaleSlope) + float(info.RescaleIntercept)

        return V_HU, info, V_mapped

    @staticmethod
    def saveDicomVolume(folder_path, V, info):
        z = len(info)

        fol_name = str(info[0].SeriesDescription)
        target = Path(folder_path) / fol_name
        target.mkdir(parents=True, exist_ok=True)

        for i in range(1, z + 1):
            dataset = info[i - 1]
            dataset.PixelData = np.ascontiguousarray(V[:, :, i - 1].astype(np.int16)).tobytes()
            dataset.save_as(target / f"ct_file{i}.dcm", enforce_file_format=True)

    @staticmethod
    def read_diameter_from_dicom(info):
        try:
            Diameter_tag = info[0x0029, 0x1140].value[2][0x0029, 0x1044].value
        except Exception:
            try:
                Diameter_tag = info[0x0029, 0x1240].value[0][0x0029, 0x1044].value
            except Exception:
                raise ValueError(
                    "Error in function IOHelperFunctions.read_diameter_from_dicom: "
                    "The DICOM image has no diameter attribute."
                )

        return IOHelperFunctions.encode_private_header(Diameter_tag)

    @staticmethod
    def encode_private_header(header):
        """The tag holds space separated 10-character numbers; fields 2 and 3 are the diameters."""
        encoded = header.decode("ascii", errors="ignore") if isinstance(header, (bytes, bytearray)) else str(header)

        horiz_diameter = float(encoded[11:21])
        vert_diameter = float(encoded[22:32])

        return horiz_diameter, vert_diameter

    @staticmethod
    def readPFOinfo(info, path_name):
        isPFO = "PFO"
        noPFO_tags = ["noPFO", "noIBHC"]

        for tag in noPFO_tags:
            if IOHelperFunctions.contains(path_name, tag, "IgnoreCase", True):
                isPFO = "noPFO"

        return isPFO

    @staticmethod
    def extractSeriesDescription(folderName, tissue_list, diameters):
        """Derives tissue and phantom diameter from the folder naming convention."""
        try:
            seriesDescription = str(folderName).replace("\\", "/").split("/")[-1]
            splitted = seriesDescription.split("_")

            diam_hits = [part for part in splitted if part in diameters]
            lowered = [part.lower() for part in splitted]
            tissue_hits = [splitted[i] for i, part in enumerate(lowered)
                           if part in [str(t).lower() for t in tissue_list]]

            if len(tissue_hits) == 1 and len(diam_hits) == 1:
                return tissue_hits[0], diam_hits[0]
        except Exception:
            pass

        return float("nan"), float("nan")

    # ----------------------------------------------------------------- images

    @staticmethod
    def create_circular_mask(radius, centers, resolution):
        line = np.arange(1, resolution + 1)
        indices = np.tile(line, (resolution, 1))

        x = indices - float(centers[0])
        y = indices.T - float(centers[1])

        return (x ** 2 + y ** 2) < radius ** 2

    @staticmethod
    def center_of_gravity(circle, centers, r):
        """Intensity-weighted centroid inside a circular mask, returned as [x, y]."""
        from scipy.ndimage import center_of_mass, correlate

        circle = np.asarray(circle, dtype=float)

        mask = IOHelperFunctions.create_circular_mask(r, centers, circle.shape[0])

        window_size = 5
        h = np.ones(window_size) / window_size
        H = np.outer(h, h)

        # Brain or adipose inserts can go negative, so weight with the magnitude.
        filtered = np.abs(correlate(np.abs(circle), H, mode="constant"))

        row, col = center_of_mass(filtered * mask)
        return np.array([col + 1.0, row + 1.0])

    @staticmethod
    def getCenter(img):
        """Estimates the object center from the half-maximum width of both axis profiles."""
        img = np.asarray(img, dtype=float)
        center = np.zeros(2, dtype=np.int32)

        for axis in (0, 1):
            profile = img.sum(axis=axis) / img.shape[axis]

            peak = int(np.argmax(profile)) + 1
            value = profile[peak - 1]

            threshold = value - abs((value - profile.min()) * 0.5)
            below = profile < threshold

            backwards = np.flatnonzero(below[:peak][::-1])
            lower_bound = peak - (backwards[0] + 1) if backwards.size else 1

            forwards = np.flatnonzero(below[peak - 1:])
            higher_bound = peak + (forwards[0] + 1) if forwards.size else img.shape[1]

            center[axis] = int(np.floor((lower_bound + higher_bound) / 2 + 0.5))

        # The patient table biases the vertical profile, so shift the first guess slightly up.
        center[1] = int(np.floor(center[1] * 0.98 + 0.5))

        return center

    # ------------------------------------------------------------------ files

    @staticmethod
    def Save(volume, filename):
        """Saves to the global output folder as binary. Requires a '.bin' file extension."""
        IO = IOHelperFunctions()

        filename = IO.GlobalOutputFolderPath / filename

        if ".bin" not in str(filename):
            raise ValueError("Invalid file-extension! Only binary files allowed")

        TypeCast.To(volume, Precision.Default).tofile(filename)

    @staticmethod
    def Load(filename):
        """Loads from the global output folder. Requires a '.bin' file extension."""
        IO = IOHelperFunctions()

        filename = IO.GlobalOutputFolderPath / filename

        if not Path(filename).is_file():
            raise FileNotFoundError("File does not exist!")

        if ".bin" not in str(filename):
            raise ValueError("Invalid file-extension! Only binary files allowed")

        raw = np.fromfile(filename, dtype=np.float64)
        return TypeCast.To(raw, Precision.Value())

    @staticmethod
    def SaveDoubleToFile(filePath, data):
        np.asarray(data, dtype=np.float64).tofile(filePath)

    @staticmethod
    def SaveStructToMat(filePath, dataStruct):
        from scipy.io import savemat

        savemat(str(filePath), dataStruct)

    @staticmethod
    def SaveReconAsGif(recon_data, gifPath, Nx, Ny, NRecons, label, WC, WW, fps):
        """Writes the reconstruction phases as an animated GIF with window/level clipping."""
        from PIL import Image

        recon_data = np.asarray(recon_data, dtype=float)

        # GIF stores the delay in 1/100 s; flooring avoids an accidental slowdown.
        delayTime = max(0.02, np.floor((1 / fps) * 100) / 100)

        WC_min = WC - WW / 2
        WC_max = WC + WW / 2

        frames = []
        for phase in range(1, NRecons + 1):
            img = recon_data[phase - 1, :].reshape(Ny, Nx).T

            img_norm = np.clip((np.clip(img, WC_min, WC_max) - WC_min) / (WC_max - WC_min), 0.0, 1.0)
            frames.append(Image.fromarray((img_norm * 255).astype(np.uint8)).convert("P"))

        frames[0].save(gifPath, save_all=True, append_images=frames[1:],
                       duration=int(delayTime * 1000), loop=0)

        print(f"  Saved: {gifPath} (fps req: {fps:.3f}, fps eff: {1 / delayTime:.3f})")

    @staticmethod
    def ExportReadings(Readings, folder):
        from PIL import Image

        Readings = np.asarray(Readings, dtype=float)

        for idx in range(1, Readings.shape[0] + 1):
            image = np.squeeze(Readings[idx - 1, :, :]).T
            Image.fromarray(_ToUint8(image)).save(Path(folder) / f"Reading_{idx}.png")

    @staticmethod
    def ExportSlices(Volume, folder):
        from PIL import Image

        Volume = np.asarray(Volume, dtype=float)
        slices = Volume.shape[2]

        for idx in range(1, slices + 1):
            Image.fromarray(_ToUint8(Volume[:, :, idx - 1])).save(Path(folder) / f"Slice_{idx}.png")

    @staticmethod
    def printMatrixforXML(CTlookup, SPRlookup, spr999, diameter):
        CTlookup = np.atleast_2d(np.asarray(CTlookup))
        SPRlookup = np.ravel(np.asarray(SPRlookup))

        for i in range(CTlookup.shape[0]):
            print(f'<DiameterLookUp Size="{diameter[i]:.0f}">')
            print("<CTLookUp>" + "|".join(f"{value:.0f}" for value in CTlookup[i, :]) + "</CTLookUp>")
            print("<SPRLookUp>" + "".join(f"{value:.4f}|" for value in SPRlookup)
                  + f"{spr999[i]:.4f}</SPRLookUp>")
            print("</DiameterLookUp>")


def _ToUint8(image):
    """MATLAB `imwrite` of a double image expects [0, 1]; values outside are clipped."""
    return (np.clip(np.asarray(image, dtype=float), 0.0, 1.0) * 255).astype(np.uint8)
