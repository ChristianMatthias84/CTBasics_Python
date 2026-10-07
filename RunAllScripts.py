"""Runs the full CTBasics script suite in the same order as MATLAB's RunAllScripts.m."""
import argparse
import importlib
import time

# Exactly the order of the active run() calls in RunAllScripts.m. The two commented out
# TestScriptScannerData3D entries stay disabled because that script needs scanner raw data.
SCRIPTS = [
    "CTBasics.LecturesAndTestScripts.TestScript2D",
    "CTBasics.LecturesAndTestScripts.TestScript3D",
    "CTBasics.LecturesAndTestScripts.CTRecon101",
    "CTBasics.LecturesAndTestScripts.CTRecon102",
    "CTBasics.LecturesAndTestScripts.CTRecon103",
    "CTBasics.LecturesAndTestScripts.Test4DCT",
    "CTBasics.LecturesAndTestScripts.TestMAR",
    "CTBasics.Projects.StaticCT.StaticCT",
    "CTBasics.Projects.StaticCT.StaticCT3D",
    "CTBasics.Projects.StaticCT.ProjectsAH.DualSource."
    "DualSource_CardiacMotion4DCT_Main",
    "CTBasics.Projects.StaticCT.ProjectsJP.test_4DCT_prosp",
    "CTBasics.Projects.StaticCT.ProjectsCH.StaticCT_4_RealTimeImaging_Script_CH",
]


def main(Scripts=None, StopOnError=True):
    Scripts = list(SCRIPTS if Scripts is None else Scripts)

    tOuter = time.perf_counter()
    Results = {}

    for name in Scripts:
        print(f"=== {name} ===")
        tScript = time.perf_counter()
        try:
            module = importlib.import_module(name)
            Results[name] = module.main()
        except Exception as error:  # noqa: BLE001 - mirrors MATLAB aborting on error
            print(f"{name} failed: {error}")
            Results[name] = error
            if StopOnError:
                raise
        print(f"Elapsed time is {time.perf_counter() - tScript:.2f} seconds.")

    print(f"Total elapsed time is {time.perf_counter() - tOuter:.2f} seconds.")
    return Results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--continue-on-error", action="store_true",
                        help="keep running the remaining scripts after a failure")
    parser.add_argument("script", nargs="*", default=None,
                        help="optional subset of module names to run")
    arguments = parser.parse_args()

    main(Scripts=arguments.script or None, StopOnError=not arguments.continue_on_error)
