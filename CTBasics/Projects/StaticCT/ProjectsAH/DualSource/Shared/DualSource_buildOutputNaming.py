"""Builds the output naming struct for a dual source experiment.

Port of Projects/Static CT/Projects AH/DualSource/Shared/DualSource_buildOutputNaming.m.
"""
import re


def DualSource_buildOutputNaming(selectedPhantom, plaque, cm, stent):
    if selectedPhantom is None:
        selectedPhantom = ""
    selectedPhantom = str(selectedPhantom)

    if not selectedPhantom:
        selectedPhantom = "UnknownPhantom"

    selectedPhantom = selectedPhantom.strip()

    phantomLabel = re.sub(r"[^A-Za-z0-9_-]", "", selectedPhantom)
    if not phantomLabel:
        phantomLabel = "UnknownPhantom"

    if "linepattern" in selectedPhantom.lower():
        rootFolderLabel = "linePattern"
    else:
        rootFolderLabel = phantomLabel

    phantomCode = f"P{round(plaque)}C{int(bool(cm))}S{int(bool(stent))}"

    return {
        "SelectedPhantom": selectedPhantom,
        "PhantomLabel": phantomLabel,
        "RootFolderLabel": rootFolderLabel,
        "PhantomCode": phantomCode,
        "PhantomTag": f"{phantomLabel}_{phantomCode}",
    }
