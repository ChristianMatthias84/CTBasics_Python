import re
from pathlib import Path


class ReadRawDataFunctions:
    @staticmethod
    def get_mode_type(mode_id):
        mapping = {24: "P83A", 31: "P64A"}
        return mapping.get(mode_id, "UNKNOWN")

    @staticmethod
    def get_table_path(table_name, root_dir):
        return Path(root_dir) / "Tables" / table_name


class RawDataScanDescriptor:
    @staticmethod
    def load(descriptor, tag_name):
        xml_text = descriptor.ModeParXML.String
        match = re.search(rf"<{re.escape(tag_name)}>(.*?)</{re.escape(tag_name)}>", xml_text, flags=re.DOTALL)
        if match:
            return match.group(1)
        return None
