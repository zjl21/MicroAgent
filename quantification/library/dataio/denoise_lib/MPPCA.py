import os
import shutil
import subprocess

class MPPCA:
    def __init__(self, input_file, output_file, mask_file):
        self.input_file = input_file
        self.output_file = output_file
        self.mask_file = mask_file

    def run(self):
        mrtrix_bin = os.environ.get("MRTRIX3_BIN", "").strip()
        command = os.path.join(mrtrix_bin, "dwidenoise") if mrtrix_bin else shutil.which("dwidenoise")
        if not command or not os.path.isfile(command) or not os.access(command, os.X_OK):
            raise FileNotFoundError(
                "MRtrix3 dwidenoise was not found. Set MRTRIX3_BIN to the MRtrix3 bin directory."
            )
        subprocess.run(
            [command, "-mask", self.mask_file, self.input_file, self.output_file],
            check=True,
        )
