import os
import abc
import shutil
import logging
import tempfile
import subprocess

from glob import glob
from typing import List, Union


class ImageConverter(abc.ABC):
    @abc.abstractmethod
    def _select_best_image(self, output_images_paths: List[str]) -> str:
        pass

    def convert(self, dicom_input: Union[str, List[str]], output_path: str):

        if isinstance(dicom_input, list):
            with tempfile.TemporaryDirectory() as tmp_dir:
                for dicom_path in dicom_input:
                    os.symlink(dicom_path, os.path.join(tmp_dir, dicom_path.replace('/', '_')))
                convert_command = ["dcm2niix", "-o", output_path, "-z", 'y', '-v', 'n', tmp_dir]
                subprocess.run(convert_command, check=False)
        else:
            convert_command = ["dcm2niix", "-o", output_path, "-z", 'y', '-v', 'n', dicom_input]
            subprocess.run(convert_command, check=False)