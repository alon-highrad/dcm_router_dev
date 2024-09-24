"""
Script for creating the nifti dataset from the DICOM library on our servers.
"""
import os
import subprocess
import tempfile
from multiprocessing import Pool
import fnmatch
import tqdm
from classifier.mri_classification import MRISequenceClassification
from glob import glob
from pathlib import Path

def process_path(path):
    name = os.path.basename(path)
    cur_output = os.path.join(OUTPUT_DIR, name)
    os.makedirs(cur_output, exist_ok=True)
    print(cur_output)
    print(os.path.join(DICOM_LIBRARY_PATH, path))
    convert_command = ["dcm2niix", "-o", cur_output, "-z", 'y', os.path.join(DICOM_LIBRARY_PATH, path) + '/DICOM']
    subprocess.run(convert_command, check=False)

def process_path_t1(path):
    name = os.path.basename(path)
    cur_output = os.path.join(OUTPUT_DIR, name)
    os.makedirs(cur_output, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp_dir:
        dcm_study_file_paths = []
        for path in Path(os.path.join(DICOM_LIBRARY_PATH, path)).rglob('*'):
            if path.is_file():
                dcm_study_file_paths.append(str(path))
        # dcm_study_file_paths = glob(os.path.join(DICOM_LIBRARY_PATH, path) + '/DICOM/**/*.dcm', recursive=True)
        # print(dcm_study_file_paths)
        step = MRISequenceClassification(dcm_study_file_paths)
        cr = step.execute()
        try:
            for f in cr.get_filtered_dcm_filenames([('t1w', 'ax')]):
                os.symlink(f, os.path.join(tmp_dir, f.replace('/', '_')))

            convert_command = ["dcm2niix", "-o", cur_output, "-z", 'y', tmp_dir]
            subprocess.run(convert_command, check=False)
        except KeyError:
            print('no axial t1w found')

if __name__ == '__main__':
    DICOM_LIBRARY_PATH = "/media/alon/My Passport/NBIA_brain_dataset/manifest-1669766397961/UPENN-GBM/"
    OUTPUT_DIR = "/media/alon/My Passport/NBIA_brain_nifti_files/"
    dirs = os.listdir(DICOM_LIBRARY_PATH)

    # Create a pool of worker processes
    # pool = Pool()

    # Process each path using the worker pool
    # pool.map(process_path_t1, dirs)

    flag = False
    for d in tqdm.tqdm(dirs):
        print('############################')
        print(d)
        print('############################')
        process_path_t1(d)

    # Close the worker pool and wait for all processes to finish
    # pool.close()
    # pool.join()