"""
script that goes over the nifti files from the entire nifti dataset extracted directly from the DICOM
We then go over each ground truth image used in the training database and compare it to the images.
For each one that exists in the training data, we give a label of 1, the rest get a label of 0.
"""
import filecmp
import json
import os
from glob import glob

import nibabel as nib

def filter_series(labels, min_size_mb=2, min_slices=10, min_resolution_mm=(1.5, 1.5, 5.0)):
    filtered_labels = {}
    total_filtered = 0
    label_1_filtered = 0

    for study_name, series in labels.items():
        filtered_labels[study_name] = {}
        for file_path, label in series.items():
            try:
                img = nib.load(file_path)
                
                # Check file size
                size_mb = os.path.getsize(file_path) / (1024 * 1024)
                if size_mb < min_size_mb:
                    total_filtered += 1
                    if label == 1:
                        label_1_filtered += 1
                        print(f"Filtered {file_path} because of size")
                    continue

                # Check number of slices
                if img.shape[2] < min_slices:
                    total_filtered += 1
                    if label == 1:
                        label_1_filtered += 1
                        print(f"Filtered {file_path} because of number of slices")
                    continue

                # Check resolution
                resolution = img.header.get_zooms()
                if any(res > min_resolution_mm[i] for i, res in enumerate(resolution)):
                    total_filtered += 1
                    if label == 1:
                        label_1_filtered += 1
                        print(f"Filtered {file_path} because of resolution")
                    continue

                filtered_labels[study_name][file_path] = label

            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                total_filtered += 1
                if label == 1:
                    label_1_filtered += 1

    print(f"Total series filtered: {total_filtered}")
    print(f"Series with label 1 filtered: {label_1_filtered}")
    return filtered_labels

if __name__ == '__main__':

    # all_labels = {}
    # NIFTI_DB_PATH = "/media/alon/My Passport/hadassah_brain_nifti_files/"
    # studies = os.listdir(NIFTI_DB_PATH)
    # for study in studies:
    #     study_name = os.path.basename(study)
    #     non_gd_files = glob(NIFTI_DB_PATH + study + "/*.nii.gz")
    #     gd_files = glob(NIFTI_DB_PATH + study + "/T1_GD/*.nii.gz")
    #     if len(gd_files) == 0:
    #         continue
        
    #     all_labels[study_name] = {}
    #     for f in gd_files:
    #         all_labels[study_name][f] = 1
    #     for f in non_gd_files:
    #         all_labels[study_name][f] = 0

    # all_labels = filter_series(all_labels)

    # with open('alon_labels_filtered.json', 'w') as f:
    #     json.dump(all_labels, f)

    all_labels = {}
    NIFTI_DB_PATH = "/media/alon/My Passport/NBIA_brain_nifti_files/"
    studies = os.listdir(NIFTI_DB_PATH)
    for study in studies:
        study_name = os.path.basename(study)
        all_labels[study_name] = {}
        for f in glob(NIFTI_DB_PATH + '/' + study_name + "/*.nii.gz"):
            if "T2" in os.path.basename(f):
                continue
            if 'stealth' in os.path.basename(f) or 'STEALTH' in os.path.basename(f):
                all_labels[study_name][f] = 1
            else:
                all_labels[study_name][f] = 0

    all_labels = filter_series(all_labels)

    with open('oo_test_labels_filtered.json', 'w') as f:
        json.dump(all_labels, f)