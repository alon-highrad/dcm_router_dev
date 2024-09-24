from series_stats import load_referrence_histograms, bhattacharyya_distance, z_normalize, create_normalized_histogram
import json
import os.path
from os import makedirs
from os.path import basename, isdir, dirname
from glob import glob
from time import time
from typing import Tuple, List, Union, Dict
from multiprocessing import Pool
from model_generation import get_number_of_false_cases
from tools import extract_numerical_measures, load_nifti_data
from joblib import load
import pandas as pd
# import mahotas
import numpy as np
from tqdm import tqdm
from scipy import ndimage
from feature_config import get_enabled_features
from nibabel import load as nib_load

REF_HISTS_PATH = "./contrast_referrence/"


def load_mip_images(case_path: str) -> Dict[str, np.ndarray]:
    base_path = case_path[:-7]  # Remove '.nii.gz'
    enabled_features = get_enabled_features()
    mip_images = {}

    def load_and_normalize(file_path):
        image = nib_load(file_path).get_fdata()
        return (image - np.min(image)) / (np.max(image) - np.min(image))

    if "Haralick_Sagital" in enabled_features or "CNR_Sagital" in enabled_features:
        mip_images["sagital"] = load_and_normalize(f"{base_path}_sagital_mip.nii.gz")
    
    if "Haralick_Coronal" in enabled_features or "CNR_Coronal" in enabled_features:
        mip_images["coronal"] = load_and_normalize(f"{base_path}_coronal_mip.nii.gz")
    
    if "Haralick_Axial" in enabled_features or "CNR_Axial" in enabled_features:
        mip_images["axial"] = load_and_normalize(f"{base_path}_axial_mip.nii.gz")

    return mip_images


def extract_measures(case: Tuple[str, str, str, bool]) -> pd.DataFrame:
    """
    Extracting measures of a given CT scan.
    The measures extracted are based on the enabled features in feature_config.py.

    :param case: A tuple in the following form: ('file path', 'Scan ID', 'Case ID', 'label').

    :return: A Dataframe containing all the enabled measures extracted.
    """

    case_nifti_file_path, scan_id, case_id, label = case
    print(case_nifti_file_path)

    mip_images = load_mip_images(case_nifti_file_path)

    numerical_measures = extract_numerical_measures(
        case_nifti_file_path, mip_images, clip_intensities=False
    )
    if len(numerical_measures) == 0:
        return pd.DataFrame()

    enabled_features = get_enabled_features()
    measures_dict = {
        "Scan ID": [scan_id],
        "Case ID": [case_id],
        "Label": [label],
    }

    measure_index = 0
    for feature in enabled_features:
        if feature in ["Haralick_Sagital", "Haralick_Coronal", "Haralick_Axial"]:
            haralick_features = ["Contrast", "Dissimilarity", "Homogeneity", "Energy", "Correlation"]
            for haralick_feature in haralick_features:
                if measure_index < len(numerical_measures):
                    value = numerical_measures[measure_index]
                    measures_dict[f"{feature}_{haralick_feature}"] = [format(np.clip(value, -1e6, 1e6), ".2f")]
                    measure_index += 1
                else:
                    measures_dict[f"{feature}_{haralick_feature}"] = ["0.00"]
        elif feature in ["CNR_Sagital", "CNR_Coronal", "CNR_Axial"]:
            if measure_index < len(numerical_measures):
                value = numerical_measures[measure_index]
                measures_dict[feature] = [format(np.clip(value, 0, 1e6), ".2f")]
                measure_index += 1
            else:
                measures_dict[feature] = ["0.00"]
        else:
            if measure_index < len(numerical_measures):
                value = numerical_measures[measure_index]
                measures_dict[feature] = [format(np.clip(value, -1e6, 1e6), ".2f")]
                measure_index += 1
            else:
                measures_dict[feature] = ["0.00"]

    # Add Bhattacharyya distance if enabled
    if "bhatt_distance" in enabled_features:
        reference_histograms = load_referrence_histograms(REF_HISTS_PATH)
        bhatt_distances = extract_distance_measures(case, reference_histograms)
        for i, distance in enumerate(bhatt_distances):
            measures_dict[f"bhatt_distance_{i+1}"] = [format(distance, ".2f")]

    measures = pd.DataFrame(measures_dict)
    measures.set_index("Scan ID", inplace=True)
    return measures


def extract_distance_measures(
    case: Tuple[str, str, str, bool], reference_histograms: List[np.ndarray]
):
    
    counts=create_normalized_histogram(z_normalize(load_nifti_data(case[0])[0]), reference_histograms[0]["bins"])
    
    return [
        bhattacharyya_distance((ref_hist["counts"], ref_hist["bins"]), (counts,ref_hist["bins"]))
        for ref_hist in reference_histograms
    ]



def write_to_excel(data: pd.DataFrame, result_excel_saving_path: str):
    """
    Writing the given data to an excel file.

    :param data: A Dataframe containing the data to save.
    :param result_excel_saving_path: The path where to save the result excel file.
    """
    dir_name = dirname(result_excel_saving_path)
    if dir_name != "":
        makedirs(dirname(result_excel_saving_path), exist_ok=True)
    writer = pd.ExcelWriter(result_excel_saving_path, engine="xlsxwriter")

    workbook = writer.book

    cell_format = workbook.add_format({"num_format": "#,##0.00"})
    cell_format.set_font_size(14)

    data.to_excel(excel_writer=writer, columns=data.keys(), startcol=0, startrow=0)

    worksheet = writer.sheets["Sheet1"]

    # Fix first column
    column_len = (
        data.axes[0].astype(str).str.len().max()
        + data.axes[0].astype(str).str.len().max() * 0.5
    )
    worksheet.set_column(0, 0, column_len, cell_format)

    # Fix all  the rest of the columns
    for i, col in enumerate(data.keys()):
        # find length of column i
        column_len = data[col].astype(str).str.len().max()
        # Setting the length if the column header is larger
        # than the max column value length
        column_len = max(column_len, len(col))
        column_len += column_len * 0.5
        # set the column length
        worksheet.set_column(i + 1, i + 1, column_len, cell_format)

    header_format = workbook.add_format(
        {"bold": True, "text_wrap": True, "font_size": 16, "valign": "top", "border": 1}
    )

    for col_num, value in enumerate(data.keys()):
        worksheet.write(0, col_num + 1, value, header_format)
    for row_num, value in enumerate(data.axes[0].astype(str)):
        worksheet.write(row_num + 1, 0, value, header_format)

    writer.close()


def generate_data(
    cases: List[Tuple[str, str, str, bool]], result_excel_saving_path: str
):
    """
    Generating data of the given cases.
    The result is saved in an Excel file.
    To see which measures are extracted see the documentation of the function 'extract_measures' in the current module.

    :param cases: A list containing for every given case a tuple in the following form:
        ('file path', 'Scan ID', 'Case ID', 'label').
    :param result_excel_saving_path: The path of the excel file where the data will be saved.
    """
    print(len(cases))
    with Pool(processes=16) as pool:
        measures = pool.map(extract_measures, cases)
    measures = [m for m in measures if m is not None]
    measures = pd.concat(measures)

    write_to_excel(measures, result_excel_saving_path)

    return measures


def filter_similar_scans(distance_excel_path: str, alon_cases: List[Tuple], oo_cases: List[Tuple], percentile: float = 80) -> Tuple[List[Tuple], List[Tuple]]:
    """
    Filter scans to keep only the most similar ones based on Bhattacharyya distance.
    
    :param distance_excel_path: Path to the Excel file containing the distance table
    :param alon_cases: List of Alon cases
    :param oo_cases: List of OO cases
    :param percentile: Percentile threshold for similarity (default: 80)
    :return: Tuple of filtered case lists (alon_filtered, oo_filtered)
    """
    # Read the distance table
    distance_df = pd.read_excel(distance_excel_path, index_col=0)
    
    # Calculate the mean distance for each scan
    alon_mean_distances = distance_df.mean(axis=1)
    oo_mean_distances = distance_df.mean(axis=0)
    
    # Calculate the threshold for the given percentile
    alon_threshold = np.percentile(alon_mean_distances, 100 - percentile)
    oo_threshold = np.percentile(oo_mean_distances, 100 - percentile)
    
    # Filter the scans
    alon_filtered_indices = set(alon_mean_distances[alon_mean_distances <= alon_threshold].index)
    oo_filtered_indices = set(oo_mean_distances[oo_mean_distances <= oo_threshold].index)
    
    alon_filtered = [case for case in alon_cases if case[0].replace(".gz", "_hist_perc_0.npz") in alon_filtered_indices]
    oo_filtered = [case for case in oo_cases if case[0].replace(".gz", "_hist_perc_0.npz") in oo_filtered_indices]
    
    return alon_filtered, oo_filtered

if __name__ == "__main__":
    experiment_name = "similar_scans_all_features_after_filtering"
    experiment_dir = os.path.join(os.getcwd(), experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)

    t = time()

    # Load alon_labels.json
    alon_cases = []
    with open("alon_labels_filtered.json", "r") as f:
        alon_labels = json.load(f)
    for study_name, series_dict in alon_labels.items():
        for scan_path, label in series_dict.items():
            alon_cases.append((scan_path, os.path.basename(scan_path), study_name, label))

    # Load oo_test_labels.json
    oo_cases = []
    with open("oo_test_labels_filtered.json", "r") as f:
        oo_labels = json.load(f)
    for study_name, series_dict in oo_labels.items():
        for scan_path, label in series_dict.items():
            oo_cases.append((scan_path, os.path.basename(scan_path), study_name, label))

    # Filter similar scans
    distance_excel_path = "./hist_results/hadassah_vs_ood_contrast_distances.xlsx"
    alon_filtered, oo_filtered = filter_similar_scans(distance_excel_path, alon_cases, oo_cases)

    print(f"Original Alon cases: {len(alon_cases)}, Filtered: {len(alon_filtered)}")
    print(f"Original OO cases: {len(oo_cases)}, Filtered: {len(oo_filtered)}")

    # Filter similar non-contrast scans
    non_contrast_distance_excel_path = "./hist_results/hadassah_vs_ood_non_contrast_distances.xlsx"
    alon_non_contrast_filtered, oo_non_contrast_filtered = filter_similar_scans(non_contrast_distance_excel_path, alon_cases, oo_cases)
    alon_filtered.extend(alon_non_contrast_filtered)
    oo_filtered.extend(oo_non_contrast_filtered)
    print(f"Original Alon cases: {len(alon_cases)}, Filtered: {len(alon_non_contrast_filtered)}")
    print(f"Original OO cases: {len(oo_cases)}, Filtered: {len(oo_non_contrast_filtered)}")

    # Generate data for filtered Alon cases
    alon_result_data_path = os.path.join(experiment_dir, "alon_filtered_measures_results.xlsx")
    alon_data = generate_data(alon_filtered, alon_result_data_path)

    print(f"Filtered Alon data was generated in {time() - t:.2f} seconds")
    print(f"Filtered Alon results saved in: {alon_result_data_path}")

    # Generate data for filtered OO cases
    t = time()
    oo_result_data_path = os.path.join(experiment_dir, "oo_filtered_measures_results.xlsx")
    oo_data = generate_data(oo_filtered, oo_result_data_path)

    print(f"Filtered OO test data was generated in {time() - t:.2f} seconds")
    print(f"Filtered OO test results saved in: {oo_result_data_path}")

    print(f"All results saved in: {experiment_dir}")