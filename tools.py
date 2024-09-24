import os
import tempfile
from typing import Tuple, Dict
from nibabel import Nifti1Image, load, as_closest_canonical
from scipy.stats import mode
import numpy as np
import gzip
import shutil
from feature_config import get_enabled_features
from skimage.feature import graycomatrix, graycoprops
from skimage.filters import threshold_otsu
from scipy.ndimage import zoom
import matplotlib.pyplot as plt

def getsize(fn: str) -> int:
    """
    Get .nii size file in bytes.
    Note
    ____
    The given file name `fn` can either be a .nii file or a .nii.gz file, where the sum of inner files are returned if
    it's a .nii.gz file.
    Parameters
    ----------
    fn : str
        A path to either a .nii or .nii.gz file.
    Returns
    -------
    : int
        The file size in bytes.
    """
    assert fn.endswith(".nii.gz") or fn.endswith(".nii")

    if fn.endswith(".nii"):
        return os.path.getsize(fn)

    with tempfile.TemporaryDirectory() as temp_dir:
        extracted_folder = os.path.join(temp_dir, "temp_extracted_files")
        file_sizes = []

        # Extract the gzip file
        with gzip.open(fn, "rb") as f_in:
            os.makedirs(extracted_folder, exist_ok=True)
            with open(os.path.join(extracted_folder, "temp_file.tar"), "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

        # Iterate over the extracted files
        for root, _, files in os.walk(extracted_folder):
            for file in files:
                file_path = os.path.join(root, file)
                # Get the size of each file
                file_size = os.path.getsize(file_path)
                file_sizes.append(file_size)

        # Clean up extracted files
        shutil.rmtree(extracted_folder)

    return sum(file_sizes)


def load_nifti_data(nifti_file_name: str) -> Tuple[np.ndarray, Nifti1Image]:
    """
    Loading data from a nifti file.

    :param nifti_file_name: The path to the desired nifti file.

    :return: A tuple in the following form: (data, file), where:
    • data is a ndarray containing the loaded data.
    • file is the file object that was loaded.
    """

    # loading nifti file
    nifti_file = load(nifti_file_name)

    try:
        nifti_file = as_closest_canonical(nifti_file)

        # extracting the data of the file
        data = nifti_file.get_fdata().astype(np.float32)
    except:
        return None, nifti_file

    return data, nifti_file


def get_positive_and_negative_modes(
    data: np.ndarray,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Extracting from an array the positive mode and negative modes.

    :param data: An ndarray.

    :return: A tuple in the following form: (hist_loc_max_x_1, hist_loc_max_y_1), (hist_loc_max_x_2, hist_loc_max_y_2) where:
        • (hist_loc_max_x_1, hist_loc_max_y_1) is a tuple containing the (x, y) coordinates of the positive mode, and
        • (hist_loc_max_x_2, hist_loc_max_y_2) is a tuple containing the (x, y) coordinates of the negative mode.
        In both tuples, the (x,y) point is relative to the positive/negative histogram, when 'y' is in percentage.
    """

    data_pos = data[data >= 0.5]
    data_neg = data[data < 0.5]

    pos_mode = mode(data_pos, axis=None)
    neg_mode = mode(data_neg, axis=None)
    hist_loc_max_x_1 = pos_mode.mode
    hist_loc_max_x_2 = neg_mode.mode
    hist_loc_max_y_1 = 100 * pos_mode.count / data.size
    hist_loc_max_y_2 = 100 * neg_mode.count / data.size

    return (hist_loc_max_x_1, hist_loc_max_y_1), (hist_loc_max_x_2, hist_loc_max_y_2)


def calculate_haralick_features(image):
    # Convert the image to unsigned 8-bit integer
    image_uint8 = (image * 255).astype(np.uint8)
    
    glcm = graycomatrix(image_uint8, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast').mean()
    dissimilarity = graycoprops(glcm, 'dissimilarity').mean()
    homogeneity = graycoprops(glcm, 'homogeneity').mean()
    energy = graycoprops(glcm, 'energy').mean()
    correlation = graycoprops(glcm, 'correlation').mean()
    return [contrast, dissimilarity, homogeneity, energy, correlation]

def calculate_cnr(image):
    """Calculate the Contrast-to-Noise Ratio (CNR) of an image."""
    threshold = threshold_otsu(image)
    foreground = image[image > threshold]
    background = image[image <= threshold]
    
    if len(foreground) == 0 or len(background) == 0:
        return 0.0
    
    bg_std = np.std(background)
    if bg_std == 0:
        return 0.0
    
    cnr = abs(np.mean(foreground) - np.mean(background)) / bg_std
    return np.clip(cnr, 0, 1e6)  # Clip to a reasonable maximum value

def extract_numerical_measures(
    case_path: str,
    mip_images: Dict[str, np.ndarray],
    clip_intensities: bool = True,
    return_metadata: bool = False,
    normalize_data: bool = True,
) -> tuple:
    """
    Extracting measures of a given CT scan path.
    The measures extracted are based on the enabled features in feature_config.py.

    :param case_path: The path to the CT scan's nifti file
    :param mip_images: A dictionary containing the pre-calculated MIP images
    :return: A tuple containing all the enabled measures.
    """

    data, nifti_file = load_nifti_data(case_path)

    data_shape = data.shape

    working_data = data.flatten()
    if clip_intensities and working_data.size > 0:
        working_data = np.clip(working_data, -150, 150)
        working_data = np.delete(
            working_data,
            np.where(
                np.logical_or(
                    working_data == working_data.min(),
                    working_data == working_data.max(),
                )
            ),
        )
    if normalize_data and working_data.size > 0:
        working_data = (working_data - working_data.mean()) / working_data.std()

    vox_dims = nifti_file.header.get_zooms()
    
    enabled_features = get_enabled_features()
    res = []

    if "Mean" in enabled_features:
        res.append(working_data.mean())
    if "STD" in enabled_features:
        res.append(working_data.std())
    if any(p in enabled_features for p in ["Median", "10th percentile", "25th percentile", "75th percentile", "90th percentile"]):
        percentiles = np.percentile(working_data, (10, 25, 50, 75, 90))
        if "10th percentile" in enabled_features:
            res.append(percentiles[0])
        if "25th percentile" in enabled_features:
            res.append(percentiles[1])
        if "Median" in enabled_features:
            res.append(percentiles[2])
        if "75th percentile" in enabled_features:
            res.append(percentiles[3])
        if "90th percentile" in enabled_features:
            res.append(percentiles[4])
    if "File Size (MB)" in enabled_features:
        res.append(getsize(case_path) / 1000000)
    if "z-axis" in enabled_features:
        res.append(data_shape[2])
    if "Voxel volume" in enabled_features:
        res.append(vox_dims[0] * vox_dims[1] * vox_dims[2])

    if any(h in enabled_features for h in ["hist_loc_max_x_1", "hist_loc_max_y_1", "hist_loc_max_x_2", "hist_loc_max_y_2"]):
        (hist_loc_max_x_1, hist_loc_max_y_1), (hist_loc_max_x_2, hist_loc_max_y_2) = get_positive_and_negative_modes(working_data)
        if "hist_loc_max_x_1" in enabled_features:
            res.append(hist_loc_max_x_1)
        if "hist_loc_max_y_1" in enabled_features:
            res.append(hist_loc_max_y_1)
        if "hist_loc_max_x_2" in enabled_features:
            res.append(hist_loc_max_x_2)
        if "hist_loc_max_y_2" in enabled_features:
            res.append(hist_loc_max_y_2)

    if "Haralick_Sagital" in enabled_features and "sagital" in mip_images:
        res.extend(calculate_haralick_features(mip_images["sagital"]))
    if "Haralick_Coronal" in enabled_features and "coronal" in mip_images:
        res.extend(calculate_haralick_features(mip_images["coronal"]))
    if "Haralick_Axial" in enabled_features and "axial" in mip_images:
        res.extend(calculate_haralick_features(mip_images["axial"]))

    if "CNR_Sagital" in enabled_features and "sagital" in mip_images:
        res.append(calculate_cnr(mip_images["sagital"]))
    if "CNR_Coronal" in enabled_features and "coronal" in mip_images:
        res.append(calculate_cnr(mip_images["coronal"]))
    if "CNR_Axial" in enabled_features and "axial" in mip_images:
        res.append(calculate_cnr(mip_images["axial"]))

    res = [np.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6) for x in res]

    if return_metadata:
        return tuple(res), data, nifti_file
    return tuple(res)

def mip(img_data):
    mip_list = [np.max(img_data, axis =i) for i in range(3)]
    for i in range(3):
        zoom_factor = [256 / mip_list[i].shape[j] for j in range(2)]
        mip_list[i] = zoom(mip_list[i], zoom_factor)
    return np.stack(mip_list, axis=0)

def plot_precision_recall_curve(precisions, recalls, a_ps, save_file_name='precision_recall_curve.jpg'):
    plt.figure(figsize=(14, 10))
    
    # get min len of the precisions and recalls
    min_len = min([len(precision) for precision in precisions])
    # truncate the last precision and recall to the min_len
    precisions = [precision[:min_len] for precision in precisions]
    recalls = [recall[:min_len] for recall in recalls]
    for i, (precision, recall) in enumerate(zip(precisions, recalls)):
        ap = a_ps[i]
        plt.plot(recall, precision, lw=1, alpha=0.3,
                 label=f'Fold {i+1} (AP = {ap:.2f})')

    mean_precision = np.mean(precisions, axis=0)
    mean_recall = np.mean(recalls, axis=0)
    mean_ap = np.mean(a_ps)
    std_ap = np.std(a_ps)
    plt.rcParams.update({'font.size': 16})
    plt.plot(mean_recall, mean_precision, color='b',
             label=f'Mean PR (AP = {mean_ap:.2f} ± {std_ap:.2f})',
             lw=2, alpha=.8)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.savefig(save_file_name)
    plt.close()

def plot_roc_curve(fprs, tprs, aucs, save_file_name='roc_curve.jpg'):
    plt.figure(figsize=(14, 10))
    
    for i, (fpr, tpr) in enumerate(zip(fprs, tprs)):
        roc_auc = aucs[i]
        plt.plot(fpr, tpr, lw=1, alpha=0.3,
                 label=f'Fold {i+1} (AUC = {roc_auc:.2f})')

    mean_fpr = np.linspace(0, 1, 100)
    mean_tpr = np.mean([np.interp(mean_fpr, fpr, tpr) for fpr, tpr in zip(fprs, tprs)], axis=0)
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)

    
    plt.rcParams.update({'font.size': 16})
    # make the font bold
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=f'Mean ROC (AUC = {mean_auc:.2f} ± {std_auc:.2f})',
             lw=2, alpha=.8)

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(save_file_name)
    plt.close()