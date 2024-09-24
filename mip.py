import nibabel as nib
import numpy as np
from series_stats import z_normalize
from scipy.ndimage import gaussian_filter, zoom
import os
import json
from totalsegmentator.python_api import totalsegmentator

def get_mip_image(image_array, plane='sagital'):
    """
    Save the maximum intensity projection of a 3D image to a 2D image.
    """
    if plane == 'sagital':
        axis = 0
    elif plane == 'coronal':
        axis = 1
    elif plane == 'axial':
        axis = 2
    else:
        raise ValueError(f"Invalid plane: {plane}")
    
    image = z_normalize(image_array)
    
    res_image = np.max(image, axis=axis)
    return res_image

def save_mip_image(image_nif, dest_path, plane='sagital', output_shape=None, brain_mask=None):
    image_array = image_nif.get_fdata()
    if brain_mask is not None:
        image_array = image_array * brain_mask
    mip_image = get_mip_image(image_array, plane=plane)

    if output_shape is not None:
        # Calculate zoom factors
        current_shape = mip_image.shape
        zoom_factors = [output_shape[i] / current_shape[i] for i in range(2)]
        
        # Resample the image
        resampled_mip_image = zoom(mip_image, zoom_factors, order=1)
        nib.save(nib.Nifti1Image(resampled_mip_image, image_nif.affine), f"{dest_path}_resampled_{plane}_mip.nii.gz")
    
    nib.save(nib.Nifti1Image(mip_image, image_nif.affine), f"{dest_path}_{plane}_mip.nii.gz")

if __name__ == "__main__":
    # Load the JSON file created by create_labels.py
    with open('alon_labels_filtered.json', 'r') as f:
        alon_labels = json.load(f)
    
    with open('oo_test_labels_filtered.json', 'r') as f:
        oo_test_labels = json.load(f)
    
    # Combine both label sets
    all_labels = {**alon_labels, **oo_test_labels}
    # all_labels={**oo_test_labels}
    # all_labels = {**alon_labels}
    # Create MIP images for every series specified in the JSON file
    for study, series in all_labels.items():
        for file_path in series.keys():
            if os.path.exists(file_path):
                print(f"Processing: {file_path}")
                image_nifti = nib.load(file_path)
                brain_mask_nif = totalsegmentator(image_nifti, task="total_mr", roi_subset=["brain"], device="gpu")
                mask_data = brain_mask_nif.get_fdata()
                
                save_mip_image(image_nifti, file_path[:-7], plane='sagital', output_shape=(128,128), brain_mask=mask_data)
                save_mip_image(image_nifti, file_path[:-7], plane='coronal', output_shape=(128,128), brain_mask=mask_data)
                save_mip_image(image_nifti, file_path[:-7], plane='axial', output_shape=(128,128), brain_mask=mask_data)
            else:
                print(f"File not found: {file_path}")

    