FEATURES = {
    "Mean": False,
    "STD": False,
    "Median": True,
    "10th percentile": True,
    "25th percentile": True,
    "75th percentile": True,
    "90th percentile": True,
    "File Size (MB)": False,
    "z-axis": False,
    "Voxel volume": False,
    "hist_loc_max_x_1": True,
    "hist_loc_max_y_1": True,
    "hist_loc_max_x_2": True,
    "hist_loc_max_y_2": True,
    "bhatt_distance": True,
    # features on MIPs
    "Haralick_Sagital": False,
    "Haralick_Coronal": False,
    "Haralick_Axial": False,
    "CNR_Sagital": False,
    "CNR_Coronal": False,
    "CNR_Axial": False,
}

def get_enabled_features():
    return [feature for feature, enabled in FEATURES.items() if enabled]