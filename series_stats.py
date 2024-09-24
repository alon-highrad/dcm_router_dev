import shutil
from pathlib import Path
from glob import glob
from typing import List, Dict, Tuple, Union
import numpy as np
from pandas.core.dtypes.dtypes import re
import nibabel as nib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from tqdm import tqdm
import multiprocessing as mp
import json
import os
import itertools
import pandas as pd


def z_normalize(data):
    return (data - np.mean(data)) / np.std(data)


def create_normalized_histogram(data, bins):
    """Create a normalized histogram with fixed bins."""
    if type(data) != np.ndarray:
        data = np.array(data)
    counts, _ = np.histogram(data.flatten(), bins=bins)
    normalized_counts = counts / np.sum(counts)
    return normalized_counts


def scale_array(source_start, source_end, target_start, target_end, arr):
    """
    Scale a numpy array from a source interval to a target interval.

    Parameters:
    source_start (float): Start of the source interval
    source_end (float): End of the source interval
    target_start (float): Start of the target interval
    target_end (float): End of the target interval
    arr (numpy.ndarray): Input array scaled to the source interval

    Returns:
    numpy.ndarray: Scaled array to the target interval
    """
    # Calculate the scaling factor
    scale = (target_end - target_start) / (source_end - source_start)

    # Scale the array
    return target_start + (arr - source_start) * scale


def interpolate(sorted_values, probabilities, percentage):
    if len(sorted_values) != len(probabilities):
        raise ValueError("Arrays must be of the same size")

    if not 0 <= percentage <= 100:
        raise ValueError("Percentage must be between 0 and 100")

    cumsum = np.cumsum(probabilities)

    if cumsum[-1] == 0:
        raise ValueError("Sum of probabilities cannot be zero")

    # Normalize cumsum
    cumsum /= cumsum[-1]

    # Find the position where the percentage falls
    pos = np.searchsorted(cumsum, percentage / 100)

    if pos == 0:
        return sorted_values[0]
    elif pos == len(sorted_values):
        return sorted_values[-1]
    else:
        # Interpolate between the two nearest points
        x0, x1 = sorted_values[pos - 1], sorted_values[pos]
        y0, y1 = cumsum[pos - 1], cumsum[pos]
        return x0 + (x1 - x0) * (percentage / 100 - y0) / (y1 - y0)


def save_intensities_histogram(data, output_path, bins, chosen_percentile):
    """Create and save a normalized histogram of intensities for a single series using fixed bins."""
    # Assuming create_histogram function exists and is correctly implemented
    normalized_counts = create_normalized_histogram(data, bins)

    # Create a DataFrame for seaborn
    df = pd.DataFrame(
        {"Normalized Intensity": bins[:-1], "Percentage": normalized_counts * 100}
    )

    # Calculate percentiles based on the original data
    th = interpolate(bins[:-1], normalized_counts, chosen_percentile)
    # Create the plot using only seaborn
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.2)
    g = sns.FacetGrid(df, height=6, aspect=1.5)
    g.map(sns.barplot, "Normalized Intensity", "Percentage", color="skyblue")
    g.set_axis_labels("Normalized Intensity", "Percentage")
    g.fig.suptitle("Normalized Intensity Distribution")

    # Customize x-axis ticks
    max_labels = 10  # Maximum number of labels to display
    x_ticks = df["Normalized Intensity"][
        :: max(1, len(df) // max_labels)
    ]  # Select a subset of labels
    plt.xticks(ticks=x_ticks.index, labels=[f"{x:.1f}" for x in x_ticks])
    ticks, _ = plt.xticks()

    th = scale_array(bins[0], bins[-1], ticks[0], ticks[-1], np.array([th]))[0]
    plt.axvline(th, color="red", linestyle="--")
    # Adjust layout and save the figure
    g.tight_layout()
    g.fig.subplots_adjust(top=0.9)  # Adjust title position
    g.savefig(output_path + ".png")

    # Save histogram data
    np.savez(output_path + ".npz", counts=normalized_counts, bins=bins)


def load_and_process_file(file_path, percentile, bins):
    """Load a NIfTI file, calculate basic statistics, and save histogram."""
    img = nib.load(file_path)
    data = img.get_fdata()
    # Normalize
    data = z_normalize(data)
    # Create and save histogram
    hist_path = f"{os.path.splitext(file_path)[0]}_hist_perc_{percentile}"
    save_intensities_histogram(data, hist_path, bins, percentile)

    # Apply percentile filter
    # threshold = np.percentile(data, percentile)
    # normalized_filtered_data = z_normalize(data[data >= threshold])

    return {
        "mean_intensity": np.mean(data),
        "std_intensity": np.std(data),
        "skewness": stats.skew(data),
        "kurtosis": stats.kurtosis(data),
        "histogram_path": hist_path + ".npz",
    }


def bhattacharyya_distance(hist1: Union[str, tuple], hist2: Union[str, tuple]):
    """Calculate Bhattacharyya distance between two histograms."""
    if type(hist1) == str and type(hist2) == str:
        if hist1 == hist2:
            return None

        hist1 = np.load(hist1)
        hist2 = np.load(hist2)
        p = hist1["counts"] / np.sum(hist1["counts"])
        q = hist2["counts"] / np.sum(hist2["counts"])

    elif type(hist1) == tuple and type(hist2) == tuple:
        p = hist1[0] / np.sum(hist1[0])
        q = hist2[0] / np.sum(hist2[0])
    else:
        print("invalid input")
        return
    # Calculate Bhattacharyya coefficient
    bc = np.sum(np.sqrt(p * q))

    # Calculate Bhattacharyya distance
    return -np.log(bc)


def process_group(file_list, percentile, bins):
    """Process a group of files using multiprocessing."""
    with mp.Pool(processes=mp.cpu_count() - 10) as pool:
        results = list(
            tqdm(
                pool.starmap(
                    load_and_process_file, [(f, percentile, bins) for f in file_list]
                ),
                total=len(file_list),
            )
        )

    return {
        "mean_intensities": [r["mean_intensity"] for r in results],
        "std_intensities": [r["std_intensity"] for r in results],
        "skewness_values": [r["skewness"] for r in results],
        "kurtosis_values": [r["kurtosis"] for r in results],
        "histogram_paths": [r["histogram_path"] for r in results],
    }


def plot_normalized_histograms(ax, data1, data2, label1, label2, title, xlabel):
    """Plot normalized histograms for two datasets on the same axes."""
    bins = np.linspace(min(min(data1), min(data2)), max(max(data1), max(data2)), 30)

    # Calculate normalized histogram counts
    normalized_counts1 = create_normalized_histogram(data1, bins)
    normalized_counts2 = create_normalized_histogram(data2, bins)

    # Plot the normalized histograms
    ax.bar(
        bins[:-1],
        normalized_counts1 * 100,
        width=np.diff(bins),
        alpha=0.5,
        label=label1,
    )
    ax.bar(
        bins[:-1],
        normalized_counts2 * 100,
        width=np.diff(bins),
        alpha=0.5,
        label=label2,
    )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Percentage")
    ax.legend()


def plot_scatter(ax, x1, y1, x2, y2, label1, label2, title, xlabel, ylabel):
    """Plot scatter plot for two datasets on the same axes."""
    ax.scatter(x1, y1, alpha=0.5, label=label1)
    ax.scatter(x2, y2, alpha=0.5, label=label2)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()


def create_visualizations(results, output_file):
    """Create and save visualizations comparing two groups."""
    fig, axes = plt.subplots(2, 2, figsize=(20, 20))

    plot_normalized_histograms(
        axes[0, 0],
        results["Contrast"]["mean_intensities"],
        results["Non-Contrast"]["mean_intensities"],
        "Contrast",
        "Non-Contrast",
        "Distribution of Mean Intensities",
        "Mean Intensity",
    )

    plot_normalized_histograms(
        axes[0, 1],
        results["Contrast"]["std_intensities"],
        results["Non-Contrast"]["std_intensities"],
        "Contrast",
        "Non-Contrast",
        "Distribution of Standard Deviations",
        "Standard Deviation",
    )

    plot_scatter(
        axes[1, 0],
        results["Contrast"]["skewness_values"],
        results["Contrast"]["kurtosis_values"],
        results["Non-Contrast"]["skewness_values"],
        results["Non-Contrast"]["kurtosis_values"],
        "Contrast",
        "Non-Contrast",
        "Skewness vs Kurtosis",
        "Skewness",
        "Kurtosis",
    )

    plot_scatter(
        axes[1, 1],
        results["Contrast"]["mean_intensities"],
        results["Contrast"]["std_intensities"],
        results["Non-Contrast"]["mean_intensities"],
        results["Non-Contrast"]["std_intensities"],
        "Contrast",
        "Non-Contrast",
        "Mean Intensity vs Standard Deviation",
        "Mean Intensity",
        "Standard Deviation",
    )

    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()


def calculate_overall_stats(results):
    """Calculate overall statistics for each group."""
    overall_stats = {}
    for group_name, group_data in results.items():
        overall_stats[group_name] = {
            "mean_intensity": np.mean(group_data["mean_intensities"]),
            "std_intensity": np.mean(group_data["std_intensities"]),
            "mean_skewness": np.mean(group_data["skewness_values"]),
            "mean_kurtosis": np.mean(group_data["kurtosis_values"]),
        }
    return overall_stats


def calculate_bhattacharyya_distances(histogram_paths1, histogram_paths2):
    """Calculate Bhattacharyya distances between all pairs of histograms."""
    distances = dict()
    number_of_nones = 0
    for hist1, hist2 in tqdm(
        itertools.product(histogram_paths1, histogram_paths2),
        total=len(histogram_paths1) * len(histogram_paths2),
    ):
        distance = bhattacharyya_distance(hist1, hist2)
        if distance is not None:
            distances[(hist1, hist2)] = distance
        else:
            number_of_nones += 1
    print(f"{number_of_nones} None values in distance calculation")
    return distances


def print_most_distanced_pairs(distances, n=3):
    """Print the n most distanced pairs of scans."""
    sorted_distances = sorted(distances.items(), key=lambda x: x[1], reverse=True)
    print(f"\nTop {n} most distanced pairs:")
    for (hist1, hist2), distance in sorted_distances[:n]:
        print(f"Distance: {distance:.4f}")
        print(f"Scan 1: {os.path.basename(hist1)}")
        print(f"Scan 2: {os.path.basename(hist2)}")
        print()


def plot_bhattacharyya_distances(distances, output_file, title):
    """Plot the distribution of Bhattacharyya distances."""
    plt.figure(figsize=(10, 6))
    plt.hist(list(distances.values()), bins=50, density=True)
    plt.title(f"Distribution of Bhattacharyya Distances - {title}")
    plt.xlabel("Bhattacharyya Distance")
    plt.ylabel("Density")
    plt.savefig(output_file)
    plt.close()


def process_file_for_intensity_range(file_path):
    """Process a single file to find its intensity range."""
    img = nib.load(file_path)
    data = img.get_fdata()
    data = z_normalize(data)
    return np.min(data), np.max(data)


def determine_global_intensity_range(file_list):
    """Determine global min and max intensity using parallel processing."""
    print("Determining global intensity range...")
    with mp.Pool(processes=mp.cpu_count() - 1) as pool:
        results = list(
            tqdm(
                pool.imap(process_file_for_intensity_range, file_list),
                total=len(file_list),
            )
        )

    global_min = min(result[0] for result in results)
    global_max = max(result[1] for result in results)

    print(f"Global intensity range: {global_min} , {global_max}")
    return global_min, global_max


def analyze_mri_dataset(file_list_contrast, file_list_non_contrast, percentile):
    """Main function to analyze MRI dataset."""
    # Determine global min and max for bin range
    all_files = file_list_contrast + file_list_non_contrast
    global_min, global_max = determine_global_intensity_range(all_files)
    print(f"Global intensity range: {global_min} , {global_max}")
    # Create fixed bins
    bins = np.linspace(global_min, global_max, 201)  # 200 bins
    np.savez(
        bins,
    )
    groups = {"Contrast": file_list_contrast, "Non-Contrast": file_list_non_contrast}

    results = {}
    for group_name, file_list in groups.items():
        print(f"Processing {group_name} group...")
        results[group_name] = process_group(file_list, percentile, bins)

    create_visualizations(results, "mri_dataset_analysis_comparison.png")

    overall_stats = calculate_overall_stats(results)
    return results, overall_stats


def calculate_all_bhattacharyya_distances(results):
    """Calculate Bhattacharyya distances for all combinations."""
    distances = {}

    # Between Contrast and Non-Contrast
    print("Calculating Bhattacharyya distances between Contrast and Non-Contrast...")
    distances["Contrast_vs_NonContrast"] = calculate_bhattacharyya_distances(
        results["Contrast"]["histogram_paths"],
        results["Non-Contrast"]["histogram_paths"],
    )
    print_most_distanced_pairs(distances["Contrast_vs_NonContrast"])
    plot_bhattacharyya_distances(
        distances["Contrast_vs_NonContrast"],
        "bhattacharyya_distances_contrast_vs_noncontrast.png",
        "Contrast vs Non-Contrast",
    )

    # Within Contrast
    print("Calculating Bhattacharyya distances within Contrast group...")
    distances["Within_Contrast"] = calculate_bhattacharyya_distances(
        results["Contrast"]["histogram_paths"], results["Contrast"]["histogram_paths"]
    )
    print_most_distanced_pairs(distances["Within_Contrast"])
    plot_bhattacharyya_distances(
        distances["Within_Contrast"],
        "bhattacharyya_distances_within_contrast.png",
        "Within Contrast",
    )

    # Within Non-Contrast
    print("Calculating Bhattacharyya distances within Non-Contrast group...")
    distances["Within_NonContrast"] = calculate_bhattacharyya_distances(
        results["Non-Contrast"]["histogram_paths"],
        results["Non-Contrast"]["histogram_paths"],
    )
    print_most_distanced_pairs(distances["Within_NonContrast"])
    plot_bhattacharyya_distances(
        distances["Within_NonContrast"],
        "bhattacharyya_distances_within_noncontrast.png",
        "Within Non-Contrast",
    )

    return distances


def kennad_stone(group: List, distances: Dict, n: int):
    """
    algorithm for finding the best spanning subset

    Args
        group: List of group members
        distances: Dict, every tule of group members is a key. The values are the distances
        n: the size of the desired subgroup
    """

    def farthest_from_a_group(group_a, group_b):

        number_of_nones = 0
        _best = (-np.inf, None)
        for x in group_a:
            cur_score = 0
            for y in group_b:
                if (x, y) in distances.keys():
                    cur_score += distances[x, y]
                elif (y, x) in distances.keys():
                    cur_score += distances[y, x]
                else:
                    number_of_nones += 1
            if cur_score > _best[0]:
                _best = (cur_score, x)

        print(f"{number_of_nones} Nones in kennad_stone")
        print(_best[1])
        return _best

    S = group
    L = []
    scores = []

    res = farthest_from_a_group(S, S)
    L.append(res[1])
    S.remove(res[1])
    scores.append(res[0])
    for _ in range(n - 1):
        res = farthest_from_a_group(S, L)
        L.append(res[1])
        S.remove(res[1])
        scores.append(res[0])

    return L, scores


def save_referrence_histograms(
    hists_path_list: List[Path], n: int, dest_dir_path: Path
):
    """
    save best spanning contrast enhanced histograms from 'hist_lst' that will be used as referrence by the random forest model

    Args:
        hists_path_list (List): list of contrast enhanced histograms candidates
        n (int): the number of referrences to save
        dest_dir_path (Path): path of a directory to save the referrences
    """

    distances = calculate_bhattacharyya_distances(hists_path_list, hists_path_list)
    res = kennad_stone(contrast_hists, distances, n)[0]
    os.makedirs(dest_dir_path, exist_ok=True)
    for p in res:
        shutil.copy(p, os.path.join(dest_dir_path, os.path.basename(p)))


def load_referrence_histograms(referrences_dir_path):
    return [
        np.load(os.path.join(referrences_dir_path, p))
        for p in os.listdir(referrences_dir_path)
    ]


def create_distance_table(distances, first_file_list, second_file_list, output_path):
    """
    Create a distance table and save it as an Excel file.
    
    Args:
    distances (dict): Dictionary of distances between scan pairs
    file_list (list): List of scan file paths
    output_path (str): Path to save the Excel file
    """
    # Create a DataFrame with scan file paths as both index and columns
    df = pd.DataFrame(index=first_file_list, columns=second_file_list)
    None_count = 0
    # Fill the DataFrame with distances
    for (scan1, scan2), distance in distances.items():
        if distance is None:
            df.at[scan1, scan2] = np.round(distance, 2)
            df.at[scan2, scan1] = np.round(distance, 2) 
        else:
            df.at[scan1, scan2] = np.round(distance, 2)
    
    print(f"{None_count} None values in distance calculation")
    # Fill diagonal with zeros (distance to self)
    if first_file_list == second_file_list:
        for scan in first_file_list:
            df.at[scan, scan] = 0
    
    # Fill NaN values with a placeholder (e.g., -1) if any remain
    df = df.fillna(-1)
    
    # Save to Excel
    df.to_excel(output_path)
    print(f"Distance table saved to {output_path}")


def plot_row_means_histogram(excel_path, output_path):
    """
    Plots a histogram of the mean of every row of a distance table and prints paths with highest, lowest, and median mean values.
    
    Args:
    excel_path (str): Path to the Excel file containing the distance table
    output_path (str): Path to save the output histogram plot
    """
    # Read the Excel file
    df = pd.read_excel(excel_path, index_col=0)
    
    # Calculate the mean of each row, excluding -1 values (which were used as placeholders for NaN)
    row_means = df[df != -1].mean(axis=1)
    
    # Find rows with highest, lowest, and median mean values
    highest_mean_row = row_means.idxmax()
    lowest_mean_row = row_means.idxmin()
    median_mean_value = row_means.median()
    median_mean_row = row_means.iloc[(row_means - median_mean_value).abs().argsort()[:1]].index[0]
    
    print(f"Row with highest mean value: {highest_mean_row}")
    print(f"Highest mean value: {row_means[highest_mean_row]:.2f}")
    print(f"Row with lowest mean value: {lowest_mean_row}")
    print(f"Lowest mean value: {row_means[lowest_mean_row]:.2f}")
    print(f"Row with median mean value: {median_mean_row}")
    print(f"Median mean value: {row_means[median_mean_row]:.2f}")
    
    # Calculate statistics
    mean = row_means.mean()
    std = row_means.std()
    median = row_means.median()
    
    # Create the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(row_means, bins=30, edgecolor='black')
    plt.title('Histogram of Row Means in Distance Table')
    plt.xlabel('Mean Distance')
    plt.ylabel('Frequency')
    
    # Add text box with statistics
    stats_text = f'Mean: {mean:.2f}\nSTD: {std:.2f}\nMedian: {median:.2f}'
    plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    print(f"Histogram saved to {output_path}")


if __name__ == "__main__":

    res_dir = './hist_results'
    os.makedirs(res_dir, exist_ok=True)

    # Load both JSON files
    with open('alon_labels_filtered.json', 'r') as f:
        alon_labels = json.load(f)
    
    with open('oo_test_labels_filtered.json', 'r') as f:
        oo_test_labels = json.load(f)
    
    # Extract contrast and non-contrast series from both datasets
    alon_contrast_series = []
    alon_non_contrast_series = []
    oo_test_contrast_series = []
    oo_test_non_contrast_series = []
    
    for study_name, series_dict in alon_labels.items():
        for scan_path, label in series_dict.items():
            if label == 1:  
                alon_contrast_series.append(scan_path)
            elif label == 0:
                alon_non_contrast_series.append(scan_path)
    
    for study_name, series_dict in oo_test_labels.items():
        for scan_path, label in series_dict.items():
            if label == 1:
                oo_test_contrast_series.append(scan_path)
            elif label == 0:
                oo_test_non_contrast_series.append(scan_path)
    
    print(f"Number of contrast series in Hadassah dataset: {len(alon_contrast_series)}")
    print(f"Number of non-contrast series in Hadassah dataset: {len(alon_non_contrast_series)}")
    print(f"Number of contrast series in OOD Test dataset: {len(oo_test_contrast_series)}")
    print(f"Number of non-contrast series in OOD Test dataset: {len(oo_test_non_contrast_series)}")
    
    percentile = 0
    bins = np.linspace(-3, 3, 201)

    # Process contrast and non-contrast series if histograms don't exist
    def process_series(series_list):
        for file_path in tqdm(series_list):
            hist_path = f"{os.path.splitext(file_path)[0]}_hist_perc_{percentile}.npz"
            if not os.path.exists(hist_path):
                load_and_process_file(file_path, percentile, bins)

    print("\nProcessing contrast series...")
    process_series(alon_contrast_series)
    process_series(oo_test_contrast_series)

    print("\nProcessing non-contrast series...")
    process_series(alon_non_contrast_series)
    process_series(oo_test_non_contrast_series)

    # Update file paths to point to the generated histogram files
    alon_contrast_series = [path.replace(".gz", f"_hist_perc_{percentile}.npz") for path in alon_contrast_series]
    alon_non_contrast_series = [path.replace(".gz", f"_hist_perc_{percentile}.npz") for path in alon_non_contrast_series]
    oo_test_contrast_series = [path.replace(".gz", f"_hist_perc_{percentile}.npz") for path in oo_test_contrast_series]
    oo_test_non_contrast_series = [path.replace(".gz", f"_hist_perc_{percentile}.npz") for path in oo_test_non_contrast_series]

    # Calculate Bhattacharyya distances for contrast series
    print("\nCalculating distances for contrast series...")
    
    print("Calculating Bhattacharyya distances between Hadassah and OOD Test contrast series...")
    hadassah_ood_contrast_distances = calculate_bhattacharyya_distances(
        alon_contrast_series,
        oo_test_contrast_series
    )
    print_most_distanced_pairs(hadassah_ood_contrast_distances)
    plot_bhattacharyya_distances(
        hadassah_ood_contrast_distances,
        os.path.join(res_dir, "bhattacharyya_distances_hadassah_vs_ood_test_contrast.png"),
        "Hadassah vs OOD Test Contrast Series"
    )
    
    print("Calculating Bhattacharyya distances within Hadassah contrast dataset...")
    hadassah_contrast_distances = calculate_bhattacharyya_distances(
        alon_contrast_series,
        alon_contrast_series
    )
    print_most_distanced_pairs(hadassah_contrast_distances)
    plot_bhattacharyya_distances(
        hadassah_contrast_distances,
        os.path.join(res_dir, "bhattacharyya_distances_within_hadassah_contrast.png"),
        "Within Hadassah Contrast Series"
    )
    
    print("Calculating Bhattacharyya distances within OOD Test contrast dataset...")
    ood_test_contrast_distances = calculate_bhattacharyya_distances(
        oo_test_contrast_series,
        oo_test_contrast_series
    )
    print_most_distanced_pairs(ood_test_contrast_distances)
    plot_bhattacharyya_distances(
        ood_test_contrast_distances,
        os.path.join(res_dir, "bhattacharyya_distances_within_ood_test_contrast.png"),
        "Within OOD Test Contrast Series"
    )

    # Calculate Bhattacharyya distances for non-contrast series
    print("\nCalculating distances for non-contrast series...")
    
    print("Calculating Bhattacharyya distances between Hadassah and OOD Test non-contrast series...")
    hadassah_ood_non_contrast_distances = calculate_bhattacharyya_distances(
        alon_non_contrast_series,
        oo_test_non_contrast_series
    )
    print_most_distanced_pairs(hadassah_ood_non_contrast_distances)
    plot_bhattacharyya_distances(
        hadassah_ood_non_contrast_distances,
        os.path.join(res_dir, "bhattacharyya_distances_hadassah_vs_ood_test_non_contrast.png"),
        "Hadassah vs OOD Test Non-Contrast Series"
    )
    
    print("Calculating Bhattacharyya distances within Hadassah non-contrast dataset...")
    hadassah_non_contrast_distances = calculate_bhattacharyya_distances(
        alon_non_contrast_series,
        alon_non_contrast_series
    )
    print_most_distanced_pairs(hadassah_non_contrast_distances)
    plot_bhattacharyya_distances(
        hadassah_non_contrast_distances,
        os.path.join(res_dir, "bhattacharyya_distances_within_hadassah_non_contrast.png"),
        "Within Hadassah Non-Contrast Series"
    )
    
    print("Calculating Bhattacharyya distances within OOD Test non-contrast dataset...")
    ood_test_non_contrast_distances = calculate_bhattacharyya_distances(
        oo_test_non_contrast_series,
        oo_test_non_contrast_series
    )
    print_most_distanced_pairs(ood_test_non_contrast_distances)
    plot_bhattacharyya_distances(
        ood_test_non_contrast_distances,
        os.path.join(res_dir, "bhattacharyya_distances_within_ood_test_non_contrast.png"),
        "Within OOD Test Non-Contrast Series"
    )

    # Calculate Bhattacharyya distances between contrast and non-contrast series
    print("\nCalculating distances between contrast and non-contrast series...")
    
    print("Calculating Bhattacharyya distances between Hadassah contrast and non-contrast series...")
    hadassah_contrast_vs_non_contrast_distances = calculate_bhattacharyya_distances(
        alon_contrast_series,
        alon_non_contrast_series
    )
    print_most_distanced_pairs(hadassah_contrast_vs_non_contrast_distances)
    plot_bhattacharyya_distances(
        hadassah_contrast_vs_non_contrast_distances,
        os.path.join(res_dir, "bhattacharyya_distances_hadassah_contrast_vs_non_contrast.png"),
        "Hadassah Contrast vs Non-Contrast Series"
    )
    
    print("Calculating Bhattacharyya distances between OOD Test contrast and non-contrast series...")
    ood_test_contrast_vs_non_contrast_distances = calculate_bhattacharyya_distances(
        oo_test_contrast_series,
        oo_test_non_contrast_series
    )
    print_most_distanced_pairs(ood_test_contrast_vs_non_contrast_distances)
    plot_bhattacharyya_distances(
        ood_test_contrast_vs_non_contrast_distances,
        os.path.join(res_dir, "bhattacharyya_distances_ood_test_contrast_vs_non_contrast.png"),
        "OOD Test Contrast vs Non-Contrast Series"
    )

    print("Calculating Bhattacharyya distances between Hadassah contrast and OOD Test non-contrast series...")
    hadassah_contrast_vs_ood_test_non_contrast_distances = calculate_bhattacharyya_distances(
        alon_contrast_series,
        oo_test_non_contrast_series
    )
    print_most_distanced_pairs(hadassah_contrast_vs_ood_test_non_contrast_distances)
    plot_bhattacharyya_distances(
        hadassah_contrast_vs_ood_test_non_contrast_distances,
        os.path.join(res_dir, "bhattacharyya_distances_hadassah_contrast_vs_ood_test_non_contrast.png"),
        "Hadassah Contrast vs OOD Test Non-Contrast Series"
    )

    print("Calculating Bhattacharyya distances between Hadassah non-contrast and OOD Test contrast series...")
    hadassah_non_contrast_vs_ood_test_contrast_distances = calculate_bhattacharyya_distances(
        alon_non_contrast_series,
        oo_test_contrast_series
    )
    print_most_distanced_pairs(hadassah_non_contrast_vs_ood_test_contrast_distances)
    plot_bhattacharyya_distances(
        hadassah_non_contrast_vs_ood_test_contrast_distances,
        os.path.join(res_dir, "bhattacharyya_distances_hadassah_non_contrast_vs_ood_test_contrast.png"),
        "Hadassah Non-Contrast vs OOD Test Contrast Series"
    )

    # Create and save distance tables
    distance_tables = {
        "hadassah_vs_ood_contrast": (hadassah_ood_contrast_distances, alon_contrast_series, oo_test_contrast_series),
        "within_hadassah_contrast": (hadassah_contrast_distances, alon_contrast_series, alon_contrast_series),
        "within_ood_contrast": (ood_test_contrast_distances, oo_test_contrast_series, oo_test_contrast_series),
        "hadassah_vs_ood_non_contrast": (hadassah_ood_non_contrast_distances, alon_non_contrast_series, oo_test_non_contrast_series),
        "within_hadassah_non_contrast": (hadassah_non_contrast_distances, alon_non_contrast_series, alon_non_contrast_series),
        "within_ood_non_contrast": (ood_test_non_contrast_distances, oo_test_non_contrast_series, oo_test_non_contrast_series),
        "hadassah_contrast_vs_non_contrast": (hadassah_contrast_vs_non_contrast_distances, alon_contrast_series, alon_non_contrast_series),
        "ood_test_contrast_vs_non_contrast": (ood_test_contrast_vs_non_contrast_distances, oo_test_contrast_series, oo_test_non_contrast_series),
        "hadassah_contrast_vs_ood_test_non_contrast": (hadassah_contrast_vs_ood_test_non_contrast_distances, alon_contrast_series, oo_test_non_contrast_series),
        "hadassah_non_contrast_vs_ood_test_contrast": (hadassah_non_contrast_vs_ood_test_contrast_distances, alon_non_contrast_series, oo_test_contrast_series)
    }

    for name, (distances, first_list, second_list) in distance_tables.items():
        create_distance_table(
            distances,
            first_list,
            second_list,
            os.path.join(res_dir, f"{name}_distances.xlsx")
        )

    # Plot histograms for all distance tables
    for name in distance_tables.keys():
        plot_row_means_histogram(
            os.path.join(res_dir, f"{name}_distances.xlsx"),
            os.path.join(res_dir, f"{name}_distances_mean_histogram.png")
        )



