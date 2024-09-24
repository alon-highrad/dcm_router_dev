import os
import sys
import shutil
from tempfile import TemporaryDirectory
from glob import glob
import time
from dcm_router.classifier.mri_classification.mri_metadata_classification import MRIMetadataClassification
from dcm_router.classifier.mri_classification.mri_sequence_classification import MRISequenceClassification
from dcm_router.classifier.mri_classification.brain_gd_classification import BrainGdClassification
from dcm_router.classifier.mri_classification.mri_brain_roi_classification import MRIBrainRoiClassification

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from converter import get_converter, ImageConverter

if __name__ == '__main__':

    metadata_classifier = MRIMetadataClassification()
    sequence_classifier = MRISequenceClassification()
    brain_roi_classifier = MRIBrainRoiClassification()
    brain_gd_classifier = BrainGdClassification()
    converter: ImageConverter = get_converter('icasbr')

    log_folder_path = 'log'
    # create a new log folder. overwrite if it exists.
    if os.path.exists(log_folder_path):
        shutil.rmtree(log_folder_path)
    os.makedirs(log_folder_path, exist_ok=True)

    time_list = []
    studies_with_no_t1w_ax = []
    t1w_ax_histogram = {}
    num_studies_with_gt = 0
    num_studies_match_gt = 0
    failed_gd_test = []
    dataset_folder = "/home/alon/projects/Segmed/"

    for patient in os.listdir(dataset_folder):
        patient_folder = os.path.join(dataset_folder, patient)
        for study in os.listdir(patient_folder):
            study_folder = os.path.join(patient_folder, study)
            if not os.path.exists(os.path.join(study_folder, 'GT')):
                continue
            num_studies_with_gt += 1
            dcm_folder = os.path.join(study_folder, 'DICOM')
            dcm_files = glob(dcm_folder + '/*/*.dcm')
            log_file_path = os.path.join(log_folder_path, f'{patient}_{study}.txt')

            start_time = time.time()

            metadata_classifier.set_files(dcm_files)
            metadata_classifier_res = metadata_classifier.execute()
            metadata_classifier_res.log_results(log_file_path)

            sequence_classifier.set_files(metadata_classifier_res.get_filtered_filenames([metadata_classifier.PASSED]))
            sequence_classifier_res = sequence_classifier.execute()
            sequence_classifier_res.log_results(log_file_path)

            # prepare the filtered file names. try t1w ax. If that does not exist, try any t1w. if that does not exists try (INVALID, INVALID).
            filtered_files = sequence_classifier_res.get_filtered_filenames([('t1w', 'ax')])
            if not filtered_files:
                studies_with_no_t1w_ax.append(patient + '_' + study)
                filtered_files = sequence_classifier_res.get_filtered_filenames([('t1w', 'sag')])
                filtered_files.extend(sequence_classifier_res.get_filtered_filenames([('t1w', 'cor')]))
            if not filtered_files:
                filtered_files = sequence_classifier_res.get_filtered_filenames([('INVALID', 'INVALID')])


            with TemporaryDirectory() as nif_tmp_dir:
                converter.convert(filtered_files, nif_tmp_dir)
                nifti_files = glob(nif_tmp_dir + '/*.nii*')
                t1w_ax_histogram[len(nifti_files)] = t1w_ax_histogram.get(len(nifti_files), 0) + 1

                # filter by brain roi
                brain_roi_classifier.set_files(nifti_files)
                brain_roi_classifier_res = brain_roi_classifier.execute()
                brain_roi_classifier_res.log_results(log_file_path)
                filtered_files = brain_roi_classifier_res.get_filtered_filenames([brain_roi_classifier.PASSED])

                # filter by gd
                brain_gd_classifier.set_files(filtered_files)
                brain_gd_classifier_res = brain_gd_classifier.execute()
                brain_gd_classifier_res.log_results(log_file_path)
            
            pred_list = brain_gd_classifier_res.get_filtered_filenames([brain_gd_classifier.PASSED])
            if not pred_list:
                failed_gd_test.append(patient + '_' + study)
                continue
            else:   
                # list the files in the GT folder, sorted by their size
                gt_files = os.listdir(os.path.join(study_folder, 'GT'))
                gt_files.sort(key=lambda x: os.path.getsize(os.path.join(study_folder, 'GT', x)))
                gt_scan = gt_files[-1]
                
                gt_file_name = os.path.basename(gt_scan)
                gt_file_name = gt_file_name.replace('DICOM_', '')
                if gt_file_name in pred_list[0][0]:
                    num_studies_match_gt += 1
                else:
                    failed_gd_test.append(patient + '_' + study)
            

            time_list.append(time.time() - start_time)

    print('\n\nstudies with no t1w ax:')
    for study in studies_with_no_t1w_ax:
        print(study)
    print('\n\n studies faild to find gt')
    for study in failed_gd_test:
        print(study)
    print('\n')
    print(f'Average time: {sum(time_list) / len(time_list)}')
    print(f'studies with gt: {num_studies_with_gt}')
    print(f'studies with gt that match: {num_studies_match_gt}')

    # add a plot of the t1w ax histogram to the log folder
    
    import matplotlib.pyplot as plt
    # Plot the histogram
    ax = plt.gca()
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.bar(t1w_ax_histogram.keys(), t1w_ax_histogram.values())
    plt.xlabel('Number of filtered files')
    plt.ylabel('Frequency')
    plt.title('Histogram of T1W AX filtered files')
    
    # Save the plot to the log folder
    plt.savefig(os.path.join(log_folder_path, 't1w_ax_histogram.png'))
    plt.close()

    
# def f():
#     filtered_dicom = dc.filter(dicoms)
#     #
#     t1gd = nc.filter(nifti,seg_dir)




    


    