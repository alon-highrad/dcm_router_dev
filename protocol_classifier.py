from os.path import isdir, basename
from time import time
from typing import List
from joblib import load
from glob import glob
from multiprocessing import Pool
from tools import extract_numerical_measures
import numpy as np

# global variables
MODEL_PATH = 'protocol_chooser_model.joblib'


def sort_cases_by_phase_priority(cases_paths: List[str], model_path: str = MODEL_PATH):
    """
    Given a list of case paths, the function filters out the cases that their x and y axes are not 512 × 512, or if they
    are consisted of less than 3 'z' axis slices. If there is in the rest of the list more than 1 case, we send them all
    to the fitted model described above trained on the train-set, and return them sorted according to the probability
    prediction score they get, in a descending order.

    :param cases_paths:
    :param model_path:

    :return:
    """
    # loading the classifier model
    model = load(model_path)

    # extracting the numerical measures of the data
    with Pool() as pool:
        cases_measures = pool.map(extract_numerical_measures, cases_paths)

    priority_list = np.array([-1 if len(case) == 0 else 0 for case in cases_measures], dtype=np.float)

    cases_measures = [case for case in cases_measures if len(case) != 0]

    if cases_measures:

        if len(cases_measures) > 1:
            cases_measures = np.array(cases_measures)

            # predicting the cases
            y_predict_proba = model.predict_proba(cases_measures)[:, 1]

            priority_list[priority_list == 0] = y_predict_proba

        return list(np.take(cases_paths, np.argsort(priority_list)[::-1]))

    else:
        return np.copy(cases_paths)


def get_file_names_ordered_by_phase_priority(dir_name: str, model_path: str = MODEL_PATH):
    file_names = [file_name for file_name in glob(f'{dir_name}/*') if file_name.endswith('.nii.gz')]
    return sort_cases_by_phase_priority(file_names, model_path)


if __name__ == '__main__':
    # usage example
    # dir_name_with_several_nifti_files_in_it = 'some/directory/path'
    # fils_orderd_by_phase_priority = get_file_names_ordered_by_phase_priority(dir_name_with_several_nifti_files_in_it)

    # cases = []
    # for file in glob('/cs/casmip/public/for_aviv/for_shalom/A_Y_04_05_2020/*'):
    #     if isdir(file) and file.lower().endswith('old'):
    #         for inner_file in glob(f'{file}/*'):
    #             cases.append(inner_file)
    #     if basename(file).startswith('DICOM'):
    #         cases.append(file)
    #
    # ordered_cases = sort_cases_by_phase_priority(cases)
    #
    # print(ordered_cases)

    t = time()

    dataset_path = '/cs/casmip/public/for_aviv/livers-data_01.11.2020'

    for dir in sorted(glob(f'{dataset_path}/*')):
        t1 = time()
        case = []
        for file in glob(f'{dir}/*'):
            if file.endswith('.gz'):
                case.append(file)

        sort_scans = sort_cases_by_phase_priority(case)
        txt_file_path = f'{dir}/protocol_chooser_result.txt'
        with open(txt_file_path, 'w') as file:
            for i, scan in enumerate(sort_scans):
                file.write(f'{i + 1}:    {basename(scan)}\n')

        print(f'Finished case {basename(dir)} in {time() - t1:.2f} sec.')

    print(f'\n --- Finished all case in {time() - t:.2f} sec. ---')
