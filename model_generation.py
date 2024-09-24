from typing import List, Tuple
from sklearn.ensemble import RandomForestClassifier
from joblib import dump
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from scikitplot.metrics import plot_precision_recall, plot_roc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# Add this at the beginning of the file, after the import statements
plt.rcParams.update({'font.size': 14})

def plot_roc_curve(fprs, tprs, save_file_name='roc_curve.jpg'):
    """Plot the Receiver Operating Characteristic from a list
    of true positive rates and false positive rates."""

    # Initialize useful lists + the plot axes.
    tprs_interp = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    f, ax = plt.subplots(figsize=(16, 12))  # Increased figure size

    # Plot ROC for each K-Fold + compute AUC scores.
    for i, (fpr, tpr) in enumerate(zip(fprs, tprs)):
        tprs_interp.append(np.interp(mean_fpr, fpr, tpr))
        tprs_interp[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        ax.plot(fpr, tpr, lw=1, alpha=0.3,
                label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

    # Plot the luck line according to the balance of the data
    # base_rate = sum(y_test) / len(y_test)
    # plt.plot([0, 1], [base_rate, base_rate], linestyle='--', lw=2, color='r',
    #          label='Luck', alpha=.8)

    # Plot the mean ROC.
    mean_tpr = np.mean(tprs_interp, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color='b',
            label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
            lw=2, alpha=.8)

    # Plot the standard deviation around the mean ROC.
    std_tpr = np.std(tprs_interp, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                    label=r'$\pm$ 1 std. dev.')

    # Increase font size for title and labels
    ax.set_xlabel('False Positive Rate', fontsize=16)
    ax.set_ylabel('True Positive Rate', fontsize=16)
    ax.set_title('Receiver operating characteristic', fontsize=18)
    ax.legend(loc="lower right", fontsize=12)

    if save_file_name is not None:
        plt.savefig(save_file_name)

    return (f, ax)


def plot_precision_recall_curve(fprs, tprs, a_ps, no_skill, save_file_name='precision_recall_curve.jpg'):
    """Plot the Precision Recall curve from a list
    of true positive rates and false positive rates."""

    # Initialize useful lists + the plot axes.
    tprs_interp = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    f, ax = plt.subplots(figsize=(16, 12))  # Increased figure size

    # Plot ROC for each K-Fold + compute AUC scores.
    for i, (fpr, tpr) in enumerate(zip(fprs, tprs)):
        tprs_interp.append(np.interp(mean_fpr, fpr, tpr))
        tprs_interp[-1][0] = 1.0
        roc_auc = a_ps[i]
        aucs.append(roc_auc)
        ax.plot(fpr, tpr, lw=1, alpha=0.3,
                label=f'Average precision{f" fold {i + 1}" if len(fprs) > 1 else ""} (AP = {roc_auc:0.2f})')


    # Plot the mean ROC.
    mean_tpr = np.mean(tprs_interp, axis=0)
    mean_tpr[-1] = 0.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color='b',
            label=r'Mean A_P (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
            lw=2, alpha=.8)

    # Plot the standard deviation around the mean ROC.
    std_tpr = np.std(tprs_interp, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                    label=r'$\pm$ 1 std. dev.')

    # Increase font size for title and labels
    ax.set_xlabel('Recall', fontsize=16)
    ax.set_ylabel('Precision', fontsize=16)
    ax.set_title('Precision-Recall curve', fontsize=18)
    ax.legend(loc="lower right", fontsize=12)

    if save_file_name is not None:
        plt.savefig(save_file_name)

    return (f, ax)


def compute_roc_auc(y_test, y_predict_proba):
    fpr, tpr, thresholds = roc_curve(y_test, y_predict_proba)
    auc_score = auc(fpr, tpr)
    precision, recall, threholds = precision_recall_curve(y_test, y_predict_proba)
    a_p = average_precision_score(y_test, y_predict_proba)
    return fpr, tpr, auc_score, precision, recall, a_p


def get_number_of_false_cases(y_predict_proba: np.ndarray, cases_df: pd.DataFrame,
                              consider_as_falsies: List[int]) -> Tuple[List[int], int]:
    df = cases_df.assign(proba=y_predict_proba)
    df.sort_values(by=['proba'], inplace=True, ascending=False)
    place_of_errors = df.groupby('Case ID', sort=False).apply(lambda case: np.argmax(case['Label'])).array
    n_cases = place_of_errors.size
    falsies = [(place_of_errors >= i).sum() for i in consider_as_falsies]

    return falsies, n_cases


def get_failed_positive_cases(y_true, y_pred_proba, cases_df, top_n=10):
    """Get the top N positive cases that the model fails on."""
    df = cases_df.copy()
    df['true_label'] = y_true
    df['pred_proba'] = y_pred_proba
    
    # Filter for positive cases only
    positive_cases = df[df['true_label'] == 1]
    
    # Sort by prediction probability (ascending) to get the worst predictions
    failed_cases = positive_cases.sort_values('pred_proba')
    
    return failed_cases[['Case ID', 'pred_proba']].head(top_n)


def print_and_plot_validation_results(data: pd.DataFrame, unique_cases: np.ndarray,
                                      train: np.ndarray, model: RandomForestClassifier,
                                      experiment_dir: str,
                                      n_cv_folds: int = 5, random_state: int = 8):
    print('\n ----- Validation results -----\n')

    avg_res_1st_place = []
    avg_res_2nd_place = []
    avg_res_3rd_place = []
    fprs, tprs, precisions, recalls, a_ps = [], [], [], [], []
    all_failed_cases = []

    for i, (train_split, validation) in \
            enumerate(KFold(n_splits=n_cv_folds, shuffle=True, random_state=random_state).split(train)):

        relevant_train_cases = data.where(data['Case ID'].isin(unique_cases[train_split])).dropna()
        relevant_validation_cases = data.where(data['Case ID'].isin(unique_cases[validation])).dropna()

        x_train = relevant_train_cases.drop(['Scan ID', 'Case ID', 'Label'], axis=1)
        y_train = relevant_train_cases['Label']

        x_validation = relevant_validation_cases.drop(['Scan ID', 'Case ID', 'Label'], axis=1)
        y_validation = relevant_validation_cases['Label']

        model.fit(x_train, y_train)

        y_predict_proba = model.predict_proba(x_validation)[:, 1]

        (n_false_as_1, n_false_as_2, n_false_as_3), n_cases = get_number_of_false_cases(y_predict_proba,
                                                                                        relevant_validation_cases,
                                                                                        consider_as_falsies=[1, 2, 3])

        print(f'Fold {i + 1}:')
        print(f'Falsies as 1st place: {n_false_as_1}/{n_cases} = {100 * n_false_as_1 / n_cases:.2f}%')
        print(f'Falsies as 2nd place: {n_false_as_2}/{n_cases} = {100 * n_false_as_2 / n_cases:.2f}%')
        print(f'Falsies as 3rd place: {n_false_as_3}/{n_cases} = {100 * n_false_as_3 / n_cases:.2f}%')
        
        # Get and print failed positive cases
        failed_cases = get_failed_positive_cases(y_validation, y_predict_proba, relevant_validation_cases)
        print("\nTop 10 failed positive cases:")
        print(failed_cases)
        print('----------------------------------')

        all_failed_cases.append(failed_cases)

        avg_res_1st_place += [100 * n_false_as_1 / n_cases]
        avg_res_2nd_place += [100 * n_false_as_2 / n_cases]
        avg_res_3rd_place += [100 * n_false_as_3 / n_cases]

        fpr, tpr, _, precision, recall, a_p = compute_roc_auc(y_validation, y_predict_proba)

        fprs.append(fpr)
        tprs.append(tpr)
        precisions.append(precision)
        recalls.append(recall)
        a_ps.append(a_p)
        no_skill = len(validation[validation == 1]) / len(validation)

    print(f'\nAverage over the folds:')
    print(
        f'Falsies as 1st place: {np.array(avg_res_1st_place).mean():.2f}%  (std={np.array(avg_res_1st_place).std():.2f})')
    print(
        f'Falsies as 2nd place: {np.array(avg_res_2nd_place).mean():.2f}%  (std={np.array(avg_res_2nd_place).std():.2f})')
    print(
        f'Falsies as 3rd place: {np.array(avg_res_3rd_place).mean():.2f}%  (std={np.array(avg_res_3rd_place).std():.2f})')
    
    # Aggregate failed cases across all folds
    all_failed_cases_df = pd.concat(all_failed_cases)
    top_failed_cases = all_failed_cases_df.groupby('Case ID')['pred_proba'].mean().sort_values().head(10)
    
    print("\nTop 10 consistently failed positive cases across all folds:")
    print(top_failed_cases)
    print('')

    plot_roc_curve(fprs, tprs, os.path.join(experiment_dir, 'cv_roc_curve.jpg'))
    plot_precision_recall_curve(precisions, recalls, a_ps, no_skill, os.path.join(experiment_dir, 'cv_precision_recall_curve.jpg'))


def print_and_plot_testing_results(data: pd.DataFrame, unique_cases: np.ndarray, train: np.ndarray,
                                   test: np.ndarray, model: RandomForestClassifier, 
                                   experiment_dir: str) -> RandomForestClassifier:
    print('\n ----- Testing results -----\n')

    relevant_train_cases = data.where(data['Case ID'].isin(unique_cases[train])).dropna()
    relevant_test_cases = data.where(data['Case ID'].isin(unique_cases[test])).dropna()

    x_train = relevant_train_cases.drop(['Scan ID', 'Case ID', 'Label'], axis=1)
    y_train = relevant_train_cases['Label']

    x_test = relevant_test_cases.drop(['Scan ID', 'Case ID', 'Label'], axis=1)
    y_test = relevant_test_cases['Label']

    model.fit(x_train, y_train)

    y_test_predict_proba = model.predict_proba(x_test)[:, 1]
    (test_false_as_1, test_false_as_2, test_false_as_3), n_cases_test = get_number_of_false_cases(y_test_predict_proba,
                                                                                                  relevant_test_cases,
                                                                                                  consider_as_falsies=[
                                                                                                      1, 2, 3])

    print(f'Falsies as 1st place: {test_false_as_1}/{n_cases_test} = {100 * test_false_as_1 / n_cases_test:.2f}%')
    print(f'Falsies as 2nd place: {test_false_as_2}/{n_cases_test} = {100 * test_false_as_2 / n_cases_test:.2f}%')
    print(f'Falsies as 3rd place: {test_false_as_3}/{n_cases_test} = {100 * test_false_as_3 / n_cases_test:.2f}%')

    # plotting precision recall curve
    a_p = average_precision_score(y_test, y_test_predict_proba)
    predict_proba = np.concatenate(((1 - y_test_predict_proba).reshape([y_test_predict_proba.shape[0], 1]),
                                    y_test_predict_proba.reshape([y_test_predict_proba.shape[0], 1])), axis=1)
    plot_precision_recall(y_test, predict_proba, f'Test Precision-Recall curve: (AP={a_p:0.2f})',
                          plot_micro=False, classes_to_plot=[1], figsize=(16, 12))
    plt.title(f'Test Precision-Recall curve: (AP={a_p:0.2f})', fontsize=18)
    plt.xlabel('Recall', fontsize=16)
    plt.ylabel('Precision', fontsize=16)
    plt.legend(fontsize=12)
    plt.savefig(os.path.join(experiment_dir, 'final_precision_recall_curve.jpg'))

    # plotting roc curve
    plot_roc(y_test, predict_proba, 'Test Receiver Operating Characteristic (ROC) curve',
              plot_micro=False, plot_macro=False, classes_to_plot=[1], figsize=(16, 12))
    plt.title('Test Receiver Operating Characteristic (ROC) curve', fontsize=18)
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.legend(fontsize=12)
    plt.savefig(os.path.join(experiment_dir, 'final_roc_curve.jpg'))

    return model


def plot_feature_importances(model, feature_names, save_file_name='feature_importances.jpg'):
    """Plot the feature importances of the model."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(16, 12))  # Increased figure size
    plt.title("Feature importances", fontsize=18)
    plt.xlabel("Features", fontsize=16)
    plt.ylabel("Importance", fontsize=16)
    plt.bar(range(len(importances)), importances[indices], align="center")
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90, fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlim([-1, len(importances)])
    plt.tight_layout()
    plt.savefig(save_file_name)


def print_and_plot_ood_results(data: pd.DataFrame, model: RandomForestClassifier, save_prefix: str = 'ood_'):
    """Print and plot results for out-of-distribution test set."""
    print('\n ----- Out-of-Distribution Testing Results -----\n')

    x_ood = data.drop(['Scan ID', 'Case ID', 'Label'], axis=1)
    y_ood = data['Label']

    y_ood_predict_proba = model.predict_proba(x_ood)[:, 1]
    y_ood_predict_proba = 1 - y_ood_predict_proba
    (ood_false_as_1, ood_false_as_2, ood_false_as_3), n_cases_ood = get_number_of_false_cases(y_ood_predict_proba,
                                                                                              data,
                                                                                              consider_as_falsies=[1, 2, 3])

    print(f'Falsies as 1st place: {ood_false_as_1}/{n_cases_ood} = {100 * ood_false_as_1 / n_cases_ood:.2f}%')
    print(f'Falsies as 2nd place: {ood_false_as_2}/{n_cases_ood} = {100 * ood_false_as_2 / n_cases_ood:.2f}%')
    print(f'Falsies as 3rd place: {ood_false_as_3}/{n_cases_ood} = {100 * ood_false_as_3 / n_cases_ood:.2f}%')

    # plotting precision recall curve
    a_p = average_precision_score(y_ood, y_ood_predict_proba)
    predict_proba = np.concatenate(((1 - y_ood_predict_proba).reshape([y_ood_predict_proba.shape[0], 1]),
                                    y_ood_predict_proba.reshape([y_ood_predict_proba.shape[0], 1])), axis=1)
    plot_precision_recall(y_ood, predict_proba, f'OOD Test Precision-Recall curve: (AP={a_p:0.2f})',
                          plot_micro=False, classes_to_plot=[1], figsize=(16, 12))
    plt.title(f'OOD Test Precision-Recall curve: (AP={a_p:0.2f})', fontsize=18)
    plt.xlabel('Recall', fontsize=16)
    plt.ylabel('Precision', fontsize=16)
    plt.legend(fontsize=12)
    plt.savefig(f'{save_prefix}precision_recall_curve.jpg')

    # plotting roc curve
    plot_roc(y_ood, predict_proba, 'OOD Test Receiver Operating Characteristic (ROC) curve',
             plot_micro=False, plot_macro=False, classes_to_plot=[1], figsize=(16, 12))
    plt.title('OOD Test Receiver Operating Characteristic (ROC) curve', fontsize=18)
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.legend(fontsize=12)
    plt.savefig(f'{save_prefix}roc_curve.jpg')


if __name__ == '__main__':
    experiment_name = "similar_scans_all_features_after_filtering"
    experiment_dir = os.path.join(os.getcwd(), experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)

    data_path = os.path.join(experiment_dir, 'alon_filtered_measures_results.xlsx')
    ood_data_path = os.path.join(experiment_dir, 'oo_filtered_measures_results.xlsx')
    model_saving_path = os.path.join(experiment_dir, 'protocol_chooser_model.joblib')
    
    model = RandomForestClassifier(
        n_estimators=100,
        criterion='gini',
        max_depth=10,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features='sqrt',  
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=True,
        oob_score=False,
        n_jobs=-1,
        random_state=0,
        verbose=0,
        warm_start=False,
        class_weight='balanced'
    )

    # reading the data
    data = pd.read_excel(data_path)
    columns_to_drop = [col for col in data.columns if 'Feature' in col]
    data = data.drop(columns=columns_to_drop)

    # train/test splitting of the data
    unique_cases = data['Case ID'].unique()
    train, test = KFold(n_splits=5, shuffle=True, random_state=8).split(unique_cases).__next__()

    print_and_plot_validation_results(data, unique_cases, train, model, experiment_dir)
    model = print_and_plot_testing_results(data, unique_cases, train, test, model, experiment_dir)

    # saving the trained model
    dump(model, model_saving_path)

    # Plot and print feature importances
    feature_names = data.drop(['Scan ID', 'Case ID', 'Label'], axis=1).columns
    plot_feature_importances(model, feature_names, save_file_name=os.path.join(experiment_dir, 'feature_importances.jpg'))
    importances = model.feature_importances_
    for feature, importance in zip(feature_names, importances):
        print(f'{feature}: {importance:.4f}')

    # Load and process out-of-distribution test data
    ood_data = pd.read_excel(ood_data_path)
    ood_data = ood_data.drop(columns=columns_to_drop)

    # Print and plot results for out-of-distribution test set
    print_and_plot_ood_results(ood_data, model, save_prefix=os.path.join(experiment_dir, 'ood_'))

    print(f"Results saved in: {experiment_dir}")