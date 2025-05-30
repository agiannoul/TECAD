import os
import json
import importlib
from pathlib import Path

import numpy as np
import pandas as pd

# add absolute src directory to python path to import other project modules
import sys

from matplotlib import pyplot as plt

src_path = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
sys.path.append(src_path)
from data.helpers import get_matching_sampling, NumpyJSONEncoder



# spark streaming application ids
APP_IDS = list(range(1, 11))
# spark streaming trace types
TRACE_TYPES = [
    'undisturbed', 'bursty_input', 'bursty_input_crash',
    'stalled_input', 'cpu_contention', 'process_failure'
]
# anomaly types (the integer label for a type is its corresponding index in the list +1)
ANOMALY_TYPES = [
    'bursty_input', 'bursty_input_crash', 'stalled_input',
    'cpu_contention', 'driver_failure', 'executor_failure', 'unknown'
]
# path dictionary interface containing the relevant root paths to load data from
DATA_ROOT = Path(f"./preprossedData")
DATA_PATHS_DICT = dict(
    {'labels': DATA_ROOT},
    **{f'app{i}': os.path.join(DATA_ROOT, f'app{i}') for i in APP_IDS}
)
# getter function returning the application id for a given trace file name
def get_app_from_name(file_name): return int(file_name.split('_')[0])

dark2_palette = plt.get_cmap('Dark2').colors
ANOMALY_COLORS = {type_: dark2_palette[i] for i, type_ in enumerate(ANOMALY_TYPES)}

# colors used for reporting performance globally and per anomaly type
METRICS_COLORS = dict({'global': 'blue'}, **ANOMALY_COLORS)








def get_metric_names_dict(evaluation_step, anomaly_types=None, *, beta=None):
    """Returns a dictionary mapping the computed metrics to their name for `evaluation_step`.

    Args:
        evaluation_step (str): evaluation step (must be either "scoring", "detection" or "explanation").
        anomaly_types (list|None): names of the anomaly types that we might encounter in the data (if relevant).
        beta (float|None): beta parameter for the F-score, must be provided and between 0 and 1
            if `evaluation_step` is "detection".

    Returns:
        dict: the dictionary mapping the computed metrics to their name.
    """
    a_t = 'the evaluation step must be either `scoring`, `detection` or `explanation`'
    assert evaluation_step in ['scoring', 'detection', 'explanation'], a_t
    # scoring evaluation
    if evaluation_step == 'scoring':
        return {'auprc': 'AUPRC'}
    # detection evaluation
    if evaluation_step == 'detection':
        assert beta is not None and 0 <= beta <= 1, 'beta must be set and between 0 and 1'
        return {
            'precision': 'PRECISION',
            'recall': 'RECALL',
            'f-score': f'F{beta}_SCORE'
        }
    # explanation evaluation
    shared_m_names = ['prop_covered', 'prop_explained', 'time']
    ed_m_names = ['conciseness', 'norm_consistency', 'precision', 'recall', 'f1_score']
    ed_levels = [1]
    # ED2 metrics make no sense in case of unknown anomaly types
    if anomaly_types is not None:
        ed_levels.append(2)
    return dict(
        {k: k.upper() for k in shared_m_names},
        **{f'ed{i}_{m}': f'ed{i}_{m}'.upper() for i in ed_levels for m in ed_m_names}
    )



def save_evaluation(evaluation_step, data_dict, evaluator, evaluation_string, config_name, spreadsheet_path,
                    *, used_data=None, method_path=None, ignore_anomaly_types=False):
    """Adds the evaluation of the provided configuration to a sorted comparison spreadsheet.

    If the used data is specified and provides multiple anomaly types, the evaluation will be
    both performed "globally" (i.e. considering all types together) and type-wise.

    Note: the term "global" is both used when making no difference between applications or traces
    for spark data ("global granularity") and when making no difference between anomaly types ("global type").

    Note: conceptually, `anomaly_types` being None means that we do not have any information
    regarding the types of anomalies in the labels, which is different from knowing there is
    a single type of anomaly. In the latter case, the single anomaly type should be named and
    provided in `utils.{used_data}.ANOMALY_TYPES`.
    If `used_data` does not provide anomaly types, or `ignore_anomaly_types` is True, only the
    global performance will be reported, whether the provided labels are multiclass or not.

    Args:
        evaluation_step (str): evaluation step (must be either "scoring", "detection" or "explanation").
        data_dict (dict): datasets periods record-wise (data/outlier scores/binary predictions), labels and info:
            - full data as `{set_name: ndarray}`. Only relevant for ED evaluation.
            - scores/predictions as `{(set_name)_scores|preds: ndarray}`. Only relevant for AD evaluation.
            - labels as `{y_(set_name): ndarray}`. Either ground-truth or predictions for ED evaluation.
            - info as `{(set_name)_info: ndarray}`. With each period info of the form
                `(file_name, anomaly_type, period_rank)`.
            The first three array types are of shape `(n_periods, period_length)`;
                With `period_length` depending on the period.
        evaluator (ADEvaluator|EDEvaluator): object defining the metrics of interest.
        evaluation_string (str): formatted evaluation string to compare models under the same requirements.
        config_name (str): unique configuration identifier serving as an index in the spreadsheet.
        spreadsheet_path (str): comparison spreadsheet path.
        used_data (str|None): used data (if relevant, used to derive anomaly types and granularity levels).
        method_path (str): scoring/detection/explanation method path, to save any extended evaluation to.
        ignore_anomaly_types (bool): if True, ignore any information regarding the presence of
            distinct anomaly types in the labels.
    """
    # set anomaly types and colors depending on the used data if relevant
    anomaly_types, metrics_colors = None, None
    if used_data is not None and not ignore_anomaly_types:
        if used_data == "spark":
            anomaly_types = ANOMALY_TYPES
            metrics_colors = METRICS_COLORS
    # set elements depending on the evaluation step
    a_t = 'the evaluation step must be either `scoring`, `detection` or `explanation`'
    assert evaluation_step in ['scoring', 'detection', 'explanation'], a_t
    try:
        beta = evaluator.beta
    except AttributeError:
        beta = None
    metric_names_dict = get_metric_names_dict(evaluation_step, anomaly_types, beta=beta)
    if evaluation_step == 'scoring':
        get_metrics_row, pred_suffix = get_scoring_metrics_row, '_scores'
    elif evaluation_step == 'detection':
        get_metrics_row, pred_suffix = get_detection_metrics_row, '_preds'
    else:
        # "predictions" are just the periods records here
        get_metrics_row, pred_suffix = get_explanation_metrics_row, ''

    # set the full path for the comparison spreadsheet
    full_spreadsheet_path = os.path.join(
        spreadsheet_path, f'{evaluation_string}_{evaluation_step}_comparison.csv'
    )

    # extract evaluated dataset names from the label keys
    label_prefix = 'y_'
    set_names = [n.replace(label_prefix, '') for n in data_dict if label_prefix in n]

    # evaluation DataFrame for each considered dataset
    set_evaluation_dfs = []
    for n in set_names:
        # setup column space and hierarchical index for the current dataset
        periods_labels, periods_preds = data_dict[f'y_{n}'], data_dict[f'{n}{pred_suffix}']
        column_names = get_column_names(evaluation_step, periods_labels, anomaly_types, metric_names_dict, n)
        set_evaluation_dfs.append(
            pd.DataFrame(
                columns=column_names, index=pd.MultiIndex.from_tuples([], names=['method', 'granularity'])
            )
        )
        evaluation_df = set_evaluation_dfs[-1]

        # align sampling periods of the current dataset elements
        if evaluation_step in ['scoring', 'detection']:
            # upsample model predictions to match the sampling period of labels
            periods_preds = get_matching_sampling(periods_preds, periods_labels)
        else:
            # downsample labels or model predictions to match the sampling period of data records
            periods_labels = get_matching_sampling(periods_labels, periods_preds, agg_func=np.max)

        # add metrics when considering all traces and applications the same
        print('computing metrics "globally".')
        periods_info = data_dict[f'{n}_info']
        evaluation_df.loc[(config_name, 'global'), :] = get_metrics_row(
            periods_labels, periods_preds, evaluator, column_names, metric_names_dict,
            anomaly_types, granularity='global', method_path=method_path,
            evaluation_string=evaluation_string, metrics_colors=metrics_colors,
            periods_info=periods_info, used_data=used_data
        )

        # if using spark data, add metrics for each application and trace
        if used_data == 'spark':

            periods_info_cor=[]
            for kati in periods_info:
                periods_info_cor.append((kati[0],kati[1],kati[1]))
            periods_info=periods_info_cor

            app_ids = set([get_app_from_name(info[0]) for info in periods_info])
            for app_id in app_ids:
                app_indices = [i for i, info in enumerate(periods_info) if get_app_from_name(info[0]) == app_id]
                app_labels, app_preds, app_info = [], [], []
                for i in range(len(periods_info)):
                    if i in app_indices:
                        app_labels.append(periods_labels[i])
                        app_preds.append(periods_preds[i])
                        app_info.append(periods_info[i])
                print(f'computing metrics for application {app_id}.')
                evaluation_df.loc[(config_name, f'app{app_id}'), :] = get_metrics_row(
                    app_labels, app_preds, evaluator, column_names, metric_names_dict,
                    anomaly_types, granularity='app', metrics_colors=metrics_colors,
                    method_path=method_path, periods_key=f'app{app_id}',
                    periods_info=app_info, used_data=used_data
                )
                # trace-wise performance
                trace_names = set([info[0] for info in app_info])
                for trace_name in trace_names:
                    trace_indices = [i for i, info in enumerate(app_info) if info[0] == trace_name]
                    trace_labels, trace_preds, trace_info = [], [], []
                    for j in range(len(app_info)):
                        if j in trace_indices:
                            trace_labels.append(app_labels[j])
                            trace_preds.append(app_preds[j])
                            trace_info.append(app_info[j])
                    print(f'computing metrics for trace {trace_name}.')
                    evaluation_df.loc[(config_name, trace_info[0][0]), :] = get_metrics_row(
                        trace_labels, trace_preds, evaluator, column_names, metric_names_dict,
                        anomaly_types, granularity='trace', method_path=method_path,
                        periods_key=trace_name, periods_info=trace_info, used_data=used_data
                    )
                # average performance across traces (in-trace separation ability)
                trace_rows = evaluation_df.loc[~(
                    (evaluation_df.index.get_level_values('granularity').str.contains('global')) |
                    (evaluation_df.index.get_level_values('granularity').str.contains('app'))
                )]
                evaluation_df.loc[(config_name, 'trace_avg'), :] = trace_rows.mean(axis=0)
            # average performance across applications (in-application separation ability)
            app_rows = evaluation_df.loc[
                evaluation_df.index.get_level_values('granularity').str.contains('app')
            ]
            evaluation_df.loc[(config_name, 'app_avg'), :] = app_rows.mean(axis=0)

    # add the new evaluation to the comparison spreadsheet, or create it if it does not exist
    evaluation_df = pd.concat(set_evaluation_dfs, axis=1)
    try:
        comparison_df = pd.read_csv(full_spreadsheet_path, index_col=[0, 1]).astype(float)
        print(f'adding evaluation of `{config_name}` to {full_spreadsheet_path}...', end=' ', flush=True)
        for index_key in evaluation_df.index:
            comparison_df.loc[index_key, :] = evaluation_df.loc[index_key, :].values
        comparison_df.to_csv(full_spreadsheet_path)
        print('done.')
    except FileNotFoundError:
        print(f'creating {full_spreadsheet_path} with evaluation of `{config_name}`...', end=' ', flush=True)
        evaluation_df.to_csv(full_spreadsheet_path)
        print('done.')

def get_explanation_metrics_row(labels, periods, evaluator, column_names, metric_names_dict,
                                anomaly_types, granularity, method_path=None, evaluation_string=None,
                                metrics_colors=None, periods_key=None, periods_info=None, used_data=None):
    """Returns the metrics row to add to an explanation evaluation DataFrame.

    Args:
        labels (ndarray): periods labels of shape `(n_periods, period_length)`.
            With `period_length` depending on the period. Could be either ground-truth or model predictions.
        periods (ndarray): periods records of shape `(n_periods, period_length, n_features)`.
        evaluator (EDEvaluator): object defining the ED metrics of interest.
        column_names (list): list of column names corresponding to the metrics to compute.
        metric_names_dict (dict): dictionary mapping metrics to compute to their name.
        anomaly_types (list|None): names of the anomaly types that we might encounter in the data (if relevant).
        granularity (str): evaluation granularity.
            Must be either `global`, for overall, `app`, for app-wise or `trace`, for trace-wise.
        method_path (str|None): ED method path, to save any extended evaluation to.
        evaluation_string (str|None): formatted evaluation string to compare models under the same requirements.
        metrics_colors (dict|str|None): color to use for the curves if single value, color to use
            for each anomaly type if dict (the keys must then belong to `anomaly_types`).
        periods_key (str|None): if granularity is not `global`, name to use to identify the periods.
            Has to be of the form `appX` if `app` granularity or `trace_name` if `trace` granularity.
        periods_info (list|None): optional list of periods information.
        used_data (str|None): used data, if relevant.

    Returns:
        list: list of metrics to add to the evaluation DataFrame (corresponding to `column_names`).
    """
    assert granularity in ['global', 'app', 'trace']

    # set metrics keys to make sure the output matches to order of `column_names`
    metrics_row = pd.DataFrame(columns=column_names)
    metrics_row._append(pd.Series(), ignore_index=True)
    #metrics_row = pd.concat([metrics_row, pd.DataFrame([pd.Series()])], ignore_index=True)

    # recover the dataset prefix from the column names
    set_prefix = f'{column_names[0].split("_")[0]}_'

    # compute the metrics defined by the evaluator and get the instances explanations
    metrics_dict, explanations_dict = evaluator.compute_metrics(
        periods, labels, periods_info, used_data, include_ed2=(anomaly_types is not None)
    )
    if granularity == 'global':
        # save the instances explanations as a JSON file
        print("pass")
        # print(f'saving instances explanations to {method_path}...', end=' ', flush=True)
        # os.makedirs(method_path, exist_ok=True)
        # with open(os.path.join(method_path, 'explanations.json'), 'w') as json_file:
        #     json.dump(
        #         explanations_dict, json_file, separators=(',', ':'), sort_keys=True, indent=4,
        #         cls=NumpyJSONEncoder
        #     )
        # print('done.')

    # we do not use average metrics across types here (ED1 conciseness is only returned globally)
    only_global_metric = 'ed1_conciseness'
    for k in metrics_dict:
        if k != only_global_metric:
            metrics_dict[k].pop('avg')

    # case of unknown anomaly types (only use "global" values of type-wise metrics)
    if anomaly_types is None:
        for m_name, m_value in metrics_dict.items():
            if m_name != only_global_metric:
                m_value = m_value['global']
            metrics_row.at[0, f'{set_prefix}{metric_names_dict[m_name]}'] = m_value
    # case of multiple known anomaly types
    else:
        # `global` and interpretable anomaly types that belong to the keys of inference time
        class_names = ['global'] + [
            anomaly_types[i-1] for i in range(1, len(anomaly_types) + 1) if i in metrics_dict['time']
        ]
        # add the metric and column corresponding to each type
        for cn in class_names:
            if cn == 'global':
                for m_name, m_value in metrics_dict.items():
                    # the "global" consideration does not make sense for ED2 metrics
                    if 'ed2' not in m_name:
                        if m_name != only_global_metric:
                            m_value = m_value['global']
                        metrics_row.at[0, f'{set_prefix}GLOBAL_{metric_names_dict[m_name]}'] = m_value
            else:
                label_key = anomaly_types.index(cn) + 1
                type_metrics_dict = {k: v for k, v in metrics_dict.items() if k != only_global_metric}
                for m_name, m_value in type_metrics_dict.items():
                    metrics_row.at[0, f'{set_prefix}{cn.upper()}_{metric_names_dict[m_name]}'] = \
                        m_value[label_key]
    return metrics_row.iloc[0].tolist()


def get_column_names(evaluation_step, periods_labels, anomaly_types, metric_names_dict, set_name):
    """Returns the column names to use for an evaluation DataFrame.

    Args:
        evaluation_step (str): evaluation step (must be either "scoring", "detection" or "explanation").
        periods_labels (ndarray): periods labels of shape `(n_periods, period_length,)`.
            Where `period_length` depends on the period.
        anomaly_types (list|None): names of the anomaly types that we might encounter in the data (if relevant).
        metric_names_dict (dict): dictionary mapping metrics to compute to their name.
        set_name (str): the set name will be prepended to the column names.

    Returns:
        list: the column names to use for the evaluation.
    """
    a_t = 'the evaluation step must be either `scoring`, `detection` or `explanation`'
    assert evaluation_step in ['scoring', 'detection', 'explanation'], a_t
    set_prefix = f'{set_name.upper()}_'
    # case of unknown anomaly types
    if anomaly_types is None:
        return [f'{set_prefix}{m}' for m in metric_names_dict.values()]
    column_names = []
    # positive classes encountered in the periods of the dataset
    pos_classes = np.delete(np.unique(np.concatenate(periods_labels, axis=0)), 0)
    # anomaly "classes" (`global` + one per type)
    class_names = ['global'] + [anomaly_types[i-1] for i in range(1, len(anomaly_types) + 1) if i in pos_classes]
    # metrics that are not considered globally, metrics that are only considered globally
    no_global, only_global = [], []
    if evaluation_step == 'detection':
        only_global += ['precision']
    elif evaluation_step == 'explanation':
        no_global += [k for k in metric_names_dict if 'ed2' in k]
        only_global += ['ed1_conciseness']
    for cn in class_names:
        removed_metrics = no_global if cn == 'global' else only_global
        class_metric_names = [v for k, v in metric_names_dict.items() if k not in removed_metrics]
        column_names += [f'{set_prefix}{cn.upper()}_{m}' for m in class_metric_names]
    return column_names