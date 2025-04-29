import os
import importlib

import numpy as np
import argparse
# add absolute src directory to python path to import other project modules
import sys
src_path = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
sys.path.append(src_path)
from utils.common import (
    PIPELINE_TRAIN_NAME, PIPELINE_TEST_NAME, MODELING_TEST_NAME, ANOMALY_TYPES, CHOICES,
    parsers, get_output_path, get_modeling_task_and_classes, get_args_string, get_best_thresholding_args
)
from data.helpers import load_datasets_data, load_mixed_formats
from modeling.data_splitters import get_splitter_classes
from modeling.forecasting.helpers import get_trimmed_periods
from detection.detector import Detector
from explanation.explainers import get_explanation_classes
from metrics.evaluation import save_evaluation
from metrics.ed_evaluators import evaluation_classes

def getscores(data):
    randomscores=[0,0.1,0.2,0.1,0.2,0.3,0.4,0.3,0.1,0,0.1]
    size_rand_scores=len(randomscores)
    numpy_3d_data=data['test']
    scores=[]
    for period in numpy_3d_data:
        period_scores=[]
        counter = 0
        for record_in_period in period:
            period_scores.append(randomscores[counter%size_rand_scores])
            counter+=1
        scores.append(np.array(period_scores))
    return np.array(scores)

def getpreds(data,th):
    numpy_3d_data = data['test_scores']
    scores = []
    for period in numpy_3d_data:
        period_scores=[]
        for record_in_period in period:
            if record_in_period>=th:
                period_scores.append(1)
            else:
                period_scores.append(0)
        scores.append(np.array(period_scores))
    return np.array(scores)

# args
# explanation discovery arguments
temp_my_arg={'explanation_method': 'exstream',
    'explained_predictions': 'model',
    # ED evaluation parameters
    'ed_eval_min_anomaly_length': 1,
    'ed1_consistency_n_disturbances': 5,
    # model-free evaluation
    'mf_eval_min_normal_length': 1,
    'mf_ed1_consistency_sampled_prop': 0.8,
    'mf_ed1_accuracy_n_splits': 5,
    'mf_ed1_accuracy_test_prop': 0.2,
    # model-dependent evaluation
    'md_eval_small_anomalies_expansion': 'before',
    'md_eval_large_anomalies_coverage': 'all',
    # EXstream
    'exstream_fp_scaled_std_threshold': 1.64,
    # MacroBase
    'macrobase_n_bins': 10,
    'macrobase_min_support': 0.4,
    'macrobase_min_risk_ratio': 1.5,
    # LIME
    'lime_n_features': 5}
### CHOICES
# 'train_explainer': {
#         # explanation discovery methods (relying on a model or not)
#         'model_free_explanation': ['exstream', 'macrobase'],
#         'model_dependent_explanation': ['lime'],
#         # whether to "explain" ground-truth labels or predictions from an AD method
#         'explained_predictions': ['ground.truth', 'model'],
#         # model-dependent evaluation (expansion and coverage policies of small and large anomalies, respectively)
#         'md_eval_small_anomalies_expansion': ['none', 'before', 'after', 'both'],
#         'md_eval_large_anomalies_coverage': ['all', 'center', 'end']
#     }

def getparserFromDictionary(temp_my_arg):
    parser = argparse.ArgumentParser(
        prog='ProgramName',
        description='What the program does',
        epilog='Text at the bottom of help')
    for key in temp_my_arg.keys():
        parser.add_argument(
            f'--{key}', default=temp_my_arg[key],
        )
    return parser.parse_args()



if __name__ == '__main__':
    # parse and get command line arguments
    argsold = parsers['train_explainer'].parse_args()
    args = getparserFromDictionary(temp_my_arg)

    #set input and output paths
    DATA_INFO_PATH = get_output_path(argsold, 'make_datasets')
    DATA_INPUT_PATH = get_output_path(argsold, 'build_features', 'data')

    #load the periods records, labels and information used to train and evaluate explanation discovery
    explanation_sets = [PIPELINE_TEST_NAME]

    #all testing data {test: 3D, y_test{2D} and info}
    explanation_data = load_datasets_data(DATA_INPUT_PATH, DATA_INFO_PATH, explanation_sets)
    #print(explanation_data)

    explanation_data['test_scores'] = getscores(explanation_data)
    th=0.2
    explanation_data['test_preds'] = getpreds(explanation_data,th)


    explainer_args,training_samples=dict(),None





    ############ CHOOSE EXPLAINER:
    # get ED method type
    if args.explanation_method in CHOICES['train_explainer']['model_dependent_explanation']:
        print("NOT SUPPORTED YET")
        exit(-1)
        method_type = 'model_dependent'
    elif args.explanation_method in CHOICES['train_explainer']['model_free_explanation']:
        method_type = 'model_free'
    else:
        raise ValueError('the provided ED method must be either "model-free" or "model-dependent"')

    # set output and comparison paths at the end so that they use potentially updated args

    #spreadsheet_path = "/media/agiannous/dd7e73ed-3cbe-48a4-8fa9-513925c229a6/Desktop2/exathlon-master/src/detection/myADoutput"
    #OUTPUT_PATH = "/media/agiannous/dd7e73ed-3cbe-48a4-8fa9-513925c229a6/Desktop2/exathlon-master/src/detection/myADoutput/scoringOutput"



    spreadsheet_path = "/media/agiannous/dd7e73ed-3cbe-48a4-8fa9-513925c229a6/Desktop2/exathlon-master/src/detection/myexplainer"
    OUTPUT_PATH = "/media/agiannous/dd7e73ed-3cbe-48a4-8fa9-513925c229a6/Desktop2/exathlon-master/src/detection/myexplainer/scoringOutput"


    #initialize the relevant explainer based on command-line and method-specific arguments
    explainer = get_explanation_classes()[args.explanation_method](args, OUTPUT_PATH, **explainer_args)

    # fit explainer to training samples if they were set
    if training_samples is not None:
        explainer.fit(training_samples)

    # save explanation discovery performance
    config_name = get_args_string(args, 'explanation')
    evaluation_string = get_args_string(args, 'ed_evaluation')
    evaluator = evaluation_classes[method_type](args, explainer)
    print(evaluator)
    save_evaluation(
        'explanation', explanation_data, evaluator, evaluation_string, config_name, spreadsheet_path, method_path=OUTPUT_PATH,
        ignore_anomaly_types=(args.explained_predictions == 'model')
    )
