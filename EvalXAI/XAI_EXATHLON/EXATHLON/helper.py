import os
import argparse

# add absolute src directory to python path to import other project modules
import sys

import numpy as np

src_path = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
sys.path.append(src_path)
from data.helpers import load_mixed_formats, load_datasets_data
from metrics.evaluation import save_evaluation
from metrics.ed_evaluators import evaluation_classes as evaluation_classes_ed



# load the periods records, labels and information used to derive and evaluate anomaly predictions
# data contain:
# test 3D data, (many periods , each period many records, each record many features
# y_test 2D data  (for each period an array which indicate the anomaly type of records
# info : for each period [filename,anomaly_type] -> [[filename,anomaly_type],...,[filename,anomaly_type]]
def get_test_data():
    data1 = load_datasets_data("preprossedData/app1/", "./preprossedData/app1/", ["test"])
    data = data1
    for i in [2,3,5,6,9,10]:
        data2 = load_datasets_data(f"./preprossedData/app{i}/", f"./preprossedData/app{i}/", ["test"])
        print(data2["y_test"].shape)
        for key in data.keys():
            if "info" in key:
                oldliust=data[key]
                for new_triple in data2[key]:
                    oldliust.append(new_triple)
            else:
                data[key] = np.append(data[key], data2[key])

    return data
def get_train_data():
    data1 = load_datasets_data("preprossedData/app1/", "./preprossedData/app1/", ["train"])
    data = data1
    for i in [2,3,5,6,9,10]:
        data2 = load_datasets_data(f"./preprossedData/app{i}/", f"./preprossedData/app{i}/", ["train"])
        for key in data.keys():
            data[key] = np.append(data[key], data2[key])

    return data


#
# data: dictionary with 'test', 'y_test', 'test_info' and 'test_scores' in case of scores equal True or
#     'test_preds' in case of scores equal False
# ev_type: the level of the evaluation one of the ['ad1', 'ad2', 'ad3', 'ad4']
# experiment_name: the name under which will be stored the results in the csv file
# scores: indicates the valuation of scores or predictions.
def evaluationSingleTypeED(data,explaiener,args,ev_type='model_dependent',experiment_name="try_1",scores=False,coverage="center"):
    args.evaluation_type = ev_type
    args.ed_eval_min_anomaly_length = 1
    args.ed1_consistency_n_disturbances = 5
    args.md_eval_small_anomalies_expansion = "before"
    #args.md_eval_large_anomalies_coverage = "all"
    args.md_eval_large_anomalies_coverage = coverage
    args.f_score_beta = 1.0

    spreadsheet_path = "myADoutput/"
    OUTPUT_PATH = f"myADoutput/scoringOutput"

    #
    #'model_free': ModelFreeEvaluator,
    #'model_dependent': ModelDependentEvaluator
    #
    evaluator = evaluation_classes_ed[ev_type](args,explaiener)

    # this is equal to ad2_1.0
    evaluation_string = ev_type+"_"+str(args.f_score_beta)+f"_cov_{coverage}"

    # unique configuration identifier serving as an index in the spreadsheet
    config_name = experiment_name
    # (must be either "scoring", "detection" or "explanation")



    save_evaluation(
        "explanation", data, evaluator, evaluation_string, config_name,
        spreadsheet_path, used_data="spark", method_path=OUTPUT_PATH
    )

