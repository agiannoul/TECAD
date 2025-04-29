import os
import sys
import mlflow


sys.path.append("./src/OnlineADEngine/")

from src.OnlineADEngine.experiment.batch.auto_profile_semi_supervised_experiment import AutoProfileSemiSupervisedPdMExperiment
from src.OnlineADEngine.pipeline.pipeline import PdMPipeline
from src.OnlineADEngine.method.ocsvm import OneClassSVM
from src.OnlineADEngine.method.cnn import Cnn

from src.OnlineADEngine.method.isolation_forest import IsolationForest
from src.OnlineADEngine.method.ContextCombinator import ContextCombinator
from src.OnlineADEngine.method.dist_k_Semi import Distance_Based_Semi
from src.OnlineADEngine.preprocessing.record_level.default import DefaultPreProcessor
from src.OnlineADEngine.postprocessing.default import DefaultPostProcessor
from src.OnlineADEngine.thresholding.constant import ConstantThresholder
from src.OnlineADEngine.constraint_functions.constraint import auto_profile_max_wait_time_constraint
import loadDataset
from src.OnlineADEngine.utils.utils import calculate_mango_parameters
from src.OnlineADEngine.postprocessing.min_max_scaler import MinMaxPostProcessor
import socket
import subprocess
from src.OnlineADEngine.method.classifier_raw_and_events import Classifier_feed_raw_events
from src.OnlineADEngine.method.classifier_raw import Classifier_raw

conda_env = os.environ.get('CONDA_DEFAULT_ENV')

def is_port_in_use(host, port):
    """Check if a given port is in use on the specified host."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((host, port)) == 0

def run_mlflow_server():
    """Function to run MLflow server if it's not already running."""
    host = "127.0.0.1"
    port = 8080

    if is_port_in_use(host, port):
        print(f"MLflow server is already running at http://{host}:{port}.")
    else:
        print("Starting MLflow server...")
        subprocess.Popen(["mlflow", "ui", "--host", host, "--port", str(port)])
        print(f"MLflow server started at http://{host}:{port}.")



def run_experiment_Azure(dataset,methods, param_space_dict_per_method,method_names,experiments, experiment_names,MAX_RUNS=1, MAX_JOBS=1, INITIAL_RANDOM=1,profile_size=None):
    if profile_size==None:
        profile_size=132
        fit_size=132
    else:
        profile_size=profile_size
        fit_size=profile_size

    for current_method, current_method_param_space, current_method_name in zip(methods, param_space_dict_per_method,
                                                                               method_names):
        postprocessor = DefaultPostProcessor
        if len(sys.argv) > 1:
            if sys.argv[1] == 'minmax':
                postprocessor = MinMaxPostProcessor

        my_pipeline = PdMPipeline(
            steps={
                'preprocessor': DefaultPreProcessor,
                'method': current_method,
                'postprocessor': postprocessor,
                'thresholder': ConstantThresholder,
            },
            dataset=dataset,
            auc_resolution=30
        )

        for experiment, experiment_name in zip(experiments, experiment_names):
            current_param_space_dict = {
                'thresholder_threshold_value': [0.5],
            }

            current_param_space_dict['profile_size'] = [profile_size]

            for key, value in current_method_param_space.items():
                current_param_space_dict[f'method_{key}'] = value

            num, jobs, initial_random = calculate_mango_parameters(current_param_space_dict, MAX_JOBS, INITIAL_RANDOM,
                                                                   MAX_RUNS)

            my_experiment = experiment(
                experiment_name=experiment_name + ' ' + current_method_name,
                target_data=dataset['target_data'],
                target_sources=dataset['target_sources'],
                pipeline=my_pipeline,
                param_space=current_param_space_dict,
                num_iteration=num-initial_random,
                n_jobs=jobs,
                initial_random=initial_random,
                artifacts='./artifacts/' + experiment_name + ' artifacts',
                constraint_function=auto_profile_max_wait_time_constraint(my_pipeline),
                debug=True
            )

            best_params = my_experiment.execute()
            print(experiment_name)
            print(best_params)
    return best_params


def execute_Azure_combinator(Classifier = "xgb", detector_weight = None, selected_method = "IF", method_class=IsolationForest,
            method_parameters=None,
            method_th=0.62, MAX_RUNS=1, MAX_JOBS=1, INITIAL_RANDOM=1,
                               profile_size=None):


    if method_parameters is None:
        method_parameters = {}
    print("script: azure/run_online.py")

    tracking_uri = mlflow.get_tracking_uri()
    print("MLflow Tracking URI:", tracking_uri)

    dataset = loadDataset.get_dataset("azure")

    experiments = [
        AutoProfileSemiSupervisedPdMExperiment,
    ]

    experiment_names = [
        'AZURE',
    ]

    param_space_dict_per_method = [
        {
            'names': [[selected_method]],
            'classes': [[method_class]],
            'paramlist': [[method_parameters]],
            'all_event_names': [['error1', 'error3', 'error5', 'error4', 'error2', 'comp2', 'comp4', 'comp3', 'comp1']],
            'context_horizon': ["36 hours"],
            'add_data_to_context': [False],
            'include_detectors': [False],
            'threshold_per_method': [[method_th]],
            'include_scores_in_model': [False],
            'cross_source': [True],
            'PHhorizon': ["96 hours"],
            'classifier': [Classifier],#svc
            'detector_weight': [detector_weight],
            'retrain_anomaly_detector': [profile_size is not None],
        }

    ]
    method_names = [
        f'CC-{Classifier}-{selected_method}',
    ]
    methods=[
        ContextCombinator,
    ]
    run_experiment_Azure(dataset,methods, param_space_dict_per_method, method_names, experiments, experiment_names, MAX_RUNS, MAX_JOBS, INITIAL_RANDOM,profile_size=profile_size)


def execute_Azure_cassifier_raw(Classifier = "xgb",rs=123, MAX_RUNS=1, MAX_JOBS=1, INITIAL_RANDOM=1):
    print("script: azure/run_online.py")

    tracking_uri = mlflow.get_tracking_uri()
    print("MLflow Tracking URI:", tracking_uri)

    dataset = loadDataset.get_dataset("azure")

    experiments = [
        AutoProfileSemiSupervisedPdMExperiment,
    ]

    experiment_names = [
        'AZURE',
    ]


    param_space_dict_per_method = [
        {
            'cross_source': [True],
            'PHhorizon': ["96 hours"],
            'clasifier': [Classifier],
        },
    ]
    method_names = [
        f'Cl {Classifier} Raw'
    ]
    methods = [
        Classifier_raw,
    ]
    run_experiment_Azure(dataset,methods, param_space_dict_per_method, method_names, experiments, experiment_names, MAX_RUNS, MAX_JOBS, INITIAL_RANDOM)


def execute_Azure_cassifier_events_raw(Classifier = "xgb",rs=123, MAX_RUNS=1, MAX_JOBS=1, INITIAL_RANDOM=1):
    print("script: azure/run_online.py")

    tracking_uri = mlflow.get_tracking_uri()
    print("MLflow Tracking URI:", tracking_uri)

    dataset = loadDataset.get_dataset("azure")

    experiments = [
        AutoProfileSemiSupervisedPdMExperiment,
    ]

    experiment_names = [
        'AZURE',
    ]


    param_space_dict_per_method = [
        {
            'cross_source': [True],
            'PHhorizon': ["96 hours"],
            'all_event_names': [['error1', 'error3', 'error5', 'error4', 'error2', 'comp2', 'comp4', 'comp3', 'comp1']],
            'clasifier': [Classifier],
        },
    ]
    method_names = [
        f'Cl {Classifier} Raw-Event'
    ]
    methods = [
        Classifier_feed_raw_events,
    ]
    run_experiment_Azure(dataset,methods, param_space_dict_per_method, method_names, experiments, experiment_names, MAX_RUNS, MAX_JOBS, INITIAL_RANDOM)


def execute_Azure_method(selected_method = "IF", MAX_RUNS=1, MAX_JOBS=1, INITIAL_RANDOM=1):
    print("script: azure/run_online.py")

    tracking_uri = mlflow.get_tracking_uri()
    print("MLflow Tracking URI:", tracking_uri)

    dataset = loadDataset.get_dataset("azure")

    experiments = [
        AutoProfileSemiSupervisedPdMExperiment,
    ]

    experiment_names = [
        'AZURE',
    ]


    profile_size=None
    allmethodnames = ['KNN', 'IF', 'OCSVM', 'CNN']
    allmethodclasses = [Distance_Based_Semi, IsolationForest, OneClassSVM, Cnn]
    allmethod_param_list = [
        {'k': [5,10], 'window_norm': [False]},
        {'n_estimators': [200], 'max_samples': [112], 'random_state': [42], 'max_features': [0.8], 'bootstrap': [True]},
        {'kernel': ['sigmoid'], 'nu': [0.5], 'gamma': ['scale']},
        {'window_size': [10]},

    ]

    pos = allmethodnames.index(selected_method)
    method_class = allmethodclasses[pos]
    method_parameters = allmethod_param_list[pos]

    param_space_dict_per_method={}
    for parmkey in method_parameters:
        # param_space_dict_per_method[parmkey] = [method_parameters[parmkey]]
        param_space_dict_per_method[parmkey] = method_parameters[parmkey]
    param_space_dict_per_method=[param_space_dict_per_method]
    method_names = [
        selected_method
    ]
    methods = [
        method_class,
    ]
    return run_experiment_Azure(dataset,methods, param_space_dict_per_method, method_names, experiments, experiment_names, MAX_RUNS, MAX_JOBS, INITIAL_RANDOM,profile_size=profile_size)




def run_experiment_combinator(Classifier="xgb",selected_method = "IF",detector_weight=None):
    allmethodnames = ['KNN', 'IF',"OCSVM","CNN"]
    allmethodclasses = [Distance_Based_Semi, IsolationForest, OneClassSVM,Cnn]
    allmethod_param_list = [
        {'k': 5,'window_norm': False},
        {'n_estimators': 200,'max_samples': 112,'random_state': 42,'max_features': 0.8,'bootstrap': True },
        {'kernel': 'sigmoid', 'nu': 0.5, 'gamma': 'scale', },
        {'window_size': 10},

    ]


    profiles=[None,None,None,None]



    pos=allmethodnames.index(selected_method)
    method_class=allmethodclasses[pos]
    method_parameters=allmethod_param_list[pos]
    method_th = execute_Azure_method(selected_method=selected_method)["th"]
    print(method_th)
    profile_size=None


    execute_Azure_combinator(Classifier = Classifier, detector_weight = detector_weight,selected_method =selected_method,method_class=method_class,method_parameters=method_parameters,method_th=method_th,profile_size=profile_size)


import random
random.seed(0)
import torch
torch.manual_seed(0)
import numpy as np
np.random.seed(0)

# execute_Azure_cassifier_raw(Classifier="svc")
# execute_Azure_cassifier_raw(Classifier="xgb")
# execute_Azure_cassifier_events_raw(Classifier="svc")
# execute_Azure_cassifier_events_raw(Classifier="xgb")


# run_experiment_combinator(Classifier="svc",selected_method = "CNN",detector_weight=0)
# run_experiment_combinator(Classifier="xgb",selected_method = "CNN",detector_weight=0)
# run_experiment_combinator(Classifier="svc",selected_method = "CNN",detector_weight=0.5)
# run_experiment_combinator(Classifier="xgb",selected_method = "CNN",detector_weight=0.5)
# run_experiment_combinator(Classifier="svc",selected_method = "OCSVM",detector_weight=None)

# execute_Azure_method(selected_method="IF")

# run_experiment_combinator(Classifier="svc",selected_method = "KNN",detector_weight=0)
# run_experiment_combinator(Classifier="xgb",selected_method = "KNN",detector_weight=0)
#
# run_experiment_combinator(Classifier="svc",selected_method = "IF",detector_weight=0)
# run_experiment_combinator(Classifier="xgb",selected_method = "IF",detector_weight=0)
#
# run_experiment_combinator(Classifier="svc",selected_method = "OCSVM",detector_weight=0)
# run_experiment_combinator(Classifier="xgb",selected_method = "OCSVM",detector_weight=0)
#
# run_experiment_combinator(Classifier="svc",selected_method = "KNN",detector_weight=0.5)
# run_experiment_combinator(Classifier="xgb",selected_method = "KNN",detector_weight=0.5)
#
# run_experiment_combinator(Classifier="svc",selected_method = "IF",detector_weight=0.5)
# run_experiment_combinator(Classifier="xgb",selected_method = "IF",detector_weight=0.5)
#
# run_experiment_combinator(Classifier="svc",selected_method = "OCSVM",detector_weight=0.5)
# run_experiment_combinator(Classifier="xgb",selected_method = "OCSVM",detector_weight=0.5)

# execute_Azure_method(selected_method="IF")
# execute_Azure_method(selected_method="KNN")
# execute_Azure_method(selected_method="HBOS")
# execute_Azure_method(selected_method="PCA")
# execute_Azure_method(selected_method="USAD")
import argparse
def main():
    parser = argparse.ArgumentParser(description="Run experiments")

    parser.add_argument("experiment", choices=["classifier", "classifier+events","combinator", "OAD"], help="Which experiment to run")
    parser.add_argument("--Classifier", type=str, help="Classifier type (e.g., svc, xgb)",choices=["xgb","svc"])
    parser.add_argument("--selected_method", type=str, default="KNN",help="Selected method (e.g., IF, KNN)",choices=["IF","KNN","OCSVM","CNN"])
    parser.add_argument("--detector_weight", type=float, default=None, help="Detector weight (only for combinator, default None for learnable )")

    args = parser.parse_args()

    if args.experiment == "classifier":
        if not args.Classifier:
            print("Error: Classifier is required for classifier experiment")
            return
        execute_Azure_cassifier_raw(Classifier=args.Classifier)
    elif args.experiment == "classifier+events":
        if not args.Classifier:
            print("Error: Classifier is required for classifier experiment")
            return
        execute_Azure_cassifier_events_raw(Classifier=args.Classifier)
    elif args.experiment == "combinator":
        if not args.Classifier or not args.selected_method:
            print("Error: Classifier and selected_method are required for combinator experiment")
            return
        run_experiment_combinator(Classifier=args.Classifier, selected_method=args.selected_method,
                                  detector_weight=args.detector_weight)

    elif args.experiment == "OAD":
        if not args.selected_method:
            print("Error: selected_method is required for method experiment")
            return
        execute_Azure_method(selected_method=args.selected_method)


if __name__ == "__main__":
    main()