import logging
import os
import math
import random
import abc
import sys
from pathlib import Path
from collections import defaultdict
import random
import re
import subprocess
from typing import Callable
from pathlib import Path
import time
import pickle

import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import locket
import uuid

from pipeline.pipeline import PdMPipeline
from evaluation.evaluation import AUCPR_new as pdm_evaluate
from evaluation.evaluation import AUCPR_ranges_new as pdm_evaluate_ranges


logging.basicConfig(level = logging.INFO)


def process_data(current_data, header, data_type) -> list[pd.DataFrame]:
    if len(current_data)==0:
        return current_data
    if isinstance(current_data, pd.DataFrame):
        result = [current_data]
    elif isinstance(current_data, str):
        # if it is a string check if it is a csv file or directory containing csv files
        if current_data.endswith('.csv'):
            result = [pd.read_csv(current_data, header=header)]
        elif Path(current_data).is_dir(): 
            result = []

            current_directory_files = os.listdir(current_data)
            current_csv_files = [file for file in current_directory_files if file.endswith('.csv')]
            for csv_file in current_csv_files:
                current_csv_file_path = os.path.join(current_data, csv_file)
                result.append(pd.read_csv(current_csv_file_path, header=header))      
    elif isinstance(current_data, list):
        result=current_data
        # necessarily nested in order to avoid exception when looping on a variable that is not a list because python does not support short-circuit evaluation
        if not all(isinstance(item, pd.DataFrame) for item in current_data):
            raise Exception(f'Some element of the list parameter \'{data_type}\' has unsupported type')
    else:
        raise Exception(f'Not supported type {type(current_data)} for parameter \'{data_type}\'')
    

    return result


class PdMExperiment(abc.ABC):
    def __init__(self, 
                 experiment_name: str, 
                 pipeline: PdMPipeline,
                 param_space: dict,
                 constraint_function: Callable = None,
                 target_data: list[pd.DataFrame] = None , # TODO str for directory with csv files for each scenario or single csv file of one scenario
                 target_sources: list[str] = None,
                 historic_data: list[pd.DataFrame] = [], # TODO str for directory with csv files for each scenario or single csv file of one scenario
                 historic_sources: list[str] = [],
                 optimization_param: str = 'AD1_AUC',
                 initial_random: int = 2,
                 num_iteration: int = 20,
                 batch_size: int = 1,
                 n_jobs: int = 1,
                 random_state: int = 42,
                 random_n_tries: int = 3,
                 constraint_max_retries: int = 10,
                 historic_data_header: str  = 'infer',
                 target_data_header: str = 'infer',
                 artifacts: str = 'artifacts',
                 debug: bool = False,
                 delay: float = None, # in milliseconds
                 log_best_scores: bool = False,
    ):
        self.experiment_name = experiment_name
        # TODO target and historic data and sources parameter should be removed, became default parameters for backwards compatibility
        self.historic_data = pipeline.dataset['historic_data']
        self.historic_sources = pipeline.dataset['historic_sources']
        self.target_data = pipeline.dataset['target_data']
        self.target_sources = pipeline.dataset['target_sources']
        self.pipeline = pipeline
        self.param_space = param_space
        self.optimization_param = optimization_param
        self.initial_random = initial_random
        self.num_iteration = num_iteration
        # self.batch_size = batch_size currently commented out because of using only scheduler.parallel, more info on issue #97 on Mango - alternatives include using only scheduler.parallel or letting the user decide depending on his hardware
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.historic_data_header = historic_data_header
        self.target_data_header = target_data_header
        self.artifacts = artifacts

        self.debug = debug
        self.delay = delay

        self.log_best_scores = log_best_scores
        current_uuid = uuid.uuid4()
        self.lock_file_path = f'pdm_evaluation_framework_lock_file_{current_uuid}.lock'
        self.best_scores_info_dict_path = f'best_scores_info_{current_uuid}.pkl'

        self.event_data = self.pipeline.event_data
        self.constraint_function = constraint_function
        
        # TODO the next line is probably useless
        Path(self.artifacts).mkdir(parents=True, exist_ok=True)
        self.extra_metrics = {}
        # process historic data
        self.historic_data = process_data(self.historic_data, historic_data_header, 'historic_data')
        
        # process target data
        self.target_data = process_data(self.target_data, target_data_header, 'target_data')
        
        self.experiment_id = None

        random.seed(self.random_state)

        import pkg_resources
        required = {'torch'}
        installed = {pkg.key for pkg in pkg_resources.working_set}
        missing = required - installed
        if not missing:
            import torch
            torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        current_dir = os.getcwd()
        os.chdir("./src/OnlineADEngine/evaluation/RBPR_official")

        # Run make clean

        if os.name == 'nt':
            powershell_command = "(Get-Content Makefile) -replace 'rm', 'del' | Out-File -encoding ASCII Makefile"
            subprocess.run(["powershell", "-Command", powershell_command])
            subprocess.run(["powershell", "-Command", "make", "clean"])
            subprocess.run(["powershell", "-Command", "make"])
        else:
            subprocess.call(["make","-f","MakefileL", "clean"])
            subprocess.call(["make","-f","MakefileL"])
            # Move the evaluate executable to the parent directory
            subprocess.call(["mv", "evaluate", ".."])
        # Run make

        # Change back to the original directory
        os.chdir(current_dir)

    @abc.abstractmethod
    def execute(self) -> dict:
        pass

    
    def _register_experiment(self) -> None:
        # if self.delay is not None:
        #     print(f'Cooldown for {self.delay} milliseconds')
        #     time.sleep(self.delay / 1000)
            
        try:
            self.experiment_id = mlflow.create_experiment(name=self.experiment_name)
        except Exception as e:
            logging.warning(f'Experiment with experiment name \'{self.experiment_name}\' already exists. Be careful if you are sure about including your run in this experiment.')
            self.experiment_id = mlflow.get_experiment_by_name(self.experiment_name).experiment_id


    def _inner_plot(self, color, rangearay, datesofscores, minvalue, maxvalue, label):
        plt.fill_between(datesofscores, minvalue, maxvalue, where=rangearay, color=color,
                        alpha=0.3, label=label)


    def _plot_scores(self, plot_dictionary, best_metrics_dict) -> None:
        tups=[]
        for rec,prc in  zip(plot_dictionary['recall'], plot_dictionary['prc']):
            tups.append((rec,prc))
        tups= sorted(tups, key=lambda x: (x[0], -x[1]))
        xaxisvalue=[]
        yaxisvalue=[]
        for tup in tups:
            xaxisvalue.append(tup[0])
            yaxisvalue.append(tup[1])
        plt.plot(xaxisvalue,yaxisvalue,"-o")
        plt.tight_layout()
        mlflow.log_figure(plt.gcf(), 'pr_curve.png')

        plt.clf()
        plt.figure(figsize=(20, 20))
        if self.debug:
            counter=0
            namescount = -1
            prelimit = 0
            for key in plot_dictionary.keys():
                if key == "recall" or key=="prc" or key == "anomaly_ranges" or key == "lead_ranges":
                    continue
                counter+=1
                data_to_plot=plot_dictionary[key]
                current_range = plot_dictionary["anomaly_ranges"][prelimit:prelimit+len(data_to_plot["scores"])]
                current_range_lead = plot_dictionary["lead_ranges"][prelimit:prelimit+len(data_to_plot["scores"])]

                prelimit += len(data_to_plot["scores"])
                plt.subplot(910+counter)
                # print()
                plt.plot(data_to_plot["index"], data_to_plot["scores"], ".-", color="black", label="anomaly score")
                plt.plot(data_to_plot["index"], [best_metrics_dict["threshold_auc"] for i in range(len(data_to_plot["index"]))], ".-", color="dodgerblue", label="best threshold")

                for date in data_to_plot["failures"]:
                    plt.axvline(date,color="red")

                # plot PH
                self._inner_plot("red", current_range, data_to_plot["index"], min(data_to_plot["scores"]), max(data_to_plot["scores"]), "predictive horizon")

                # plot lead
                self._inner_plot("grey", current_range_lead, data_to_plot["index"], min(data_to_plot["scores"]), max(data_to_plot["scores"]), "lead time")
                plt.legend(loc="center left")
                if counter==9:
                    namescount+=1
                    mlflow.log_figure(plt.gcf(), f"scores_{namescount*9}_{namescount*9+counter}.png")
                    plt.clf()
                    counter=0
            if counter>0:
                namescount += 1
                mlflow.log_figure(plt.gcf(), f"scores_{namescount * 9}_{namescount * 9 + counter}.png")
                plt.clf()
                counter = 0


    def _finish_run(self, parent_run, current_steps) -> None:
        if 'many' in current_steps['method'].get_library():
            model_sources, models = current_steps['method'].get_all_models()
            for model_source, model in zip(model_sources, models):
                current_subpackage = getattr(mlflow, re.sub('many_', '', current_steps['method'].get_library()))
                current_submodule = current_subpackage.log_model
                # TODO do not use self.artifacts
                current_submodule(model, f'{self.artifacts}/{str(current_steps["method"])}_source_{model_source}')
        elif current_steps['method'].get_library() == 'no_save':
            pass
        else:
            # TODO we should check if there is a log_model functionality for the method we have in the current run
            # TODO do not use self.artifacts
            current_subpackage = getattr(mlflow, current_steps['method'].get_library())
            current_submodule = current_subpackage.log_model
            current_submodule(current_steps['method'], f'{self.artifacts}/{str(current_steps["method"])}')

        # log parameters for each step
        for step in self.pipeline.get_steps().keys():
                mlflow.log_params({
                    f'{step}_{key}': str(value)[:499] for key, value in current_steps[step].get_params().items()
                })
                mlflow.log_param(step, str(current_steps[step]))

        if "anomaly_ranges" in self.pipeline.dataset.keys():
            mlflow.log_param('anomaly_ranges', self.pipeline.dataset['anomaly_ranges'])
            if self.pipeline.dataset["anomaly_ranges"]:
                mlflow.log_params({
                    'predictive_horizon': self.pipeline.slide,
                    'beta': self.pipeline.beta,
                    'lead': self.pipeline.slide
                })
            else:
                mlflow.log_params({
                    'predictive_horizon': self.pipeline.predictive_horizon,
                    'beta': self.pipeline.beta,
                    'lead': self.pipeline.lead
                })
        else:
            mlflow.log_params({
                'predictive_horizon': self.pipeline.predictive_horizon,
                'beta': self.pipeline.beta,
                'lead': self.pipeline.lead
            })

        mlflow.log_params({
            'slide': self.pipeline.dataset['slide'],
            'auc_resolution': self.pipeline.auc_resolution,
            'min_historic_scenario_len': self.pipeline.dataset['min_historic_scenario_len'],
            'min_target_scenario_len': self.pipeline.dataset['min_target_scenario_len'],
            'max_wait_time': self.pipeline.dataset['max_wait_time']
        })

        if 'reset_after_fail' in self.pipeline.dataset:
            mlflow.log_param('reset_after_fail', self.pipeline.dataset['reset_after_fail'])
        
        if 'setup_1_period' in self.pipeline.dataset:
            mlflow.log_param('setup_1_period', self.pipeline.dataset['setup_1_period'])
        

        current_steps['method'].destruct()
    

    def _finish_experiment(self, best_params: dict) -> dict:
        # Mango uses scikit learn and due to the autolog functionality it logs some runs to the default experiment, so we need to clear the default experiment to avoid confusion
        default_experiment_id = mlflow.get_experiment_by_name("Default").experiment_id

        runs = mlflow.search_runs(experiment_ids=default_experiment_id)

        for run in runs.iterrows():
            run_id = run[1]['run_id']
            mlflow.delete_run(run_id)

        if self.log_best_scores and os.path.exists(self.best_scores_info_dict_path):
            with open(self.best_scores_info_dict_path, 'rb') as file:
                best_scores_info_saved_dict = pickle.load(file)
                best_run_id = best_scores_info_saved_dict['best_run_id']
                pd.DataFrame(best_scores_info_saved_dict['best_scores']).to_csv(f'scores_{best_run_id}.csv', index=False, header=False)
            
                with mlflow.start_run(run_id=best_run_id, experiment_id=self.experiment_id):
                    mlflow.log_artifact(f'scores_{best_run_id}.csv')
                    os.remove(f'scores_{best_run_id}.csv')

            os.remove(self.best_scores_info_dict_path)

            os.remove(self.lock_file_path)
        
        return best_params
    

    def _evaluate(self, result_scores, result_dates, results_isfailure, plot_dictionary):
        if "anomaly_ranges" in self.pipeline.dataset.keys():
            if self.pipeline.dataset["anomaly_ranges"]:
                allresults, results_vus, anomaly_ranges, lead_ranges = pdm_evaluate_ranges(
                                                                            result_scores,
                                                                            anomalyranges=self.pipeline.dataset["predictive_horizon"],
                                                                            leadranges=self.pipeline.dataset["lead"],
                                                                            beta=self.pipeline.beta,
                                                                            resolution=self.pipeline.auc_resolution,
                                                                            slidingWindow_vus=self.pipeline.slide
                                                                        )
            else:
                allresults, results_vus, anomaly_ranges, lead_ranges = pdm_evaluate(
                                                                            result_scores, 
                                                                            datesofscores=result_dates, 
                                                                            isfailure=results_isfailure, 
                                                                            PH=self.pipeline.predictive_horizon, 
                                                                            lead=self.pipeline.lead, 
                                                                            beta=self.pipeline.beta, 
                                                                            resolution=self.pipeline.auc_resolution, 
                                                                            slidingWindow_vus=self.pipeline.slide
                                                                        )
        else:
            allresults, results_vus, anomaly_ranges, lead_ranges = pdm_evaluate(
                                                                            result_scores, 
                                                                            datesofscores=result_dates, 
                                                                            isfailure=results_isfailure, 
                                                                            PH=self.pipeline.predictive_horizon, 
                                                                            lead=self.pipeline.lead, 
                                                                            beta=self.pipeline.beta, 
                                                                            resolution=self.pipeline.auc_resolution, 
                                                                            slidingWindow_vus=self.pipeline.slide
                                                                        )


        recalls=[]
        precisions=[]
        for row in allresults:
            recalls.append(row[3])
            precisions.append(row[6])
        plot_dictionary["recall"]=recalls
        plot_dictionary["prc"]=precisions
        plot_dictionary["anomaly_ranges"] = anomaly_ranges
        plot_dictionary["lead_ranges"] = lead_ranges
        
        all_results_appended_with_vus = []
        results_vus_keys = list(results_vus.keys())
        for row in allresults:
            result_to_append = []
            result_to_append.extend(row)
            result_to_append.extend([results_vus[key] for key in results_vus_keys])
            all_results_appended_with_vus.append(result_to_append)

        param_name_to_index_dict = {
            'AD1_rcl': 3,
            'AD2_rcl': 4,
            'AD3_rcl': 5,
            'prc': 6,
            'AD1_f1': 0,
            'AD2_f1': 1,
            'AD3_f1': 2,
            'AD1_AUC': 8,
            'AD2_AUC': 9,
            'AD3_AUC': 10,
            'threshold_auc': 7,
        }

        for index, key in enumerate(results_vus_keys):
            param_name_to_index_dict['VUS_' + key] = index + 11

        best_dict = {
            'AD1_rcl': -1,
            'AD2_rcl': -1,
            'AD3_rcl': -1,
            'prc': -1,
            'AD1_f1': -1,
            'AD2_f1': -1,
            'AD3_f1': -1,
            'AD1_AUC': -1,
            'AD2_AUC': -1,
            'AD3_AUC': -1,
            'threshold_auc': -1,
        }

        for metric in results_vus_keys:
            best_dict[f"VUS_{metric}"] = -1

        metric_index_to_choose_best_from = param_name_to_index_dict[self.optimization_param] if self.optimization_param != 'AD1_AUC' else param_name_to_index_dict['AD1_f1']
        best_row_index = -1
        for current_row_index, row in enumerate(all_results_appended_with_vus):
            if row[metric_index_to_choose_best_from] > best_dict[self.optimization_param if self.optimization_param != 'AD1_AUC' else 'AD1_f1']:                        
                    best_dict = {}
                    for key, index in param_name_to_index_dict.items():
                        best_dict[key] = row[index]

                    best_row_index = current_row_index

        best_dict_to_log = {key: value for key, value in best_dict.items()}

        mlflow.log_metrics(best_dict_to_log)

        current_run = mlflow.active_run()

        if self.log_best_scores:
            with locket.lock_file(self.lock_file_path):
                if not os.path.exists(self.best_scores_info_dict_path):
                    best_scores_info_dict_to_write_on_disk = {
                        'best_scores': result_scores,
                        'best_optimization_value': best_dict_to_log[self.optimization_param],
                        'best_run_id': current_run.info.run_id
                    }

                    with open(self.best_scores_info_dict_path, 'wb') as file:
                        pickle.dump(best_scores_info_dict_to_write_on_disk, file)
                else:
                    best_scores_info_dict_to_write_on_disk = None
                    with open(self.best_scores_info_dict_path, 'rb') as file:
                        previously_saved_dict = pickle.load(file)

                        if best_dict_to_log[self.optimization_param] > previously_saved_dict['best_optimization_value']:
                            best_scores_info_dict_to_write_on_disk = {
                                'best_scores': result_scores,
                                'best_optimization_value': best_dict_to_log[self.optimization_param],
                                'best_run_id': current_run.info.run_id
                            }

                    if best_scores_info_dict_to_write_on_disk is not None:
                        with open(self.best_scores_info_dict_path, 'wb') as file:
                            pickle.dump(best_scores_info_dict_to_write_on_disk, file)
                
        all_results_appended_with_vus.pop(best_row_index)

        metrics_for_other_thresholds_df = pd.DataFrame(all_results_appended_with_vus, columns=['AD1_f1', 'AD2_f1', 'AD3_f1', 'AD1_rcl', 'AD2_rcl', 'AD3_rcl', 'prc', 'threshold_auc', 'AD1_AUC', 'AD2_AUC', 'AD3_AUC'] + [f'VUS_{key}' for key in results_vus_keys])
        
        metrics_for_other_thresholds_df.to_csv(f'./metrics_for_other_thresholds_{current_run.info.run_id}.csv', index=False)
        mlflow.log_artifact(f'./metrics_for_other_thresholds_{current_run.info.run_id}.csv')
        os.remove(f'./metrics_for_other_thresholds_{current_run.info.run_id}.csv')

        return best_dict
    

    def _check_cached_run(self, params: dict):
        current_params = params.copy()

        if 'profile_size' in current_params:
            current_params['auto_flavor_profile_size'] = current_params['profile_size']
            del current_params['profile_size']

        method_params = {re.sub('method_', '', k): v for k, v in current_params.items() if 'method' in k}
        preprocessor_params = {re.sub('preprocessor_', '', k): v for k, v in current_params.items() if 'preprocessor' in k}
        postprocessor_params = {re.sub('postprocessor_', '', k): v for k, v in current_params.items() if 'postprocessor' in k}
        thresholder_params = {re.sub('thresholder_', '', k): v for k, v in current_params.items() if 'thresholder' in k}

        runs = mlflow.search_runs(self.experiment_id, filter_string='attributes.status = "FINISHED"')
        
        found_match, found_index, found_run = False, -1, None
        for index, current_run in runs.iterrows():
            found_match = True
            found_index = index
            found_run = current_run

            for param_name, param_value in current_params.items():
                if 'params.' + param_name not in current_run.index:
                    found_match = False
                    break

                if current_run.loc['params.' + param_name] != str(param_value):
                    found_match = False
                    break
            
            current_steps = {
                'method': self.pipeline.method(event_preferences=self.pipeline.event_preferences, **method_params),
                'preprocessor': self.pipeline.preprocessor(event_preferences=self.pipeline.event_preferences, **preprocessor_params),
                'postprocessor': self.pipeline.postprocessor(event_preferences=self.pipeline.event_preferences, **postprocessor_params),
                'thresholder': self.pipeline.thresholder(event_preferences=self.pipeline.event_preferences, **thresholder_params)
            }

            for step in self.pipeline.get_steps().keys():
                if current_run.loc['params.' + step] != str(current_steps[step]):
                    found_match = False
                    break
            
            predictive_horizon_to_check, beta_to_check, lead_to_check = -1, -1, -1
            if "anomaly_ranges" in self.pipeline.dataset.keys():
                if self.pipeline.dataset["anomaly_ranges"]:
                    predictive_horizon_to_check = self.pipeline.slide
                    beta_to_check = self.pipeline.beta
                    lead_to_check = self.pipeline.slide
                else:
                    predictive_horizon_to_check = self.pipeline.predictive_horizon
                    beta_to_check = self.pipeline.beta
                    lead_to_check = self.pipeline.lead
            else:
                predictive_horizon_to_check = self.pipeline.predictive_horizon
                beta_to_check = self.pipeline.beta
                lead_to_check = self.pipeline.lead

            if str(predictive_horizon_to_check) != current_run.loc['params.predictive_horizon'] \
                or str(beta_to_check) != current_run.loc['params.beta'] \
                or str(lead_to_check) != current_run.loc['params.lead']:
                    found_match = False

            if found_match:
                break

        if found_match:
            logging.info(f'Found cached run with parameters: {current_params}, steps={[str(step) for step in current_steps.values()]}, predictive_horizon={predictive_horizon_to_check}, beta={beta_to_check} and lead={lead_to_check}. Skipping...')
            return found_run.loc['metrics.' + self.optimization_param]
        else:
            return None
