import re
import time
import gc

import numpy as np
import pandas as pd
import mlflow
from src.OnlineADEngine.mango import scheduler, Tuner

from src.OnlineADEngine.experiment.experiment import PdMExperiment
from evaluation.evaluation import AUCPR_new as pdm_evaluate, breakIntoEpisodes as split_into_episodes

from method.semi_supervised_method import SemiSupervisedMethodInterface
from exceptions.exception import IncompatibleMethodException


class AutoProfileSemiSupervisedPdMExperiment(PdMExperiment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.extra_metrics = {}

    def execute(self) -> dict:
        super()._register_experiment()
        conf_dict = {
            'initial_random': self.initial_random,
            'num_iteration': self.num_iteration,
            'constraint': self.constraint_function
            # 'batch_size': self.batch_size, currently commented out because of using only scheduler.parallel, more info on issue #97 on Mango - alternatives include using only scheduler.parallel or letting the user decide depending on his hardware
        }

        @scheduler.parallel(n_jobs=self.n_jobs)
        def optimization_objective(**params: dict):
            gc.collect()
            # cached_result = self._check_cached_run(params)
            cached_result= None
            if cached_result is not None:
                return cached_result

            with mlflow.start_run(experiment_id=self.experiment_id) as parent_run:

                result_scores = []
                result_dates = []
                result_thresholds = []
                results_isfailure =[]
                plot_dictionary={}

                if isinstance(self.pipeline.event_preferences['failure'], list):
                    if len(self.pipeline.event_preferences['failure']) == 0:
                        run_to_failure_scenarios = True
                    else:
                        run_to_failure_scenarios = False
                elif self.pipeline.event_preferences['failure'] is None:
                    run_to_failure_scenarios = True
                else:
                    run_to_failure_scenarios = False

                profile_size = params['profile_size']

                if "init_profile_size" not in  params.keys():
                    init_profile_size=profile_size
                else:
                    init_profile_size = params['init_profile_size']
                method_params = {re.sub('method_', '', k): v for k, v in params.items() if 'method' in k}
                method_params['profile_size'] = profile_size
                #print(method_params)
                mlflow.log_param('auto_flavor_profile_size', profile_size)

                current_method = self.pipeline.method(event_preferences=self.pipeline.event_preferences, **method_params)

                if not isinstance(current_method, SemiSupervisedMethodInterface):
                    raise IncompatibleMethodException('Expected a semi-supervised method to be provided')

                preprocessor_params = {re.sub('preprocessor_', '', k): v for k, v in params.items() if 'preprocessor' in k}
                current_preprocessor = self.pipeline.preprocessor(event_preferences=self.pipeline.event_preferences, **preprocessor_params)

                postprocessor_params = {re.sub('postprocessor_', '', k): v for k, v in params.items() if 'postprocessor' in k}
                #print(postprocessor_params)
                current_postprocessor = self.pipeline.postprocessor(event_preferences=self.pipeline.event_preferences, **postprocessor_params)

                thresholder_params = {re.sub('thresholder_', '', k): v for k, v in params.items() if 'thresholder' in k}
                current_thresholder = self.pipeline.thresholder(event_preferences=self.pipeline.event_preferences, **thresholder_params)
                #try:
                first_target = True
                for current_target_data, current_target_source in zip(self.target_data, self.target_sources):
                    if not first_target and self.delay is not None:
                        print(f'Cooldown for {self.delay} milliseconds')
                        time.sleep(self.delay / 1000)

                    first_target = False

                    current_failure_dates = self.pipeline.extract_failure_dates_for_source(current_target_source)
                    current_reset_dates = self.pipeline.extract_reset_dates_for_source(current_target_source)

                    current_dates = self.pipeline.target_dates
                    processed_dates=[]
                    # if the user passed a string take the corresponding column of the target_data as 'dates' for the evaluation
                    if isinstance(current_dates, str):
                        name=current_dates
                        current_dates = pd.to_datetime(current_target_data[current_dates])
                        current_dates=[date for date in current_dates]
                        # also drop the corresponding column from the target_data df
                        current_target_data = current_target_data.drop(name, axis=1)

                    current_target_data.index = current_dates

                    if len(current_reset_dates) != 0:
                        if current_reset_dates[-1] < current_target_data.index[-1]:
                            current_reset_dates.append(current_target_data.index[-1])
                        else:
                            current_reset_dates[-1]=current_target_data.index[-1]
                    else:
                        current_reset_dates.append(current_target_data.index[-1])

                    processed_target_scores = []
                    current_thresholds = []
                    last_date_used = current_dates[0]

                    # new
                    profile_size=init_profile_size

                    for reset_date_index, reset_date in enumerate(current_reset_dates):
                        current_target_data_until_reset = current_target_data.loc[last_date_used:reset_date] # NOTE this is inclusive

                        if reset_date in current_target_data_until_reset.index and reset_date_index != len(current_reset_dates) - 1:
                            reset_date_index_pos = current_target_data.index.get_loc(reset_date)
                            last_date_used = current_target_data.index[reset_date_index_pos + 1]
                        else:
                            last_date_used = reset_date

                        if current_target_data_until_reset.shape[0] > profile_size:
                            # fit preprocessor only on profile data
                            current_preprocessor.fit([current_target_data_until_reset.iloc[:profile_size]], [current_target_source], self.event_data)

                            current_target_data_until_reset = current_preprocessor.transform(current_target_data_until_reset, current_target_source, self.event_data)
                            processed_dates.extend([dttt for dttt in current_target_data_until_reset.index])

                            profile = current_target_data_until_reset.iloc[:profile_size]
                            current_target_data_after_profile = current_target_data_until_reset.iloc[profile_size:]

                            current_method.fit([profile], [current_target_source], self.event_data)

                            # output 0 score for profile data points
                            current_target_scores_until_reset = current_method.predict(current_target_data_after_profile, current_target_source, self.event_data)

                            processed_target_scores_until_reset = current_postprocessor.transform(current_target_scores_until_reset, current_target_source, self.event_data)
                            if len(processed_target_scores)>0:
                                tofill=min(processed_target_scores)
                            else:
                                tofill=min(processed_target_scores_until_reset)
                            processed_target_scores_until_reset = [tofill for i in range(profile.shape[0])] + processed_target_scores_until_reset

                            current_thresholds_until_reset = current_thresholder.infer_threshold(processed_target_scores_until_reset, current_target_source, self.event_data, [ind for ind in current_target_data_until_reset.index])
                        else: # not enough data for profile construction
                            if len(processed_target_scores)>0:
                                tofill=min(processed_target_scores)
                            else:
                                tofill=0
                            processed_target_scores_until_reset = [tofill for i in range(current_target_data_until_reset.shape[0])]
                            current_thresholds_until_reset = [0.1 for i in range(current_target_data_until_reset.shape[0])]
                            processed_dates.extend([dttt for dttt in current_target_data_until_reset.index])
                        assert current_target_data_until_reset.shape[0] == len(processed_target_scores_until_reset), f"{current_target_data_until_reset.shape[0]} != {len(processed_target_scores_until_reset)}"

                        processed_target_scores.extend(processed_target_scores_until_reset)
                        current_thresholds.extend(current_thresholds_until_reset)
                        # new
                        profile_size = params["profile_size"]
                    assert len(processed_dates) == len(processed_target_scores)

                    if self.debug:
                        plot_dictionary[current_target_source]={"scores":processed_target_scores,"failures":current_failure_dates,"thresholds":current_thresholds,"index":processed_dates}

                    if not run_to_failure_scenarios:
                        is_failure, current_scores_splitted, current_dates_splitted, current_thresholds_splitted = split_into_episodes(processed_target_scores, current_failure_dates, current_thresholds, processed_dates)
                    else:
                        is_failure = [1]
                        current_scores_splitted = [processed_target_scores]
                        current_dates_splitted = [processed_dates]
                        current_thresholds_splitted = [current_thresholds]


                    result_thresholds.extend(current_thresholds_splitted)
                    result_scores.extend(current_scores_splitted)
                    results_isfailure.extend(is_failure)
                    result_dates.extend(current_dates_splitted)


                best_metrics_dict = self._evaluate(result_scores, result_dates, results_isfailure, plot_dictionary)
                self.extra_metrics["th"]=best_metrics_dict["threshold_auc"]
                self._plot_scores(plot_dictionary, best_metrics_dict)

                self._finish_run(parent_run=parent_run, current_steps={
                    'preprocessor': current_preprocessor,
                    'method': current_method,
                    'postprocessor': current_postprocessor,
                    'thresholder': current_thresholder
                })

            return best_metrics_dict[self.optimization_param]

        tuner = Tuner(self.param_space, optimization_objective, conf_dict=conf_dict)
        results = tuner.maximize()
        dict_ro_return={}
        dict_ro_return['best_params']=results['best_params']
        dict_ro_return["best_objective"]=results["best_objective"]
        dict_ro_return["th"]=self.extra_metrics["th"]
        return self._finish_experiment(dict_ro_return)

