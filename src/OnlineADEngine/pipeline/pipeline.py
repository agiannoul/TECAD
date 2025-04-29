from typing import TypedDict, List

import pandas as pd

from preprocessing.record_level.default import DefaultPreProcessor 
from method.semi_supervised_method import SemiSupervisedMethodInterface
from postprocessing.default import DefaultPostProcessor
from thresholding.constant import ConstantThresholder
from thresholding.thresholder import ThresholderInterface
from pdm_evaluation_types.types import EventPreferences
from preprocessing.record_level.record_level_pre_processor import RecordLevelPreProcessorInterface 
from method.method import MethodInterface
from postprocessing.post_processor import PostProcessorInterface
from utils.utils import expand_event_preferences


class PdMPipelineSteps(TypedDict):
    preprocessor:  RecordLevelPreProcessorInterface
    method : MethodInterface
    postprocessor : PostProcessorInterface
    thresholder : ThresholderInterface


class PdMPipeline():
    def __init__(self, 
                steps: PdMPipelineSteps,                 
                dataset: dict,
                auc_resolution : int
    ):        
        self.dataset = dataset
        self.steps = steps
        self.event_data = dataset['event_data']
        self.event_data['date']=pd.to_datetime( self.event_data['date'])
        self.event_preferences = dataset['event_preferences']
        self.target_dates = dataset['dates']
        self.historic_dates = dataset['dates']
        self.predictive_horizon = dataset['predictive_horizon']
        self.slide = dataset['slide']
        self.lead = dataset['lead']
        self.beta = dataset['beta']
        self.auc_resolution = auc_resolution

        self.preprocessor = steps.get('preprocessor', DefaultPreProcessor(event_preferences=self.event_preferences))
        self.method = steps.get('method', SemiSupervisedMethodInterface)
        self.postprocessor = steps.get('postprocessor', DefaultPostProcessor(event_preferences=self.event_preferences))
        
        self.thresholder = steps.get('thresholder', ConstantThresholder(threshold_value=0.5, event_preferences=self.event_preferences))
        

    def get_steps(self) -> PdMPipelineSteps:
        return self.steps


    def get_step_by_name(self, step_name: str):
        return self.steps[step_name]

    
    def extract_failure_dates_for_source(self, source: str) -> list[pd.Timestamp]:
        result = []
        expanded_event_preferences = expand_event_preferences(event_data=self.event_data, event_preferences=self.event_preferences) 
        for current_preference in expanded_event_preferences['failure']:
            matched_rows = self.event_data.loc[(self.event_data['type'] == current_preference.type) & (self.event_data['source'] == current_preference.source) & (self.event_data['description'] == current_preference.description)]
            for row_index, row in matched_rows.iterrows():
                if current_preference.target_sources == '=' and str(row.source) == str(source):
                    result.append(row['date'])
                elif source in current_preference.target_sources:
                    result.append(row['date'])
                elif current_preference.target_sources == '*':
                    result.append(row['date'])
        return sorted(list(set(result)))

    
    def extract_reset_dates_for_source(self, source) -> list[pd.Timestamp]:
        result = []
        expanded_event_preferences = expand_event_preferences(event_data=self.event_data, event_preferences=self.event_preferences) 
        for current_preference in expanded_event_preferences['reset']:
            matched_rows = self.event_data.loc[(self.event_data['type'] == current_preference.type) & (self.event_data['source'] == current_preference.source) & (self.event_data['description'] == current_preference.description)]
            for row_index, row in matched_rows.iterrows():
                if current_preference.target_sources == '=' and str(row.source) == str(source):
                    result.append(row['date'])
                elif source in current_preference.target_sources:
                    result.append(row['date'])
                elif current_preference.target_sources == '*':
                    result.append(row['date'])

        return sorted(list(set(result)))
    

    def get_steps_as_str(self):
        return f'preprocessor_{self.steps["preprocessor"]}_method_{self.steps["method"]}_postprocessor_{self.steps["postprocessor"]}_thresholder_{self.steps["thresholder"]}'