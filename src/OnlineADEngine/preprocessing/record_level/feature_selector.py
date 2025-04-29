import pandas as pd

from preprocessing.record_level.record_level_pre_processor import RecordLevelPreProcessorInterface
from pdm_evaluation_types.types import EventPreferences


class FeatureSelector(RecordLevelPreProcessorInterface):
    def __init__(self, event_preferences: EventPreferences, selected_features: list[str]):
        super().__init__(event_preferences=event_preferences)
        self.selected_features = selected_features


    def fit(self, historic_data: list[pd.DataFrame], historic_sources: list[str], event_data: pd.DataFrame) -> None:
        pass
        

    def transform(self, target_data: pd.DataFrame, source: str, event_data: pd.DataFrame) -> pd.DataFrame:
        if len(self.selected_features)==0:
            return target_data
        return target_data[self.selected_features]


    def transform_one(self, new_sample: pd.Series, source: str, is_event: bool) -> pd.Series:
        return new_sample[self.selected_features]
    

    def get_params(self):
        return {
            'features': self.selected_features
        }
    

    def __str__(self) -> str:
        return 'Feature_Selector'