import abc

import pandas as pd

from pdm_evaluation_types.types import EventPreferences


class RecordLevelPreProcessorInterface(abc.ABC):
    def __init__(self, event_preferences: EventPreferences):
        self.event_preferences = event_preferences


    @abc.abstractmethod
    def fit(self, historic_data: list[pd.DataFrame], historic_sources: list[str], event_data: pd.DataFrame) -> None:
        pass
        

    @abc.abstractmethod
    def transform(self, target_data: pd.DataFrame, source: str, event_data: pd.DataFrame) -> pd.DataFrame:
        pass


    @abc.abstractmethod
    def transform_one(self, new_sample: pd.Series, source: str, is_event: bool) -> pd.Series:
        pass


    @abc.abstractmethod
    def get_params(self):
        pass


    @abc.abstractmethod
    def __str__(self) -> str:
        pass