import pandas as pd

from preprocessing.record_level.record_level_pre_processor import RecordLevelPreProcessorInterface
from pdm_evaluation_types.types import EventPreferences


class MeanAggregator(RecordLevelPreProcessorInterface):
    def __init__(self, event_preferences: EventPreferences, period= '10T'):
        super().__init__(event_preferences=event_preferences)
        self.period=period
    def fit(self, historic_data: list[pd.DataFrame], historic_sources: list[str], event_data: pd.DataFrame) -> None:
        pass

    def transform(self, target_data: pd.DataFrame, source: str, event_data: pd.DataFrame) -> pd.DataFrame:
        df = target_data.resample(self.period).mean()
        df.dropna(inplace=True)
        return df

    def transform_one(self, new_sample: pd.Series, source: str, is_event: bool) -> pd.Series:
        pass

    def get_params(self):
        return {
            'period': self.period
        }

    def __str__(self) -> str:
        return 'MeanAggregator'