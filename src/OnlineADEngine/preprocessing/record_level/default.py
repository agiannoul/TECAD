import pandas as pd

from preprocessing.record_level.record_level_pre_processor import RecordLevelPreProcessorInterface


class DefaultPreProcessor(RecordLevelPreProcessorInterface):
    def fit(self, historic_data: list[pd.DataFrame], historic_sources: list[str], event_data: pd.DataFrame) -> None:
        pass
        

    def transform(self, target_data: pd.DataFrame, source: str, event_data: pd.DataFrame) -> pd.DataFrame:
        return target_data


    def transform_one(self, new_sample: pd.Series, source: str, is_event: bool) -> pd.Series:
        return new_sample
    

    def get_params(self):
        return {}
    
    
    def __str__(self) -> str:
        return 'Default'