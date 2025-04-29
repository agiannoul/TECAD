import abc

import pandas as pd

from pdm_evaluation_types.types import EventPreferences


class PostProcessorInterface(abc.ABC):
    def __init__(self, event_preferences: EventPreferences):
        self.event_preferences = event_preferences


    @abc.abstractmethod
    def transform(self, scores: list[float], source: str, event_data: pd.DataFrame) -> list[float]:
        pass


    @abc.abstractmethod
    def transform_one(self, score_point: float, source: str, is_event: bool) -> float:
        pass


    @abc.abstractmethod
    def get_params(self):
        pass
    

    @abc.abstractmethod
    def __str__(self) -> str:
        pass