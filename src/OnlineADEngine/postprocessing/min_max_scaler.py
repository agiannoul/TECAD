import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler as SKLearnMinMaxScaler

from postprocessing.post_processor import PostProcessorInterface
from pdm_evaluation_types.types import EventPreferences


class MinMaxPostProcessor(PostProcessorInterface):
    def __init__(self, event_preferences: EventPreferences):
        super().__init__(event_preferences=event_preferences)
        self.scores_buffer_per_source = {}


    def transform(self, scores: list[float], source: str, event_data: pd.DataFrame) -> list[float]:
        scaler = SKLearnMinMaxScaler()
        scaler.fit(np.array(scores).reshape(-1, 1))

        return scaler.transform(np.array(scores).reshape(-1, 1)).ravel().tolist()
    

    def transform_one(self, score_point: float, source: str, is_event: bool) -> float:
        # TODO test if it crashes when passing one point only to scaler
        if source not in self.scores_buffer_per_source:
            self.scores_buffer_per_source[source] = []
        
        self.scores_buffer_per_source[source].append(score_point)

        scaler = SKLearnMinMaxScaler()
        scaler.fit(np.array(self.scores_buffer_per_source[source]).reshape(-1, 1))

        return scaler.transform(np.array([score_point]).reshape(-1, 1)).ravel().tolist()[0]
    

    def get_params(self):
        return {}


    def __str__(self) -> str:
        return f'MinMaxScaler'