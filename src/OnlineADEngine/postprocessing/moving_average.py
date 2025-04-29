import pandas as pd

from postprocessing.post_processor import PostProcessorInterface
from pdm_evaluation_types.types import EventPreferences


class MovingAveragePostProcessor(PostProcessorInterface):
    def __init__(self, event_preferences: EventPreferences, window_length: int):
        super().__init__(event_preferences=event_preferences)
        self.window_length = window_length
        self.scores_buffer_per_source = {}


    def transform(self, scores: list[float], source: str, event_data: pd.DataFrame) -> list[float]:
        if self.window_length<=0 or self.window_length>=len(scores):
            return scores
        # use first self.window_length scores in order to avoid NaN values in the resulting scores
        result = scores[:self.window_length] + pd.Series(scores).rolling(window=self.window_length).mean().tolist()[self.window_length:]
        assert len(result) == len(scores)

        return result
    

    def transform_one(self, score_point: float, source: str, is_event: bool) -> float:
        if source not in self.scores_buffer_per_source:
            self.scores_buffer_per_source[source] = []
        
        self.scores_buffer_per_source[source].append(score_point)

        if len(self.scores_buffer_per_source[source]) < self.window_length:
            return score_point
        
        self.scores_buffer_per_source[source] = self.scores_buffer_per_source[source][-self.window_length:]

        assert len(self.scores_buffer_per_source[source]) == self.window_length

        return sum(self.scores_buffer_per_source[source]) / len(self.scores_buffer_per_source[source])
    

    def get_params(self):
        return {
            'window_length': self.window_length
        }


    def __str__(self) -> str:
        return f'Moving_Average'