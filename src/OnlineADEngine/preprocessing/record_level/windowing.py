from utils import utils
import pandas as pd
from preprocessing.record_level.record_level_pre_processor import RecordLevelPreProcessorInterface
from pdm_evaluation_types.types import EventPreferences



class Windowing(RecordLevelPreProcessorInterface):
    def __init__(self, event_preferences: EventPreferences,slidingWindow=None,col_pos=0):
        super().__init__(event_preferences=event_preferences)

        self.col_pos=col_pos
        self.slidingWindow=slidingWindow

    def _sequencing_Univariate_data(self, df):
        data = df[df.columns[self.col_pos]].values
        if self.slidingWindow is None:
            return df
        elif self.slidingWindow < 2:
            slidingWindow = utils.find_length(data)
        else:
            slidingWindow = self.slidingWindow

        X_data = utils.Window(window=slidingWindow).convert(data).to_numpy()

        new_df = {}
        for col in range(X_data.shape[1]):
            new_df[f"s_{col}"] = X_data[:, col]
        for col in df.columns:
            if col != df.columns[self.col_pos]:
                new_df[col] = df[col].values

        new_df = pd.DataFrame(new_df)

        row = new_df.iloc[0]
        repeat_times = df.shape[0] - new_df.shape[0]
        repeated_rows = pd.concat([row.to_frame().transpose()] * repeat_times, ignore_index=True)
        new_df = pd.concat([repeated_rows, new_df], ignore_index=True)
        new_df.index = df.index
        return new_df



    def fit(self, historic_data: list[pd.DataFrame], historic_sources: list[str], event_data: pd.DataFrame) -> None:
        pass

    def transform(self, target_data: pd.DataFrame, source: str, event_data: pd.DataFrame) -> pd.DataFrame:
        return self._sequencing_Univariate_data(target_data)

    def transform_one(self, new_sample: pd.Series, source: str, is_event: bool) -> pd.Series:
        pass

    def get_params(self):
        return {"col_pos":self.col_pos,
                 "slidingWindow":self.slidingWindow}

    def __str__(self) -> str:
        return 'Windowing'