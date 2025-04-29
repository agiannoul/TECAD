import abc

import pandas as pd

from pdm_evaluation_types.types import EventPreferences


class MethodInterface(abc.ABC):
    def __init__(self, event_preferences: EventPreferences):
        self.event_preferences = event_preferences
        

    @abc.abstractmethod
    def predict(self, target_data: pd.DataFrame, source: str, event_data: pd.DataFrame) -> list[float]:
        """
        Abstract method, for predict implementation of the methods.

        :param target_data: a dataframe containing the different features in columns and dates in index (ordered according to index)
        :param source: the name of the source, to which the target_data refer
        :param event_data: a Dataframe that correspond to discrete information (events) with columns : "description","date","type","source"
        :return: a list of anomaly scores (float values) with length equal to the target_data.shape[0] (the higher score means more anomalous data)
        """
        pass


    @abc.abstractmethod
    def predict_one(self, new_sample: pd.Series, source: str, is_event: bool) -> float:
        """
        This method implements the logic of predict, but appropriate for streaming usage, passing each time one vectors (row of a dataframe) instead the whole scenario

        :param new_sample: sample of that data for which the anomaly score will be calculated
        :param source: the name of the source, to which the new_sample refers
        :param is_event: whether the new_sample corresponds to an event timestamp
        :return:
        """
        pass

    @abc.abstractmethod
    def get_library(self) -> str:
        # TODO we could also try to return a reference to the corresponding subpackage if it works
        pass


    @abc.abstractmethod
    def get_params(self) -> dict:
        pass


    @abc.abstractmethod
    def __str__(self) -> str:
        """
            Returns a string representation of the corresponding method
        """
        pass


    @abc.abstractmethod
    def get_all_models(self):
        pass

    
    def destruct(self):
        pass