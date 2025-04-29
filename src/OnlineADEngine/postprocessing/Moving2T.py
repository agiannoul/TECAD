import statistics

import numpy as np
import pandas as pd
from operator import itemgetter
import datetime
from tqdm import tqdm
from postprocessing.post_processor import PostProcessorInterface
from pdm_evaluation_types.types import EventPreferences


class Moving2Thresholder(PostProcessorInterface):
    def __init__(self, event_preferences: EventPreferences, factor: float = 3,history_window=None,exclude=False):
        super().__init__(event_preferences=event_preferences)
        self.factor = factor
        self.history_window = history_window
        self.exclude=exclude
        self.anomaly_scores_dict={}


    def transform(self, scores: list[float], source: str, event_data: pd.DataFrame):
        self.anomaly_scores_dict[source]=[]
        new_scores=[]
        for qi in range(len(scores)):
            sc=scores[qi]
            self.anomaly_scores_dict[source].append(sc)
            if self.exclude:
                succed, th = Moving2Texclude(self.anomaly_scores_dict[source],new_scores, factor=self.factor,
                                      hscaleCount=self.history_window)
            else:
                succed, th = Moving2T(self.anomaly_scores_dict[source], factor=self.factor, hscaleCount=self.history_window)

            if sc > th:
                new_scores.append(1)
            else:
                new_scores.append(0)
        return new_scores

    def transform_one(self, score_point: float, source: str, is_event: bool) -> float:
        if source in  self.anomaly_scores_dict.keys():
            self.anomaly_scores_dict[source].append(score_point)
        else:
            self.anomaly_scores_dict[source]=[score_point]
        succed,th=Moving2T(self.anomaly_scores_dict[source], factor=self.factor, hscaleCount=self.history_window)

        if score_point>th:
            return 1
        else:
            return 0

    def get_params(self):
        return {
            'factor': self.factor,
            'history_window':self.history_window,
            'exclude':self.exclude,
        }

    def __str__(self) -> str:
        return 'Moving2T'

def Moving2Texclude(MAerror,anomalies, factor, hscaleCount=1000):
    """
    This method excludes preciously found anomalies before apply Moving2T technique.

    :param MAerror: all anomaly scores until now.
    :param anomalies:  a boolean array (with size equal or smaller than MAerror) which indicate if an instance is anomaly or no.
    :param factor: the multiplier factor for standard deviation to calculate threshold.
    :param hscaleCount: the number of historical values to take in to consideration in threshold calculation.
    :return: Boolean (is anomaly) and the threshold value
    """
    withoutAnomalies = [error for error, isanomaly in zip(MAerror[:len(anomalies)], anomalies) if isanomaly == False or isanomaly==0]
    withoutAnomalies.extend(MAerror[len(anomalies):])
    return Moving2T(withoutAnomalies, factor, hscaleCount=hscaleCount)


def Moving2T(MAerror, factor, hscaleCount=1000):
    """
    Based on the historic anomaly score calculate the standard deviation and mean of scores to derive in new threshold.

    :param MAerror: all anomaly scores until now.
    :param factor: the multiplier factor for standard deviation to calculate threshold.
    :param hscaleCount: the number of historical values to take in to consideration in threshold calculation.
    :return: Boolean (is anomaly) and the threshold value
    """
    if hscaleCount is None:
        hscaleCount = len(MAerror)
    historyerrors_raw = MAerror[max(0, len(MAerror) - hscaleCount):]

    if len(historyerrors_raw) == 1:
        return False,historyerrors_raw[-1]
    historyerrors=[historyerrors_raw[0]]
    for q in historyerrors_raw[1:]:
        if q==historyerrors[-1]:
            continue
        historyerrors.append(q)

    if len(historyerrors) == 1:
        return False,historyerrors[-1]

    th = statistics.mean(historyerrors) + factor * statistics.stdev(historyerrors)
    secondpass=[ d for d in historyerrors if d<th]
    if len(secondpass) == 0:
        return False, historyerrors[-1]
    fianal_threshold= statistics.mean(secondpass) + factor * statistics.stdev(secondpass)
    return MAerror[-1]>fianal_threshold, fianal_threshold