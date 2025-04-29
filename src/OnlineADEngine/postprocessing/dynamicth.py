import statistics

import numpy as np
import pandas as pd
from operator import itemgetter
import datetime
from tqdm import tqdm
from postprocessing.post_processor import PostProcessorInterface
from pdm_evaluation_types.types import EventPreferences


class DynamicThresholder(PostProcessorInterface):
    def __init__(self, event_preferences: EventPreferences, epsilon: float = 0.05,history_window=None):
        super().__init__(event_preferences=event_preferences)
        self.epsilon = epsilon
        self.history_window = history_window
        self.alldata = False
        if self.history_window is None:
            self.history_window=1
            self.alldata=True

        self.anomaly_scores_dict={}


    def transform(self, scores: list[float], source: str, event_data: pd.DataFrame):
        self.anomaly_scores_dict[source]=[]
        new_scores=[]
        for qi in range(len(scores)):
            sc=scores[qi]
            self.anomaly_scores_dict[source].append(sc)
            succed, th = dynamicThresholding(self.anomaly_scores_dict[source], DesentThreshold=self.epsilon,
                                             hscaleCount=self.history_window,
                                             alldata=self.alldata)
            if succed == False:
                new_scores.append(0)
            else:
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
        succed,th=dynamicThresholding(self.anomaly_scores_dict[source], DesentThreshold=self.epsilon, hscaleCount=self.history_window,
                            alldata=self.alldata)
        if succed==False:
            return 0
        else:
            if score_point>th:
                return 1
            else:
                return 0

    def get_params(self):
        return {
            'epsilon': self.epsilon,
            'history_window':self.history_window,
            'All data in history':self.alldata
        }

    def __str__(self) -> str:
        return 'DynamicThresholder'

def dynamicThresholding(MAerror, DesentThreshold=0.02, hscaleCount=1000, alldata=False):
    """
    Re-Implementation of dynamic thresholding from : Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding

    This method is used to calculate the threshold only for the last sample of Anomaly scores, based on the calculated anomaly scores so far.

    Note: To calculate the threshold for each anomaly score this method needs to used iteratively.

    :param MAerror: Anomaly Scores
    :param DesentThreshold: Parameter to prune anomalies when their percentage difference from threshold is lower than this.
    :param hscaleCount: Historic anomaly scores samples to consider (History window)
    :param alldata: True in case we want to consider as historic samples, all anomaly scores so far.
    :return: False in case it couldn't produce a threshold, else True and the threshold value
    """
    normalization_in_error = False
    # start_time = time.time()


    if alldata == True:
        historyerrors_raw = MAerror
    else:
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

    error = historyerrors[-1]
    # =======================================
    # ======= define parameters of threshold calculation ===================
    z = [v / 6 for v in range(18, 30)]  # z vector for threshold calculation

    diviation = statistics.stdev(historyerrors)  # diviation of errors
    meso = statistics.mean(historyerrors)  # mean of errors
    e = [meso + (element * diviation) for element in z]  # e: set of candidate thresholds

    maximazation_value = []
    maxvalue = -1
    thfinal = e[0]
    maxEA = []
    # ============ threshold calculation ========================
    for th in e:
        EA = []  # List of sequence of anomalous errors
        ea = [(i, distt) for i, distt in enumerate(historyerrors) if distt > th]  # dataframe of anomaly errors

        # if ea equals to 0 that means no anomalies so the Δμ/μ and Δσ/σ also are equal to zero
        if len(ea) == 0:
            continue
        if len([element for element in historyerrors if element < th]) <= 1:
            continue
        # Δμ -> difference betwen mean of errors and mean of errors without anomalies
        dmes = meso - statistics.mean([element for element in historyerrors if element < th])
        # Δσ ->  difference betwen diviation of errors and diviation of errors without anomalies
        ddiv = diviation - statistics.stdev([element for element in historyerrors if element < th])

        # ========= group anomaly error in sequences================
        # ea= [ (position, dist/error) , ... , (position, dist/error)]
        posi = ea[0][0]
        while posi <= ea[-1][0]:
            sub = []

            tempea = [tupls for tupls in ea if tupls[0] >= posi]
            sub.append(tempea[0])
            # store all continues errors (in index) in same subssequence
            for row in tempea[1:]:
                # if index of error is the last index of subsequence plus 1 then error is part of this sequence
                if row[0] == sub[-1][0] + 1:
                    sub.append(row)
                    posi = row[0] + 1
                else:
                    posi = row[0]
                    break
            # add the subsequence in to the list
            EA.append(sub)
            if len(tempea[1:]) == 0:
                break

        # ================ persentage impact of the threshold =================
        argmaxError = (dmes / meso + ddiv / diviation) / (
                    len(ea) + len(EA) * len(EA))  # calculate value of formula which we try to maximize
        if maxvalue < argmaxError:
            maxvalue = argmaxError
            thfinal = th
            maxEA = EA
        maximazation_value.append(argmaxError)
    if len(maxEA) == 0:
        return False, thfinal

    if error > thfinal:
        # ==================look for prunning===========================
        # if last value belongs to anomalies then i will be a part of last anomaly sequence
        notea = [err for err in historyerrors if err <= thfinal]
        normalmax = max(notea)

        # maxEA = maxEA[:-1]
        lastSeq = maxEA[-1]
        maxlastSeq = max(lastSeq, key=itemgetter(1))
        maxErrorEA = [max(seq, key=itemgetter(1)) for seq in maxEA]
        maxErrorEA.append((-1, normalmax))
        minhistory = 0
        if normalization_in_error == True:
            minhistory = min(historyerrors)

        maxlastSeq = (maxlastSeq[0], maxlastSeq[1] - (minhistory - minhistory / 100.0))

        sortedmax = sorted(maxErrorEA, key=lambda x: x[1], reverse=True)

        checkpoint = -1
        count = -1
        for tup1, tup2 in zip(sortedmax[:-1], sortedmax[1:]):
            count += 1
            diff = (tup1[1] - tup2[1]) / tup1[1]
            if diff > DesentThreshold:
                checkpoint = count
        if checkpoint != -1:
            realAnomalies = sortedmax[:checkpoint + 1]
            if maxlastSeq[0] in list(map(list, zip(*realAnomalies)))[0]:
                return True, thfinal
    return False, thfinal