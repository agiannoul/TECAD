import collections

import pandas as pd
import mlflow
from matplotlib import pyplot as plt

from  method.semi_supervised_method import SemiSupervisedMethodInterface
from  pdm_evaluation_types.types import EventPreferences



class Classifier_feed_raw_events(SemiSupervisedMethodInterface):
    def __init__(self, event_preferences: EventPreferences,
                 PHhorizon,all_event_names,cross_source=False,clasifier="xgb",
                 *args, **kwargs):
        super().__init__(event_preferences=event_preferences)
        self.initial_args = args
        self.initial_kwargs = kwargs
        self.event_preferences = event_preferences
        self.PHhorizon = PHhorizon
        self.cross_source=cross_source
        self.all_event_names=all_event_names
        self.clasifier=clasifier

        self.model_per_source = {}
        self.keepfirst=None

    def fit(self, historic_data: list[pd.DataFrame], historic_sources: list[str], event_data: pd.DataFrame) -> None:
        for current_historic_data, current_historic_source in zip(historic_data, historic_sources):
            if current_historic_source not in self.model_per_source:
                self.model_per_source[current_historic_source]=classifier_inner(self.event_preferences,self.PHhorizon,self.all_event_names,self.clasifier)
            if self.cross_source:
                if self.keepfirst is None:
                    self.keepfirst=current_historic_source
                    current_event_data = event_data[event_data["source"] == current_historic_source]
                    self.model_per_source[current_historic_source].fit([current_historic_data],
                                                                       [current_historic_source], current_event_data)
                else:
                    current_event_data = event_data[event_data["source"] == current_historic_source]
                    self.model_per_source[self.keepfirst].fit([current_historic_data],[current_historic_source], current_event_data)

            else:
                current_event_data=event_data[event_data["source"]==current_historic_source]
                self.model_per_source[current_historic_source].fit([current_historic_data],[current_historic_source],current_event_data)

    def predict(self, target_data: pd.DataFrame, source: str, event_data: pd.DataFrame) -> list[float]:
        current_event_data = event_data[event_data["source"] == source]
        if self.cross_source:
            return self.model_per_source[self.keepfirst].predict(target_data,source, current_event_data)
        else:
            return self.model_per_source[source].predict(target_data,source,current_event_data)


    def predict_one(self, new_sample: pd.Series, source: str, is_event: bool) -> float:
        # TODO need to keep buffer until profile size are encountered and then start predicting
        return -self.model_per_source[source].score_samples([new_sample.to_numpy()]).tolist()[0]

    def get_library(self) -> str:
        return 'no_save'

    def __str__(self) -> str:
        return 'Ensembler'

    def get_params(self) -> dict:
        return {
            'PHhorizon':self.PHhorizon,
            'cross_source':self.cross_source,
            'clasifier':self.clasifier,
        }

    def get_all_models(self):
        pass


class classifier_inner():
    def __init__(self, event_preferences: EventPreferences,PHhorizon,all_event_names,clasifier,
                 *args, **kwargs):
        self.event_preferences=event_preferences
        self.initial_args = args
        self.initial_kwargs = kwargs
        self.clasifier = clasifier

        self.en_model=None
        self.PHhorizon=PHhorizon

        self.buffer_data={}
        self.clear_buffer()
        self.all_event_names=all_event_names

    def fit(self, historic_data: list[pd.DataFrame], historic_sources: list[str], event_data: pd.DataFrame) -> None:
        pass

    def predict(self, target_data: pd.DataFrame, source: str, event_data: pd.DataFrame) -> list[float]:
        pos=0
        alladata=[]
        event_data=event_data[event_data["date"] > target_data.index[0]]
        for i in range(target_data.shape[0]):
            #temp=[ dd for dd in target_data.iloc[i]] +[0 for ev in self.all_event_names]
            temp=[0 for ev in self.all_event_names]
            if pos<event_data.shape[0]:
                while event_data.iloc[pos]['date']<=target_data.index[i]:
                    if event_data.iloc[pos]['date']>target_data.index[0] and event_data.iloc[pos]['description'] in self.all_event_names:
                        temp[self.all_event_names.index(event_data.iloc[pos]['description'])]=1
                        pos+=1
                        if pos>=event_data.shape[0]:
                            break
            alladata.append(temp)
        inner_Data=alladata


        self.check_failure_event_in_start(target_data,event_data,source)

        if self.en_model is None:
            if self.clasifier=="xgb":
                self.en_model = XGB2FEED(plot=False)
            elif self.clasifier=="svc":
                self.en_model = SVMFEED(plot=False)
            en_score = self.en_model.predict(inner_Data)
        else:
            en_score = self.en_model.predict(inner_Data)
            en_score = [sc for sc in en_score]

        self.buffer_data["en_score"].extend(en_score)
        self.buffer_data["dates"].extend([dt for dt in target_data.index])
        self.buffer_data["data"].extend(inner_Data)

        print("Scores calculated")

        self.check_failure_event_in_midle(source,event_data,target_data)

        return en_score

    def check_failure_event_in_start(self,source,event_data,target_data):

        failures=extract_failure_dates_for_source(source,event_data,self.event_preferences)
        earlier_failure=None
        for fail in failures:
            if fail<=target_data.index[0] and fail>self.buffer_data['dates'][-1]:
                earlier_failure=fail
        if earlier_failure is None:
            return
        labels=[]
        for ind in self.buffer_data["dates"]:
            if ind > earlier_failure - pd.Timedelta(int(self.PHhorizon.split(" ")[0]), self.PHhorizon.split(" ")[1]) and ind<=earlier_failure:
                labels.append(1)
            else:
                labels.append(0)
        self.train_model_feedback(labels)

    def check_failure_event_in_midle(self,source,event_data,target_data):
        labels=[]
        if isinstance(self.PHhorizon, collections.abc.Sequence) and not isinstance(self.PHhorizon, str):
            allalabs = self.PHhorizon[0]
            labels = allalabs[- len(self.buffer_data["dates"]):]
        else:
            failures=extract_failure_dates_for_source(source,event_data,self.event_preferences)
            tempfails=[]
            for fail in failures:
                if fail>target_data.index[0] and fail<=self.buffer_data['dates'][-1]:
                    tempfails.append(fail)
            if len(tempfails) == 0:
                return
            labels=[0 for id in self.buffer_data["dates"]]

            for failt in tempfails:
                for i,ind in enumerate(self.buffer_data["dates"]):
                    if ind > failt - pd.Timedelta(int(self.PHhorizon.split(" ")[0]), self.PHhorizon.split(" ")[1]) and ind<=failt:
                        labels[i]=1

        self.train_model_feedback(labels)


    def train_model_feedback(self,labels):
        print(f"Training: {len(labels)}, {len(self.buffer_data['data'])}")
        self.en_model.update_wiehgts_detector(labels,self.buffer_data["data"])
        self.clear_buffer()
        print("Done Training")
    def clear_buffer(self):
        self.buffer_data["en_score"]=[]
        self.buffer_data["dates"]=[]
        self.buffer_data["data"] = []


from utils.utils import expand_event_preferences

def extract_failure_dates_for_source(source,event_data,event_preferences) -> list[pd.Timestamp]:
    result = []
    expanded_event_preferences = expand_event_preferences(event_data=event_data, event_preferences=event_preferences)
    for current_preference in expanded_event_preferences['failure']:
        matched_rows = event_data.loc[(event_data['type'] == current_preference.type) & (event_data['source'] == current_preference.source) & (event_data['description'] == current_preference.description)]
        for row_index, row in matched_rows.iterrows():
            if current_preference.target_sources == '=' and str(row.source) == str(source):
                result.append(row['date'])
            elif str(source) in str(current_preference.target_sources):
                result.append(row['date'])
            elif current_preference.target_sources == '*':
                result.append(row['date'])
    return sorted(list(set(result)))


import numpy as np
import xgboost as xgb

import abc

class XGBFEED():

    def __init__(self,plot=False,all_names=[]):
        self.plotit=plot
        self.model=None
        self.Not_fitted=True

    def predict(self,data):# last values
        if self.Not_fitted:
            self.model = xgb.XGBClassifier(objective='binary:logistic')
            return [0 for sc in range(len(data))]
        else:
            en_score=self.model.predict(np.array(data))
            return en_score

    def update_wiehgts_detector(self, labels,data,lr=0.0001):

        vectors = np.array(data)
        if self.Not_fitted:
            self.model = xgb.XGBRegressor()
            self.model = self.model.fit(vectors, labels[len(labels) - len(vectors):])
            self.Not_fitted = False
        else:
            self.model = self.model.fit(vectors, labels[len(labels) - len(vectors):],
                                        xgb_model=self.model.get_booster())
        return


from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from imblearn.under_sampling import ClusterCentroids
from sklearn.cluster import MiniBatchKMeans

class SVMFEED():

    def __init__(self,plot=False):
        self.plot=plot
        self.model = None
        self.inner_buffer_zero=[]
        self.inner_buffer_ones=[]
        self.Not_fitted=True
        self.preX0= None

    def predict(self,data):# last values

        # Show the plot
        if self.Not_fitted:
            self.model = make_pipeline(StandardScaler(), SVC(gamma='auto',probability=True))
            return [0 for sc in data]
        else:
            probs=self.model.predict_proba(np.array(data))
            en_score=[]
            for pb in probs:
                en_score.append(pb[1])
            return en_score

    def add_data_to_buffer(self,all_vectors,labels):
        for vector,lab in zip(all_vectors,labels):
            if lab==1:
                self.inner_buffer_ones.append(vector)
            else:
                self.inner_buffer_zero.append(vector)

    def resample_contexts(self, X_1, X_0):
        import random
        random.seed(123)
        if len(X_1) < 1000:
            rX1 = X_1 + random.choices(X_1, k=1000 - len(X_1))  # Sampling with replacement
        else:
            rX1 = random.sample(X_1, 1000)

        if len(X_0) < 1000:
            rX0 = X_0 + random.choices(X_0, k=1000 - len(X_0))  # Sampling with replacement
        else:
            if self.preX0 is not None:
                if len(X_0) - self.preX0 > 1000:
                    rX0 = random.sample(X_0[self.preX0:], 1000)
                else:
                    rX0 = X_0[self.preX0:]
                    rX0 = rX0 + random.sample(X_0[:self.preX0], 1000 - len(rX0))
            else:
                rX0 = random.sample(X_0, 1000)
        self.preX0 = None
        return rX1, rX0

    def get_train_data(self):
        def shuffle_lists(Xres, Yres):
            import random
            random.seed(123)
            """
            Shuffles Xres and Yres while keeping their relationships.
            """
            combined = list(zip(Xres, Yres))
            random.shuffle(combined)
            Xres_shuffled, Yres_shuffled = zip(*combined)
            return list(Xres_shuffled), list(Yres_shuffled)

        res1, res0 = self.resample_contexts(self.inner_buffer_ones, self.inner_buffer_zero)
        X_res = res1 + res0
        y_res = [1 for i in res1] + [0 for i in res0]

        return shuffle_lists(X_res, y_res)

    def update_wiehgts_detector(self, labels,data,lr=0.0001):

        all_vectors= data

        if sum(labels)==len(labels):
            newlabels=[0 if i<len(labels)/2 else 1 for i in range(len(labels))]
        else:
            newlabels=labels

        self.add_data_to_buffer(all_vectors,newlabels)
        all_vectors,labels=self.get_train_data()


        data=all_vectors
        label = labels[-len(all_vectors):]

        self.model = make_pipeline(StandardScaler(), SVC(gamma='auto',probability=True))

        self.model.fit(np.array(data), label)
        self.Not_fitted = False

        return

from xgboost import XGBClassifier

class XGB2FEED():

    def __init__(self, plot=False):
        self.plot = plot
        self.model = None
        self.inner_buffer_zero = []
        self.inner_buffer_ones = []
        self.Not_fitted = True
        self.preX0=None
    def predict(self, data):  # last values

        # Show the plot
        if self.Not_fitted:
            return [0 for sc in data]
        else:
            probs = self.model.predict_proba(np.array(data))
            en_score = []
            for pb in probs:
                en_score.append(pb[1])
            return en_score

    def add_data_to_buffer(self, all_vectors, labels):
        for vector, lab in zip(all_vectors, labels):
            if lab == 1:
                self.inner_buffer_ones.append(vector)
            else:
                self.inner_buffer_zero.append(vector)

    def resample_contexts(self, X_1, X_0):
        import random
        random.seed(123)
        if len(X_1) < 1000:
            rX1 = X_1 + random.choices(X_1, k=1000 - len(X_1))  # Sampling with replacement
        else:
            rX1 = random.sample(X_1, 1000)

        if len(X_0) < 1000:
            rX0 = X_0 + random.choices(X_0, k=1000 - len(X_0))  # Sampling with replacement
        else:
            if self.preX0 is not None:
                if len(X_0) - self.preX0 > 1000:
                    rX0 = random.sample(X_0[self.preX0:], 1000)
                else:
                    rX0 = X_0[self.preX0:]
                    rX0 = rX0 + random.sample(X_0[:self.preX0], 1000 - len(rX0))
            else:
                rX0 = random.sample(X_0, 1000)
        self.preX0 = None
        return rX1, rX0

    def get_train_data(self):
        def shuffle_lists(Xres, Yres):
            import random
            random.seed(123)
            """
            Shuffles Xres and Yres while keeping their relationships.
            """
            combined = list(zip(Xres, Yres))
            random.shuffle(combined)
            Xres_shuffled, Yres_shuffled = zip(*combined)
            return list(Xres_shuffled), list(Yres_shuffled)

        res1, res0 = self.resample_contexts(self.inner_buffer_ones, self.inner_buffer_zero)
        X_res = res1 + res0
        y_res = [1 for i in res1] + [0 for i in res0]

        return shuffle_lists(X_res, y_res)

    def update_wiehgts_detector(self, labels, data, lr=0.0001):

        all_vectors = data
        if sum(labels)==len(labels):
            newlabels=[0 if i<len(labels)/2 else 1 for i in range(len(labels))]
        else:
            newlabels=labels

        self.add_data_to_buffer(all_vectors, newlabels)
        all_vectors, labels = self.get_train_data()

        if sum(labels) == len(labels):
            newlabels = [0 if i < len(labels) / 2 else 1 for i in range(len(labels))]
        else:
            newlabels = labels

        data = all_vectors
        label = newlabels[-len(all_vectors):]

        if self.model is not None:
            self.model = make_pipeline(StandardScaler(),
                                       XGBClassifier(use_label_encoder=False, eval_metric='logloss', seed=0,
                                                     xgb_model=self.model['xgbclassifier'].get_booster()))
        else:
            self.model = make_pipeline(StandardScaler(),
                                       XGBClassifier(use_label_encoder=False, eval_metric='logloss', seed=0))

        self.model.fit(np.array(data), label)
        self.Not_fitted = False

        return