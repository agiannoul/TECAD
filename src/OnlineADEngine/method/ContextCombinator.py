import collections
import os
import pickle
import statistics

import matplotlib.pyplot as plt
import pandas as pd

from method.semi_supervised_method import SemiSupervisedMethodInterface
from pdm_evaluation_types.types import EventPreferences
from PdmContext.ContextGeneration import ContextGenerator
from PdmContext.utils.causal_discovery_functions import calculate_with_pc
from PdmContext.utils.simulate_stream import simulate_stream
from tqdm import tqdm


# Active learning models:
from method.ActiveLearningModels.XGBoost import XGB2FEED
from method.ActiveLearningModels.SVC import SVMFEED
from method.ActiveLearningModels.Base import Gen_FEED

DEBUG=True

random_state=24
class ContextCombinator(SemiSupervisedMethodInterface):
    """
    This method is used to combine anomaly detection methods (like ocsvm,isolation forest  etc) with context data using the context generation model.
    Essentially this class instantiate the CDT component of the paper. For the classifier to process context XGBoost or Support Vector Classifier are used.

    """
    def __init__(self, event_preferences: EventPreferences,
                 names, classes,paramlist,all_event_names,
                 context_horizon,PHhorizon,add_data_to_context=False,cross_source=False,
                 threshold_per_method=[],include_scores_in_model=False,classifier="xgb",
                 detector_weight=None,retrain_anomaly_detector=False,dimensonality_red=0
                 ,*args, **kwargs):
        """

        :param event_preferences: This is passed from framework and contains the information of what type of events we except.
        :param names: The name of selected method for Anomaly Detection
        :param classes: Class reference to Anomaly Detection model (that implements fit and predict functions)
        :param paramlist: Parameters of Anomaly Detector in form of dictionary.
        :param all_event_names: The name of events to monitor for generating Context
        :param context_horizon: The time window of the context.
        :param PHhorizon: This contains the information of ground truth which will simulate Feedback.
         1) We can do this o by defining events of interest in event-preferences (in loadDataset script) and pass a timespan
         in form of string (e.g. "8 hours") to highlight the period before the event that we considered as anomalous
         (used for Predictive Maintenance case study).
         2) or to pass a tuple with parallel lists of labels and their timestamp in pandas.datetime format
          (e.g. ([0,0,0,0,1,1,0,1],[2000-01-01,2000-01-02,...])
        :param add_data_to_context: Pass list annotating the features from those that are monitored from Anomaly Detector,
        that we want to also fed in Context.
        :param cross_source: Weather we want to use the same classifier or no for different cases (i.e. different machines in Azure Dataset)
        :param threshold_per_method: Provide the threshold for anomaly score in order to produce binary input (anomaly or not)
        :param include_scores_in_model: Whether we want to include also the result of anomaly detection in classifier
        :param classifier: Which classifier to use (xgb or svc)
        :param detector_weight: Whether we want learnable weights for CDT model (NONE) or fixed by passing the weight of anomaly detector,
        while the wiehgt of classifier is calculated as 1-anomalyDetectorWeight.
        :param args:
        :param kwargs:
        """
        super().__init__(event_preferences=event_preferences)
        self.initial_args = args
        self.initial_kwargs = kwargs
        self.event_preferences = event_preferences
        self.PHhorizon = PHhorizon
        self.cross_source=cross_source
        self.all_event_names=all_event_names
        self.classes=classes
        self.paramlist=paramlist
        self.classifier=classifier
        self. names= names
        self.method_list=[]
        self.add_data_to_context=add_data_to_context
        self.context_horizon=context_horizon
        self.threshold_per_method=threshold_per_method
        self.include_scores_in_model=include_scores_in_model
        self.fill_empty=False
        self.model_per_source = {}
        self.keepfirst=None
        self.detector_weight=detector_weight
        self.retrain_anomaly_detector=retrain_anomaly_detector
        self.dimensonality_red=dimensonality_red
    def fit(self, historic_data: list[pd.DataFrame], historic_sources: list[str], event_data: pd.DataFrame) -> None:
        for current_historic_data, current_historic_source in zip(historic_data, historic_sources):
            if current_historic_source not in self.model_per_source:
                self.model_per_source[current_historic_source]=Classifier_inner(self.event_preferences,
                     self.names, self.classes,self.paramlist,self.all_event_names,
                     self.context_horizon,self.PHhorizon,add_data_to_context=self.add_data_to_context,threshold_per_method=self.threshold_per_method,
                     include_scores_in_model=self.include_scores_in_model,classifier=self.classifier,
                                        fill_empty=self.fill_empty,detector_weight=self.detector_weight
                                        ,retrain_anomaly_detector=self.retrain_anomaly_detector,dimensonality_red=self.dimensonality_red)
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
            'names':self. names,
            'add_data_to_context':self. add_data_to_context,
            'context_horizon':self.context_horizon,
            'cross_source':self.cross_source,
            'PHhorizon':self.PHhorizon,
            'all_event_names':self.all_event_names,
            'include_scores_in_model':self.include_scores_in_model,
            'threshold_per_method':self.threshold_per_method,
            'classifier':self.classifier,
            'fill_empty':self.fill_empty,
            'detector_weight':self.detector_weight,
            'retrain_anomaly_detector':self.retrain_anomaly_detector,
            'dimensonality_red':self.dimensonality_red,
        }

    def get_all_models(self):
        pass


class Classifier_inner():
    def __init__(self, event_preferences: EventPreferences,
                 names, classes,paramlist,all_event_names,
                 context_horizon,PHhorizon,add_data_to_context=False,
                 threshold_per_method=[],include_scores_in_model=False,classifier="xgb",fill_empty=False,
                 detector_weight=None,retrain_anomaly_detector=False,dimensonality_red=0,*args, **kwargs):
        self.event_preferences=event_preferences

        self.initial_args = args
        self.initial_kwargs = kwargs
        self.all_event_names=all_event_names
        self.model_per_source = {}
        self.initial=True
        self. names= names
        self.method_list=[]
        self.add_data_to_context=add_data_to_context
        self.context_horizon=context_horizon
        self.PHhorizon=PHhorizon
        self.classifier=classifier
        self.en_model=None
        self.threshold_per_method=threshold_per_method
        self.include_scores_in_model=include_scores_in_model
        self.fill_empty=fill_empty
        self.classes=classes
        self.paramlist=paramlist
        for method,method_params in zip(classes,paramlist):
            self.method_list.append(method(event_preferences=event_preferences, **method_params))
        self.sources=[]
        self.calculate_contex=True
        if self.calculate_contex:
            self.context_sources = {}

        self.buffer_data={}
        self.clear_buffer()

        self.context_names = [f"d_0"]
        self.context_names.extend([evname for evname in all_event_names])

        self.presource=None

        self.retrain_anomaly_detector=retrain_anomaly_detector
        self.globalprof=None
        self.currentprof=None

        self.history_C=[]
        self.detector_weight=detector_weight
        self.history_D = [1]
        self.detector_w=1
        self.context_w=0
        self.contextsall=[]

        self.dimensonality_red=dimensonality_red
    def fit(self, historic_data: list[pd.DataFrame], historic_sources: list[str], event_data: pd.DataFrame) -> None:
        for current_historic_data, current_historic_source in zip(historic_data, historic_sources):
            for i in range(len(self.method_list)):
                if current_historic_data.shape[0]<=3:
                    self.currentprof=current_historic_data.shape[0]
                    continue
                self.method_list[i].fit([current_historic_data],[current_historic_source],event_data)
                self.globalprof=current_historic_data.shape[0]
    def predict(self, target_data: pd.DataFrame, source: str, event_data: pd.DataFrame) -> list[float]:
        if source not in self.sources:
            self.sources.append(source)
        self.check_failure_event_in_start(target_data,event_data,source)
        scores=[]
        for i in range(len(self.method_list)):
            toappend=0
            topredic=target_data
            if self.retrain_anomaly_detector and self.currentprof != self.globalprof:
                # self.method_list=[]
                # for method, method_params in zip(self.classes, self.paramlist):
                #     self.method_list.append(method(event_preferences=self.event_preferences, **method_params))
                if self.globalprof>target_data.shape[0]:
                    self.method_list[i].fit([target_data.iloc[:target_data.shape[0]//2]], [source], event_data)
                    toappend=target_data.iloc[:target_data.shape[0]//2].shape[0]
                    topredic=target_data.iloc[target_data.shape[0]//2:]
                else:
                    toappend = target_data.iloc[:self.globalprof].shape[0]
                    topredic = target_data.iloc[self.globalprof:]
                    self.method_list[i].fit([target_data.iloc[:self.globalprof]], [source], event_data)
            ith_scores_pre=self.method_list[i].predict(topredic, source, event_data)
            ith_scores=[min(ith_scores_pre) for qii in range(toappend)]+ith_scores_pre
            if len(self.threshold_per_method)>0:
                scores.append([1 if sc>self.threshold_per_method[i] else 0 for sc in ith_scores])
                #scores.append(ith_scores)
            else:
                scores.append(ith_scores)

        event_data_split=event_data[event_data['date']>=target_data.index[0]]
        event_data_split=event_data_split[event_data_split['date']<=target_data.index[-1]]

        exist,contexts=self.check_existing_context(scores,target_data,event_data_split,source)


        #contexts[-1].plotCD()
        #plt.savefig("cd.png")
        #plt.clf()
        #contexts[-1].plotRD()
        #plt.savefig("cr.png")
        # plt.clf()
        # print("Context calculated")

        timedelta = self.context_horizon.split(" ")
        if DEBUG:
            print(self.context_names)
        if self.en_model is None:
            for key in contexts[-1].CD.keys():
                if contexts[-1].CD[key] is not None:
                    size = len(contexts[-1].CD[key])
                    size=size+4-size%4
                    break
            self.en_model = self.get_classifier()
            en_score = self.en_model.predict(contexts,scores)

        else:
            en_score = self.en_model.predict(contexts,scores)
            en_score = [sc for sc in en_score]




        self.buffer_data["en_score"].extend(en_score)
        self.buffer_data["contexts"].extend(contexts)
        n_Scores=len(self.names)
        # if len(self.threshold_per_method)>0:
        #     n_Scores+=1
        for q in range(n_Scores):
            self.buffer_data["scores"][q].extend(scores[q])
        self.buffer_data["dates"].extend([dt for dt in target_data.index])
        if self.buffer_data["data"] is None:
            self.buffer_data["data"]=target_data
        else:
            self.buffer_data["data"]=pd.concat([self.buffer_data["data"],target_data])

        if DEBUG:
            print("Scores calculated")

        # This simulates the feedback from user.
        self.check_failure_event_in_midle(source,event_data,target_data)

        f_score=[d_s*self.detector_w+c_s*self.context_w for d_s,c_s in zip(scores[0],en_score)]

        return f_score

    def check_existing_context(self,scores,target_data,event_data_split,source):
        tempname=self.names[0]
        n=len(target_data.index)
        firstime=target_data.index[0]
        lasttime=target_data.index[0]
        namefile=f"{tempname}_{firstime}_{lasttime}_{n}_{source}_context.pickle"
        file_path = f"CacheData/{namefile}"
        if os.path.isfile(file_path):
            if self.initial:
                self._calculate_fields(scores, target_data, event_data_split, source,
                                             add_data_to_context=self.add_data_to_context)
                self.initial=False
            with open(file_path, "rb") as f:
                data = pickle.load(f)
            return True,data
        else:
            contexts = self._produce_context(scores, target_data, event_data_split, source,
                                             add_data_to_context=self.add_data_to_context)
            with open(file_path, "wb") as f:
                pickle.dump(contexts, f)
            return False,contexts

    def get_classifier(self):
        if self.classifier=="xgb":
            return XGB2FEED(self.context_names,plot=False,include_scores_in_model=self.include_scores_in_model,dimensonality_red=self.dimensonality_red)
        elif self.classifier=="svc":
            return SVMFEED(self.context_names,plot=False,include_scores_in_model=self.include_scores_in_model,dimensonality_red=self.dimensonality_red)
        return None
    def runToFail_train(self):
        labels = []
        if isinstance(self.PHhorizon, collections.abc.Sequence):

            allalabs = self.PHhorizon[0]
            labels = allalabs[- len(self.buffer_data["dates"]):]
        elif "regression" in self.PHhorizon:
            lendates=len(self.buffer_data["dates"])
            for i,ind in enumerate(self.buffer_data["dates"]):
                labels.append(i/lendates)
        else:
            for ind in self.buffer_data["dates"]:
                if ind > self.buffer_data["dates"][-1] - pd.Timedelta(int(self.PHhorizon.split(" ")[0]),self.PHhorizon.split(" ")[1]):
                    labels.append(1)
                else:
                    labels.append(0)
        self.train_model_feedback(labels)

    def check_failure_event_in_start(self,source,event_data,target_data):

        failures=extract_failure_dates_for_source(source,event_data,self.event_preferences)
        earlier_failure=None
        for fail in failures:
            if fail<=target_data.index[0] and fail>self.buffer_data['dates'][-1]:
                earlier_failure=fail
        if earlier_failure is None:
            return
        labels=[]
        if isinstance(self.PHhorizon, collections.abc.Sequence):
            allalabs = self.PHhorizon[0]
            alltiems = self.PHhorizon[1]
            posisiton = alltiems.index(earlier_failure)
            labels = allalabs[posisiton - len(self.buffer_data["dates"]):posisiton]
        elif "regression" in self.PHhorizon:
            if "regression" in self.PHhorizon:
                labels = self.labels_for_regrssion([earlier_failure])
        else:
            for ind in self.buffer_data["dates"]:
                if ind > earlier_failure - pd.Timedelta(int(self.PHhorizon.split(" ")[0]), self.PHhorizon.split(" ")[1]) and ind<=earlier_failure:
                    labels.append(1)
                else:
                    labels.append(0)
        self.train_model_feedback(labels)

    def check_failure_event_in_midle(self,source,event_data,target_data):

        failures=extract_failure_dates_for_source(source,event_data,self.event_preferences)
        tempfails=[]
        for fail in failures:
            if fail>target_data.index[0] and fail<=self.buffer_data['dates'][-1]:
                tempfails.append(fail)
        if len(tempfails) == 0:
            return

        if isinstance(self.PHhorizon, collections.abc.Sequence) and not isinstance(self.PHhorizon, str):
            allalabs=self.PHhorizon[0][self.sources.index(source)]
            alltiems=self.PHhorizon[1][self.sources.index(source)]
            posisiton=alltiems.index(tempfails[-1])
            labels = allalabs[posisiton-len(self.buffer_data["dates"]):posisiton]

        elif "regression" in self.PHhorizon:
            labels=self.labels_for_regrssion(tempfails)
        else:
            labels=[0 for id in self.buffer_data["dates"]]
            for failt in tempfails:
                for i,ind in enumerate(self.buffer_data["dates"]):
                    if ind > failt - pd.Timedelta(int(self.PHhorizon.split(" ")[0]), self.PHhorizon.split(" ")[1]) and ind<=failt:
                        labels[i]=1

        self.train_model_feedback(labels)

    def labels_for_regrssion(self,tempfails):
        labels = []
        tempfails = sorted(tempfails)
        counterfail = 0
        failt = tempfails[counterfail]
        templabels = []
        pos_stop = 0
        for i, ind in enumerate(self.buffer_data["dates"]):
            pos_stop = i
            if ind > failt - pd.Timedelta(int(self.PHhorizon.split(" ")[0]),
                                          self.PHhorizon.split(" ")[1]) and ind <= failt:
                templabels.append(i)
            elif ind > failt:
                lentemp = len(templabels)
                templabels = [ll / lentemp for ll in templabels]
                labels.extend(templabels)
                labels.append(0)
                counterfail += 1
                if counterfail >= len(tempfails):
                    break
                failt = tempfails[counterfail]
        labels.extend([0 for i in range(len(self.buffer_data["dates"]) - len(labels))])
        return labels

    def unnorm(self,a,b):
        return [a/(a+b),b/(a+b)]

    def train_model_feedback(self,labels):
        if len(labels) == 0:
            return
        # print("Training:")
        loss_c=self.en_model.update_wiehgts_detector( labels,self.buffer_data["contexts"],self.buffer_data["scores"],loss_func=loss_calculation)
        if self.detector_weight is None:
            self.history_C.append(loss_c)
            loss_c=statistics.mean(self.history_C)
            loss_d=loss_calculation(self.buffer_data["scores"][0],labels)
            self.history_D.append(loss_d)
            loss_d = statistics.mean(self.history_D)
            # from scipy.special import softmax
            #
            # m = softmax([3*loss_d,3*loss_c])
            m = self.unnorm(loss_d,loss_c)
            self.detector_w=m[0]
            self.context_w=m[1]
            if DEBUG:
                print("-------------------------------------------")
                print(f"loss D:{loss_d} \t loss C:{loss_c}")
                print(f"w D:{self.detector_w} \t w C:{self.context_w}")
                print("-------------------------------------------")
        else:
            self.detector_w = self.detector_weight
            self.context_w = 1-self.detector_weight
            if DEBUG:
                print("-------------------------------------------")
                print(f"fix weights, w D:{self.detector_w} , w C: {self.context_w}")
                print(f"loss C:{loss_c}")
                print("-------------------------------------------")
        self.clear_buffer()


    def clear_buffer(self):
        self.buffer_data["en_score"]=[]
        self.buffer_data["contexts"]=[]
        # if len(self.threshold_per_method)>0:
        #     self.buffer_data["scores"]=[[] for q in range(len(self.names)+1)]
        #else:
        self.buffer_data["scores"]=[[] for q in range(len(self.names))]
        self.buffer_data["dates"]=[]
        self.buffer_data["data"] = None


    def _calculate_fields(self,scores, data, event_data, sourceup, add_data_to_context=False):
        detector_series = []
        number_of_series = 0

        dates_to_add = data.index

        for q, d_score in enumerate(scores):
            if self.names[q] in self.all_event_names:
                continue
            detector_series.append((f"d_{q}", d_score, dates_to_add))
            number_of_series += 1
            break  # TO DO: This should be deleted and get all the data thorough context
        if isinstance(add_data_to_context, collections.abc.Sequence):
            for col in add_data_to_context:
                detector_series.append(
                    (f"raw_data_{col}", [v for v in data[col].values], dates_to_add))
                self.context_names.append(f"raw_data_{col}")
                number_of_series += 1
        elif add_data_to_context:
            if len(data.columns) <= 1:
                detector_series.append(("raw_data", [v for v in data[data.columns[0]].values], dates_to_add))
                self.context_names.append(f"raw_data")
                number_of_series += 1
            else:
                for i in range(len(data.columns)):
                    detector_series.append(
                        (f"raw_data_{i}", [v for v in data[data.columns[i]].values], dates_to_add))
                    self.context_names.append(f"raw_data_{i}")
                    number_of_series += 1
        categ_events = []
        conf_events = []
        for q, d_score in enumerate(scores):
            if q >= len(self.names):
                continue
            if self.names[q] not in self.all_event_names:
                continue
            conf_events.append(
                (self.names[q], [date for date, out in zip(dates_to_add, d_score) if out >= 1], "configuration"))
        for event_type in self.all_event_names:
            # TODO : accept and isolation, make all_event_names array of tuples.
            # event_data should be in form of date,description,type,source
            such_events = event_data[event_data["description"] == event_type]
            datesofevent = [dt for dt in such_events['date'].values]
            if len(datesofevent) == 0:
                continue
            if such_events.shape[0] > 0:
                type_of_such_event = such_events.iloc[0]['type']
                if type_of_such_event in ["isolated"]:
                    eventconf2 = (event_type, datesofevent, type_of_such_event)
                    conf_events.append(eventconf2)
                elif type_of_such_event in ["categorical"]:
                    valueofev = [v for v in such_events['value'].values]
                    if f"state_{event_type}" not in self.context_names:
                        self.context_names.append(f"state_{event_type}")
                    for v in set(valueofev):
                        if f"{v}_{event_type}" not in self.context_names:
                            self.context_names.append(f"{v}_{event_type}")
                    eventconf2 = (event_type, datesofevent, valueofev, type_of_such_event)
                    categ_events.append(eventconf2)
                elif type_of_such_event in ["continuous"]:
                    detector_series.append((event_type, [dt for dt in such_events['value'].values], datesofevent))
                else:
                    eventconf2 = (event_type, datesofevent, "configuration")
                    conf_events.append(eventconf2)
            else:
                eventconf2 = (event_type, datesofevent, "configuration")
                conf_events.append(eventconf2)
            number_of_series += 1

        self.context_names = list(set(self.context_names))
        if len(self.context_names) < 8 and self.fill_empty:
            for i in range(1, 8 - number_of_series + 1):
                detector_series.append((f"empty_{i}", [0 for v in dates_to_add], dates_to_add))
                self.context_names.append(f"empty_{i}")
    def _produce_context(self, scores, data, event_data, sourceup, add_data_to_context=False):

        detector_series = []
        number_of_series = 0

        dates_to_add = data.index

        for q, d_score in enumerate(scores):
            if self.names[q] in self.all_event_names:
                continue
            detector_series.append((f"d_{q}", d_score, dates_to_add))
            number_of_series += 1
            break  # TO DO: This should be deleted and get all the data thorough context
        if isinstance(add_data_to_context, collections.abc.Sequence):
            for col in add_data_to_context:
                detector_series.append(
                    (f"raw_data_{col}", [v for v in data[col].values], dates_to_add))
                self.context_names.append(f"raw_data_{col}")
                number_of_series += 1
        elif add_data_to_context:
            if len(data.columns) <= 1:
                detector_series.append(("raw_data", [v for v in data[data.columns[0]].values], dates_to_add))
                self.context_names.append(f"raw_data")
                number_of_series += 1
            else:
                for i in range(len(data.columns)):
                    detector_series.append(
                        (f"raw_data_{i}", [v for v in data[data.columns[i]].values], dates_to_add))
                    self.context_names.append(f"raw_data_{i}")
                    number_of_series += 1
        categ_events = []
        conf_events = []
        for q, d_score in enumerate(scores):
            if q>=len(self.names):
                continue
            if self.names[q] not in self.all_event_names:
                continue
            conf_events.append((self.names[q], [date for date,out in zip(dates_to_add,d_score) if out>=1], "configuration"))
        for event_type in self.all_event_names:
            # TODO : accept and isolation, make all_event_names array of tuples.
            # event_data should be in form of date,description,type,source
            such_events=event_data[event_data["description"] == event_type]
            datesofevent = [dt for dt in such_events['date'].values]
            if len(datesofevent) == 0:
                continue
            if such_events.shape[0]>0:
                type_of_such_event=such_events.iloc[0]['type']
                if type_of_such_event in ["isolated"]:
                    eventconf2 = (event_type, datesofevent, type_of_such_event)
                    conf_events.append(eventconf2)
                elif type_of_such_event in ["categorical"]:
                    valueofev = [v for v in such_events['value'].values]
                    if f"state_{event_type}" not in self.context_names:
                        self.context_names.append(f"state_{event_type}")
                    for v in set(valueofev):
                        if f"{v}_{event_type}" not in self.context_names:
                            self.context_names.append(f"{v}_{event_type}")
                    eventconf2 = (event_type, datesofevent,valueofev, type_of_such_event)
                    categ_events.append(eventconf2)
                elif type_of_such_event in ["continuous"]:
                    detector_series.append((event_type, [dt for dt in such_events['value'].values], datesofevent))
                else:
                    eventconf2 = (event_type, datesofevent, "configuration")
                    conf_events.append(eventconf2)
            else:
                eventconf2 = (event_type, datesofevent, "configuration")
                conf_events.append(eventconf2)
            number_of_series += 1

        self.context_names = list(set(self.context_names))
        if len(self.context_names) < 8 and self.fill_empty:
            for i in range(1, 8 - number_of_series+1):
                detector_series.append((f"empty_{i}", [0 for v in dates_to_add], dates_to_add))
                self.context_names.append(f"empty_{i}")
        # isolated #configuration
        stream = simulate_stream(detector_series, conf_events, categ_events,"d_0")
        c = 0
        ls = len(scores[0])
        source = "s"
        allrecords = [record for record in stream]


        #from PdmContext.utils.MovingPC import MovingPC
        # MPC = MovingPC()
        self.context_sources[sourceup] = ContextGenerator(target="d_0", context_horizon=self.context_horizon,
                                                          # Causalityfunct=MPC.calculate_with_pc_moving)
                                                          Causalityfunct=calculate_with_pc)
        contextslist=[]
        if DEBUG:
            for q in tqdm(range(len(allrecords))):
                record = allrecords[q]
                if 'FEATURE76' in record["name"]:
                    ok = 'ok'
                kati=self.context_sources[sourceup].collect_data(timestamp=record["timestamp"], source=source,
                                                            name=record["name"],
                                                            type=record["type"], value=record["value"])
                self.presource = sourceup
                if kati is not None:
                    contextslist.append(kati)
        else:
            for q in range(len(allrecords)):
                record = allrecords[q]
                kati=self.context_sources[sourceup].collect_data(timestamp=record["timestamp"], source=source,
                                                            name=record["name"],
                                                            type=record["type"], value=record["value"])

                # if self.presource is not None:
                #     file = open(f"pickles/C_{self.presource}.pickle", 'wb')
                #     pickle.dump(self.context_sources[self.presource], file)
                self.presource=sourceup
                if kati is not None:
                    contextslist.append(kati)

        context_data=contextslist
        if len (context_data)<ls:
            for i in range(ls-len(context_data)):
                context_data= [context_data[0]] +context_data
        return context_data


    def __del__(self):
        pass
        # for source in self.context_sources.keys():
        #     print("TO PLOT")
        #     self.context_sources[source].plot([["","d_0",""]])
        #     plt.savefig("d_0.png")
        #     plt.show()
        #     self.context_sources[source].plot([["","d_0","increase"]])
        #     plt.savefig("d_0_increase.png")
        #     plt.show()




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

def loss_calculation(scores,labels):
    AD1=0
    tp=0
    fp=0
    contamination=sum(labels)/len(labels)
    if len (scores)<1:
        return 0.1
    scorscopy=[sc for sc in scores]
    scorscopy.sort(reverse=True)
    th=scorscopy[min(len(scorscopy)-1,int(len(scorscopy)*contamination))]
    if th==1:
        th=0.9
    for i,lb in enumerate(labels):
        if lb>0 and scores[i]>th:
            AD1=1
        if lb>0 and scores[i]>th:
            tp+=1
        elif lb==0 and scores[i]>th:
            fp+=1
    if tp==0:
        return 0.1
    Precision=tp/(tp+fp)
    if AD1==0:
        return 0.1
    f1 = 2 * (Precision * AD1) / (Precision + AD1)
    return max(f1,0.1)

