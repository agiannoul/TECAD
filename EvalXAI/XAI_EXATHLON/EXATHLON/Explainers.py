"""LIME explanation discovery module.
"""
import os
import pickle
import re
import time
import warnings

import numpy as np
from lime import lime_tabular

# import matplotlib.pyplot as plt
# import networkx as nx
# import pandas as pd
# from PdmContext.ContextGeneration import ContextGenerator
# from PdmContext.utils.simulate_stream import simulate_from_df
#
#
# from PdmContext.utils.dbconnector import SQLiteHandler
# from PdmContext.Pipelines import ContextAndDatabase
# # add absolute src directory to python path to import other project modules
import sys
import shap
from numba.cuda.libdevice import trunc
from torch.nn.functional import threshold

src_path = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
sys.path.append(src_path)
from explanation.model_dependent.model_dependent_explainers import ModelDependentExplainer

from tqdm import tqdm


class MYShap_MD(ModelDependentExplainer):
    def __init__(self, ad_model, sequence_size,threshold=0.5):
        self.threshold = threshold
        self.ad_model = ad_model
        self.shap_model = None
        self.sample_length = sequence_size
        self.small_anomalies_expansion = 'before'
        # coverage policy for anomalies larger than sample length
        self.large_anomalies_coverage = 'center'
        self.all_importance = {}

    def extract_features_by_threshold(self,shap_values, features):
        """
        Extracts features with SHAP values greater than a given threshold for each sample.
        """
        threshold=self.threshold
        significant_features = [features[i][3:] for i in range(len(shap_values)) if abs(shap_values[i]) > threshold]
        significant_features=sorted(list(set(significant_features)))
        return {'important_fts': significant_features}, 1
    def ad_scoring_func(self,sample):
        if len(sample.shape)>1:
            scores=[]
            for innter_sample in sample:
                sc,_=self.ad_model.get_record_score_sample(innter_sample)
                scores.append(sc)
        else:
            scores, _ = self.ad_model.get_record_score_sample(sample)
        return np.array(scores)

    def explain_sample(self, sample, sample_labels=None):
        if len(sample.shape)>1:
            reci=sample[-1]
        else:
            reci=sample
        rec = tuple([kati for kati in reci])
        return self.all_importance[rec]


    def add_importance(self, test):
        for period in test:
            training_samples = np.array([sample for sample in period])
            # training_samples=period
            print(training_samples.shape)
            feature_names = [f'ft_{i}' for i in range(training_samples.shape[1])]
            X_train_summary =shap.kmeans(training_samples, 4)
            self.shap_model = shap.KernelExplainer(self.ad_scoring_func, X_train_summary)

            shap_values = self.shap_model.shap_values(np.array(period))
            for qi in tqdm(range(len(period))):
                shap_val=shap_values[qi]
                sample = period[qi]

                sc, reci = self.ad_model.get_record_score(sample)
                self.all_importance[reci] = self.extract_features_by_threshold(shap_val , feature_names)

        file = open("saved_models/shap.pickle", 'wb')
        pickle.dump(self.all_importance, file)


class MYShapGT(ModelDependentExplainer):
    def __init__(self, ad_model, sequence_size,threshold=0.1,load_existing=True):
        self.ad_model = ad_model
        self.threshold = threshold
        self.shap_model = None
        self.sample_length = sequence_size
        self.small_anomalies_expansion = 'before'
        # coverage policy for anomalies larger than sample length
        self.large_anomalies_coverage = 'center'
        self.all_importance = {}
        self._load=load_existing
    def extract_features_by_threshold(self,shap_values, features):
        """
        Extracts features with SHAP values greater than a given threshold for each sample.
        """
        threshold=self.threshold
        significant_features = [features[i][3:] for i in range(len(shap_values)) if abs(shap_values[i]) > threshold]
        significant_features=sorted(list(set(significant_features)))
        return {'important_fts': significant_features}, 1
    def ad_scoring_func(self,sample):
        if len(sample.shape)>1:
            scores=[]
            for innter_sample in sample:
                sc,_=self.ad_model.get_record_score_sample(innter_sample)
                scores.append(sc)
        else:
            scores, _ = self.ad_model.get_record_score_sample(sample)
        return np.array(scores)

    def explain_sample(self, sample, sample_labels=None):
        if len(sample.shape)>1:
            reci=sample[-1]
        else:
            reci=sample
        rec = tuple([kati for kati in reci])
        return self.all_importance[rec]


    def add_importance(self, test):
        try:
            if self._load:
                with open('saved_models/shap.pickle', 'rb') as file:
                    self.all_importance = pickle.load(file)
                return
        except Exception as e:
            print(f"No saved model: {e}")
            self._load=False

        for period in test:
            training_samples = np.array([sample for sample in period])
            # training_samples=period
            print(training_samples.shape)
            feature_names = [f'ft_{i}' for i in range(training_samples.shape[1])]
            X_train_summary =shap.kmeans(training_samples, 4)
            self.shap_model = shap.KernelExplainer(self.ad_scoring_func, X_train_summary)

            shap_values = self.shap_model.shap_values(np.array(period))
            for qi in tqdm(range(len(period))):
                shap_val=shap_values[qi]
                sample = period[qi]

                sc, reci = self.ad_model.get_record_score(sample)
                self.all_importance[reci] = self.extract_features_by_threshold(shap_val , feature_names)

        file = open("saved_models/shap.pickle", 'wb')
        pickle.dump(self.all_importance, file)


class MYLimeGT(ModelDependentExplainer):
    """Local Interpretable Model-agnostic Explanations (LIME) explanation discovery class.

    For this explainer, the `fit` method must be called before explaining samples.

    See https://arxiv.org/pdf/1602.04938.pdf for more details.



    This uses ground truth to produce labels
    """
    def __init__(self, ad_model,sequence_size,features=5,load_existing=True):
        #super().__init__(args, output_path, ad_model)
        # number of features to report in the explanations
        self.ad_model=ad_model
        # LIME model
        self.sample_length=sequence_size


        self.small_anomalies_expansion = 'before'
        # coverage policy for anomalies larger than sample length
        self.large_anomalies_coverage = 'center'
        self.n_features=features
        self.all_importance={}
        self.lime_model=None
        self._load=load_existing




    def explain_sample(self, sample, sample_labels=None):
        if len(sample.shape)>1:
            reci=sample[-1]
        else:
            reci=sample
        rec = tuple([kati for kati in reci])
        return self.all_importance[rec]

    def ad_scoring_func(self,samples):
        scores=[]
        for sample in samples:
            sample=sample[0]
            sc,_=self.ad_model.get_record_score_sample(sample)
            scores.append(sc)
        return np.array(scores)
    def add_importance(self, test):
        try:
            if self._load:
                with open('saved_models/lime.pickle', 'rb') as file:
                    self.all_importance = pickle.load(file)
                return
        except Exception as e:
            print(f"No saved model: {e}")
            self._load=False

        for period in test:
            training_samples=np.array([[sample] for sample in period])
            #training_samples=period
            print(training_samples.shape)
            feature_names = [f'ft_{i}' for i in range(training_samples[0].shape[1])]
            self.lime_model = lime_tabular.RecurrentTabularExplainer(
                training_samples, mode='regression', feature_names=feature_names,
                discretize_continuous=True, discretizer='decile'
            )
            for qi in tqdm(range(len(period))):
                sample=period[qi]
                sc, reci = self.ad_model.get_record_score(sample)
                self.all_importance[reci]=self._get_importance_features(sample)
        file=open("saved_models/lime.pickle", 'wb')
        pickle.dump(self.all_importance,file)
    def _get_importance_features(self,sample):
        explanation = self.lime_model.explain_instance(
            sample, self.ad_scoring_func, num_features=self.n_features,num_samples=100
        )

            # important feature indices are extracted from the feature names reported in the explanation
        important_fts = []
        for statement, weight in explanation.as_list():
            # "ft_{id}" names were used to identify numbers that relate to features in the strings
            ft_name = re.findall(r'ft_\d+', statement)[0]
            ft_index = int(ft_name[3:])
            important_fts.append(ft_index)
        # a same feature can occur in multiple statements
        returned_fts = sorted(list(set(important_fts)))
        if len(returned_fts) == 0:
            warnings.warn('No explanation found for the sample.')
        return {'important_fts': returned_fts},1



class MYLime_MD(ModelDependentExplainer):
    """Local Interpretable Model-agnostic Explanations (LIME) explanation discovery class.

    For this explainer, the `fit` method must be called before explaining samples.

    See https://arxiv.org/pdf/1602.04938.pdf for more details.


    This uses a model to produce labels
    """
    def __init__(self, ad_model,sequence_size,features=5):
        #super().__init__(args, output_path, ad_model)
        # number of features to report in the explanations
        self.ad_model=ad_model
        # LIME model
        self.lime_model = None
        self.sample_length=sequence_size


        self.small_anomalies_expansion = 'before'
        # coverage policy for anomalies larger than sample length
        self.large_anomalies_coverage = 'center'
        self.n_features=features
        self.all_importance={}
        self.lime_model=None




    def explain_sample(self, sample, sample_labels=None):
        if len(sample.shape)>1:
            reci=sample[-1]
        else:
            reci=sample
        return self._get_importance_features(reci)

    def ad_scoring_func(self,samples):
        scores=[]
        for sample in samples:
            sample=sample[0]
            sc,_=self.ad_model.get_record_score_sample(sample)
            scores.append(sc)
        return np.array(scores)
    def add_importance(self, train):
        allperidos=None
        for period in train:
            if allperidos is None:
                allperidos=period
            else:
                allperidos=np.concatenate((allperidos, period), axis=0)
        print(allperidos.shape)
        training_samples = np.array([[sample] for sample in allperidos])
        print(training_samples.shape)
        feature_names = [f'ft_{i}' for i in range(training_samples[0].shape[1])]
        self.lime_model = lime_tabular.RecurrentTabularExplainer(
            training_samples, mode='regression', feature_names=feature_names,
            discretize_continuous=True, discretizer='decile'
        )


    def _get_importance_features(self,sample):
        explanation = self.lime_model.explain_instance(
            sample, self.ad_scoring_func, num_features=self.n_features,num_samples=500
        )

            # important feature indices are extracted from the feature names reported in the explanation
        important_fts = []
        for statement, weight in explanation.as_list():
            # "ft_{id}" names were used to identify numbers that relate to features in the strings
            ft_name = re.findall(r'ft_\d+', statement)[0]
            ft_index = int(ft_name[3:])
            important_fts.append(ft_index)
        # a same feature can occur in multiple statements
        returned_fts = sorted(list(set(important_fts)))
        if len(returned_fts) == 0:
            warnings.warn('No explanation found for the sample.')
        return {'important_fts': returned_fts},1

