import numpy as np
from scipy.spatial.distance import cdist
from sklearn.ensemble import IsolationForest as isolation_forest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM


class distance_based_k_r:
    def __init__(self, k=5, window_norm=False, metric="euclidean"):
        self.k = k
        self.window_norm = window_norm
        self.metric = metric
        self.to_fit = None

    def fit(self, df):
        self.to_fit = df
        if len(self.to_fit)<=self.k:
            self.k=len(self.to_fit)-1
    def predict(self, df):
        to_predict = df
        D, _ = self._search(to_predict,self.to_fit)
        score = []
        for d in D[:, self.k - 1]:
            score.append(d)
        return score

    def _calc_dist(self, query, pts):
        return cdist(query, pts, metric=self.metric)

    def _search(self, query, points):
        dists = self._calc_dist(query, points)

        I = (
            np.argsort(dists, axis=1)
            if self.k > 1
            else np.expand_dims(np.argmin(dists, axis=1), axis=1)
        )
        D = np.take_along_axis(np.array(dists), I, axis=1)
        return D, I

def normalize_list(lst):
    min_val = min(lst)
    max_val = max(lst)
    normalized_lst = [(x - min_val) / (max_val - min_val) for x in lst]
    return normalized_lst
def isolation_fores_semi(normal,target,*args, **kwargs):
    if 'random_state' in kwargs.keys():
        clf=isolation_forest(n_estimators=50,**kwargs)
    else:
        clf = isolation_forest(n_estimators=50, random_state=93)

    clf.fit(normal)
    return [-1*sc for sc in clf.score_samples(target).tolist()]


def ocsvm_semi(normal,target,*args, **kwargs):
    clf=OneClassSVM()
    clf.fit(normal)
    return [-1*sc for sc in clf.score_samples(target).tolist()]

def lof_semi(normal,target,*args, **kwargs):
    clf = LocalOutlierFactor(novelty=True,**kwargs)
    clf.fit(normal.values)
    return [-1 * sc for sc in clf.score_samples(target.values).tolist()]
def distance_based(normal,target,*args, **kwargs):
    clf=distance_based_k_r(k=1)
    clf.fit(normal.values)
    return clf.predict(target.values)