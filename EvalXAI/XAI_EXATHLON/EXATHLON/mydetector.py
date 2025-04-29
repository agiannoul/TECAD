
import numpy as np
from sklearn.neighbors import KDTree
# example of implementation


class TestExplainer:
    """
    This produce labels equal to ground truth
    """
    def __init__(self):
        self.records = []
        self.window_size =1
        self.alldict = {}
    def fit_data_labels(self,trainingdata,trainlabels):
        self.records=[]
        print()
        show=True
        for period,periodlabels in zip(trainingdata,trainlabels):
            for recordi,label in zip(period,periodlabels):
                self.records.append(recordi)
                if show:
                    print(type(recordi))
                    show=False
                record=tuple([kati for kati in recordi])
                if record in self.alldict.keys():
                    self.alldict[record]=max(self.alldict[record],label)
                else:
                    self.alldict[record]=label
        self.kdt = KDTree(self.records, leaf_size=50, metric='euclidean')

    def fit(self,trainingdata):
        self.records=[]
        for period in trainingdata:
            for record in period:
                self.records.append(record)

    def mindist(self,record):
        alldis=[]
        for r in self.records:
            alldis.append(np.linalg.norm(record-r))
        return min(alldis)
    def score_windows(self,X):
        score=[]
        for window in X:
            sc,_=self.get_record_score(window[-1])
            score.append(sc)
        return score
    def get_record_score(self,recordi):

        record = tuple([kati for kati in recordi])
        return self.alldict[record],record
    def get_record_score_sample(self,recordi):
        dist,closest=  self.kdt.query([recordi],k=1)
        #print(closest)
        closest=closest[0][0]
        #closest=0
        rec=self.records[closest]
        record = tuple([kati for kati in rec])
        return self.alldict[record], record
    def get_name(self):
        return ""


from sklearn import svm
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
class ModelSpecific:
    """
    Wrapper for Anomaly Detectors , to evaluate explainability.
    """
    def __init__(self,modelname):
        self.records = []
        self.window_size =1
        self.modelname=modelname
        if modelname=="lof":
            self.model = LocalOutlierFactor(n_neighbors=2)
        elif modelname=="IF":
            self.model = IsolationForest(random_state=0)
        else:
            self.model = OneClassSVM(gamma='auto')

    def fit_data(self,trainingdata):
        X=[]
        y=[]
        for period in trainingdata:
            for recordi in period:
                X.append(recordi)
        self.model.fit(X)

    def fit_data_labels(self,trainingdata,trainlabels):
        X=[]
        y=[]
        for period,periodlabels in zip(trainingdata,trainlabels):
            for recordi,label in zip(period,periodlabels):
                X.append(recordi)
                y.append(int(label>0))
        self.model.fit(X, y)

    def get_record_score_sample(self,recordi):
        if self.model.predict([recordi])[0]==-1:
            return 1,None
        else:
            return 0,None
    def get_record_score(self,recordi):
        if self.model.predict([recordi])[0] == -1:
            outp=1
        else:
            outp=0
        record = tuple([kati for kati in recordi])
        return outp,record
    def get_model(self):
        return self.model

    def get_name(self):
        return self.modelname