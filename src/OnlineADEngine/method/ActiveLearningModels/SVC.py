from method.ActiveLearningModels.Base import Gen_FEED


DEBUG=False
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from imblearn.under_sampling import ClusterCentroids
from sklearn.cluster import MiniBatchKMeans
random_state=42


class SVMFEED(Gen_FEED):

    def __init__(self, globalkeys, plot=False, include_scores_in_model=False, dimensonality_red = 0):
        super().__init__(globalkeys, plot, include_scores_in_model, dimensonality_red)
        self.Not_fitted = True
        self.model = None
        self.inner_buffer_zero=[]
        self.inner_buffer_ones=[]

        self.add_raw = True
        self.preX0 = None


    def get_vectors_from_context(self, data, labels, scores):
        """
        Return CR vectors only and labels after resampling

        data is a context list
        scores a list of scores same size as data
        """
        all_vectors = []
        fnames = []
        labs = []

        # print(len(labels))
        for cont, lll, qi in zip(data, labels, [qq for qq in range(len(labels))]):
            neighboarsMatrix = np.zeros((len(self.globalkeys), len(self.globalkeys)))
            for edge in cont.CR['edges']:
                if edge[0] in self.globalkeys and edge[1] in self.globalkeys:
                    i = self.globalkeys.index(edge[0])
                    j = self.globalkeys.index(edge[1])
                    neighboarsMatrix[i, j] = 1
                    neighboarsMatrix[j, i] = 1
            vector, names = self.make_vector_from_categorical_edges(neighboarsMatrix)
            if self.include_scores_in_model:
                for si, scorelist in enumerate(scores):
                    vector.append(scorelist[qi])
                    names.append(f"score_{si}")
            fnames = names
            if self.add_raw:
                v_vecotr = self.extract_last_value(cont)
                vector = v_vecotr+ vector
            vector = np.array(vector)
            # exists = any((np.array_equal(vector, vec) and lll == llll) for vec, llll in zip(all_vectors, labs))

            all_vectors.append(vector)
            labs.append(lll)
        all_vectors = self.dimensonality_reduction(all_vectors, to_fit=True)
        return all_vectors, fnames, labs

    def extract_last_value(self,con):
        vector=[]
        for key in self.globalkeys:
            if key not in con.CD:
                vector.append(0)
            elif con.CD[key] is None:
                vector.append(0)
            else:
                vector.append(con.CD[key][-1])
        return vector
    def get_vectors_from_context_without(self,data,scores):
        """
        Return CR vectors only

        data is a context list
        scores a list of scores same size as data
        """
        all_vectors = []
        fnames = []
        for qi,cont in enumerate(data):
            neighboarsMatrix = np.zeros((len(self.globalkeys), len(self.globalkeys)))
            for edge in cont.CR['edges']:
                if edge[0] in self.globalkeys and edge[1] in self.globalkeys:
                    i = self.globalkeys.index(edge[0])
                    j = self.globalkeys.index(edge[1])
                    neighboarsMatrix[i, j] = 1
                    neighboarsMatrix[j, i] = 1
            vector, names = self.make_vector_from_categorical_edges(neighboarsMatrix)
            if self.include_scores_in_model:
                for si,scorelist in enumerate(scores):
                    vector.append(scorelist[qi])
                    names.append(f"score_{si}")
            fnames=names
            if self.add_raw:
                v_vecotr=self.extract_last_value(cont)
                vector=v_vecotr+vector
            vector = np.array(vector)
            all_vectors.append(vector)
        all_vectors = self.dimensonality_reduction(all_vectors)
        return all_vectors, fnames
    def predict(self,data,scores):# last values



        # Show the plot

        if self.Not_fitted:
            self.model = make_pipeline(StandardScaler(), SVC(gamma='auto',probability=True,random_state=random_state))
            maxs = max(scores[0])
            mins = min(scores[0])
            return [0 for sc in scores[0]]
        else:
            all_vectors, featnames = self.get_vectors_from_context_without(data,scores)
            probs=self.model.predict_proba(np.array(all_vectors))
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

    def update_wiehgts_detector(self, labels,data,scores,loss_func,lr=0.0001):

        all_vectors1,featnames,labels1=self.get_vectors_from_context(data,labels,scores)

        self.add_data_to_buffer(all_vectors1,labels1)
        all_vectors,labels=self.get_train_data()

        if sum(labels)==len(labels):
            newlabels=[0 if i<len(labels)/2 else 1 for i in range(len(labels))]
        else:
            newlabels=labels

        data=all_vectors
        label = newlabels[-len(all_vectors):]

        if self.Not_fitted:
            self.model = make_pipeline(MinMaxScaler(), SVC(gamma='auto',probability=True,random_state=random_state))

            self.model.fit(np.array(data), label)

            self.Not_fitted = False
            probs = self.model.predict_proba(np.array(all_vectors1))
            en_score = []
            for pb in probs:
                en_score.append(pb[1])
            return loss_func(en_score,labels1)
        else:
            probs = self.model.predict_proba(np.array(all_vectors1))
            en_score = []
            for pb in probs:
                en_score.append(pb[1])
            self.model = make_pipeline(MinMaxScaler(), SVC(gamma='auto', probability=True,random_state=random_state))

            self.model.fit(np.array(data), labels)

            self.Not_fitted = False

            return loss_func(en_score, labels1)
