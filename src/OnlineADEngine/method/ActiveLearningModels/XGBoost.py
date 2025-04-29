from method.ActiveLearningModels.Base import Gen_FEED



DEBUG=False
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from xgboost import XGBClassifier

random_state=42

class XGB2FEED(Gen_FEED):

    def __init__(self, globalkeys, plot=False, include_scores_in_model=False,dimensonality_red=0):
        super().__init__(globalkeys, plot, include_scores_in_model,dimensonality_red)
        self.Not_fitted=True
        self.model = None
        self.inner_buffer_zero=[]
        self.inner_buffer_ones=[]
        self.add_raw=True
        self.preX0=None


    def get_vectors_from_context(self,data,labels,scores):
        """
        Return CR vectors only and labels after resampling

        data is a context list
        scores a list of scores same size as data
        """
        all_vectors = []
        fnames =[]
        labs=[]

        #print(len(labels))
        for cont,lll,qi in zip(data,labels,[qq for qq in range(len(labels))]):
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
                v_vecotr=self.extract_last_value(cont)
                vector=v_vecotr+vector
            vector=np.array(vector)
            #exists = any((np.array_equal(vector, vec) and lll == llll) for vec, llll in zip(all_vectors, labs))


            all_vectors.append(vector)
            labs.append(lll)

        all_vectors=self.dimensonality_reduction(all_vectors,to_fit=True)

        return all_vectors,fnames,labs

    def predict(self,data,scores):# last values



        # Show the plot

        if self.Not_fitted:
            maxs = max(scores[0])
            mins = min(scores[0])
            return [0 for sc in scores[0]]
        else:
            all_vectors, featnames = self.get_vectors_from_context_without(data,scores)
            probs=self.model.predict_proba(np.array(all_vectors))
            en_score=[]
            for pb in probs:
                en_score.append(pb[1])
            return self.inner_transform(en_score)
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

        all_vectors=self.dimensonality_reduction(all_vectors)

        return all_vectors, fnames


    def add_data_to_buffer(self,all_vectors,labels):
        for vector,lab in zip(all_vectors,labels):
            if lab==1:
                self.inner_buffer_ones.append(vector)
            else:
                self.inner_buffer_zero.append(vector)


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
    def resample_contexts(self,X_1,X_0):
        import random
        random.seed(123)
        if len(X_1) < 1000:
            rX1=X_1+random.choices(X_1, k=1000-len(X_1))  # Sampling with replacement
        else:
            rX1=random.sample(X_1,1000)

        if len(X_0) < 1000:
            rX0=X_0+random.choices(X_0, k=1000-len(X_0))  # Sampling with replacement
        else:
            if self.preX0 is not None:
                if len(X_0)-self.preX0>1000:
                    rX0=random.sample(X_0[self.preX0:],1000)
                else:
                    rX0=X_0[self.preX0:]
                    rX0=rX0+random.sample(X_0[:self.preX0],1000-len(rX0))
            else:
                rX0=random.sample(X_0,1000)
        self.preX0=None
        return rX1,rX0

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
        res1,res0=self.resample_contexts(self.inner_buffer_ones, self.inner_buffer_zero)
        X_res=res1+res0
        y_res=[1 for i in res1]+[0 for i in res0]

        return shuffle_lists(X_res, y_res)

        #n_clusters = min(len(self.inner_buffer_zero),max(100,len(self.inner_buffer_ones)))  # Set the number of clusters based on the reduction you want


        # cc = ClusterCentroids(sampling_strategy={0:min(1000,len(self.inner_buffer_zero)),1:min(1000,len(self.inner_buffer_ones))},
        #     estimator=MiniBatchKMeans(n_init=1, random_state=0), random_state=42
        # )
        # X = self.inner_buffer_ones.copy()
        # X.extend(self.inner_buffer_zero)
        # y = [1 for i in range(len(self.inner_buffer_ones))]
        # y.extend([0 for i in range(len(self.inner_buffer_zero))])
        #
        # X_res, y_res = cc.fit_resample(X, y)
        #
        # self.inner_buffer_ones=[]
        # self.inner_buffer_zero=[]
        # for vec,lb in zip(X_res,y_res):
        #     if lb==1:
        #         self.inner_buffer_ones.append(vec)
        #     else:
        #         self.inner_buffer_zero.append(vec)
        # ratio=len(self.inner_buffer_zero)//len(self.inner_buffer_ones)
        #
        # fX_rest=[]
        # fy_res=[]
        # if ratio>=2:
        #     for i in range(len(X_res)):
        #         vec=X_res[i]
        #         lb=y_res[i]
        #         if lb==1:
        #             for q in range(ratio - 1):
        #                 fX_rest.append(vec)
        #                 fy_res.append(lb)
        #         fX_rest.append(vec)
        #         fy_res.append(lb)
        #     return fX_rest,fy_res
        # return X_res,y_res

    def update_wiehgts_detector(self, labels,data_or,scores,loss_func,lr=0.0001):

        all_vectors1,featnames,labels1=self.get_vectors_from_context(data_or,labels,scores)

        self.add_data_to_buffer(all_vectors1,labels1)
        all_vectors,labels=self.get_train_data()

        if sum(labels)==len(labels):
            newlabels=[0 if i<len(labels)/2 else 1 for i in range(len(labels))]
        else:
            newlabels=labels

        data=all_vectors
        label = newlabels[-len(all_vectors):]

        if self.Not_fitted:
            if self.add_raw:
                self.model = make_pipeline(MinMaxScaler(),  XGBClassifier(use_label_encoder=False, eval_metric='logloss',seed=0))
            else:
                self.model = make_pipeline(XGBClassifier(use_label_encoder=False, eval_metric='logloss', seed=0))
            self.model.fit(np.array(data), label)
            self.Not_fitted = False
            probs = self.model.predict_proba(np.array(all_vectors1))
            en_score = []
            for pb in probs:
                en_score.append(pb[1])
            return loss_func(self.inner_transform(en_score), labels1)

        else:
            probs = self.model.predict_proba(np.array(all_vectors1))
            en_score = []
            for pb in probs:
                en_score.append(pb[1])
            if self.add_raw:
                self.model = make_pipeline(MinMaxScaler(), XGBClassifier(use_label_encoder=False, eval_metric='logloss', seed=0,))
            else:
                self.model = make_pipeline(XGBClassifier(use_label_encoder=False, eval_metric='logloss',seed=0))

            self.model.fit(np.array(data), labels)
            self.Not_fitted = False

            return loss_func(self.inner_transform(en_score),labels1)
    def inner_transform(self,scores):
        return scores
        # size=100
        # sm=np.mean(scores[:size])
        # sstd=np.std(scores[:size])
        # return [(sc-sm)/sstd for sc in scores]