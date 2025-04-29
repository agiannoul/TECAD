import abc
import numpy as np
from sklearn.decomposition import PCA


class Gen_FEED():

    def __init__(self,globalkeys,plot=False,include_scores_in_model=False,dimensonality_red=0):
        self.plotit=plot
        self.model=None
        self.globalkeys=[f"{gk}@s" if gk not in ["d_0","Cstate"] else gk for gk in globalkeys]
        self.Not_fitted=True
        self.include_scores_in_model=include_scores_in_model

        self.categoricals = []
        for name in self.globalkeys:
            if "state_" in name:
                self.categoricals.append(name.split("state_")[-1])
        # self.params = {
        #     'objective': 'binary:logistic',
        #     'max_depth': 4,
        #     'learning_rate': 0.3
        # }
        # self.boost_rounds=70
        self.all_vectors_original = []
        self.dimensonality_red=dimensonality_red

    def dimensonality_reduction(self, vectors, to_fit=False):
        if len(vectors[0])<self.dimensonality_red or self.include_scores_in_model==0:
            return vectors
        if to_fit:
            self.all_vectors_original.extend(vectors)
            X = np.array(self.all_vectors_original)
            self.pca = PCA(n_components=self.dimensonality_red)
            self.pca.fit(X)
        X = np.array(vectors)
        reduced=self.pca.transform(X)
        return reduced

    def make_vector_from_categorical_edges(self,neighboarsMatrix):
        vector=[]
        names=[]
        for i in range(len(self.globalkeys)):
            for j in range(i,len(self.globalkeys)):
                if j==i:
                    continue
                if self.globalkeys[i].split("_")[-1] in self.categoricals and self.globalkeys[j].split("_")[-1] in self.categoricals:
                    if self.globalkeys[i].split("_")[-1]==self.globalkeys[j].split("_")[-1]:
                        # ok
                        ok='ok'
                    elif "state_" in self.globalkeys[i] and "state_" in self.globalkeys[j]:
                        ok='ok'
                    else:
                        continue
                vector.append(neighboarsMatrix[i, j])
                names.append(f"{self.globalkeys[i]}+{self.globalkeys[j]}")
        return vector,names
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
            vector = np.array(vector)
            all_vectors.append(vector)

        return all_vectors, fnames

    def get_vectors_from_context(self,data,labels,scores):
        """
        Return CR vectors only and labels after resampling

        data is a context list
        scores a list of scores same size as data
        """
        all_vectors = []
        fnames =[]
        nwelabs=[]
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

            vector=np.array(vector)
            #exists = any((np.array_equal(vector, vec) and lll == llll) for vec, llll in zip(all_vectors, labs))


            all_vectors.append(vector)
            labs.append(lll)

        ratio = sum(labs) / len(labs)
        multipplier = 1
        if ratio == 0:
            multipplier = len(labs)
        else:
            multipplier = int(1 / ratio)
        #print(f"l: {len(labs)}")
        #print(multipplier)

        all_vectors2=[]
        for i in range(len(all_vectors)):
            if labs[i]==1:
                for q in range(multipplier):
                    all_vectors2.append(all_vectors[i])
                    nwelabs.append(1)
            else:
                all_vectors2.append(all_vectors[i])
                nwelabs.append(0)
        #print(f"+: {sum(nwelabs)}")
        if sum(nwelabs)==0:
            ok='ok'
        return all_vectors2,fnames,nwelabs

    @abc.abstractmethod
    def predict(self,data,scores):# last values
        pass

    @abc.abstractmethod
    def update_wiehgts_detector(self, labels,data,scores,loss_func,lr=0.0001):
        pass