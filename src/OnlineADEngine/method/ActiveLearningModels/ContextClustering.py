import random

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from PdmContext.utils.distances import jaccard_distance_CR
from sklearn.cluster import AgglomerativeClustering

def extract_representatives(contexts_list, my_distance,k=8,number_per_cluster=3):
    """
    Extract representatives context objects from a list of contexts using Hierarchical Clustering.
    :param contexts_list: List of contexts
    :param my_distance: Distance function in form dist(context,context) returning a real value.
    :param k: Number of clusters
    :param number_per_cluster: Representative contexts to return per cluster
    :return: A list of representative contexts from the contexts_list
    """
    cp = ClusteringPipeline(k=k, dist_funct=my_distance)
    clusters = cp.fit(contexts_list, exclude_zero_edges=True)
    representatives = cp.get_representatives(number_per_class=number_per_cluster)
    return representatives

class ClusteringPipeline():
    def __init__(self,k,dist_funct=jaccard_distance_CR,):
        self.k=k
        self.dist_funct=dist_funct
        self.clusters_sets=None

    def calculate_distances(self,context_list):
        distance_Matrix = np.zeros((len(context_list), len(context_list)))

        limit = len(context_list)
        print(limit)
        for i in tqdm(range(limit)):
            for j in range(limit):
                if j == i:
                    distance_Matrix[i, j] = 0
                elif i > j:
                    distance_Matrix[i, j] = distance_Matrix[j, i]
                else:
                    distance_Matrix[i, j] = self.dist_funct(context_list[i], context_list[j])
        return distance_Matrix

    def fit(self, context_list, exclude_zero_edges=True):
        """
        Performs Hierarchical Clustering on a set of contexts.

        :param context_list: List of context objects
        :param exclude_zero_edges: Weather or not to exclude contexts with no edges from clustering.
        :return: A dictionary where keys are the labels of the different clusters and values a list of contexts objects for that cluster.
        """
        tofit=[]
        positions=[]
        for i,cont in enumerate(context_list):
            if len(cont.CR["edges"])==0 and exclude_zero_edges:
                continue
            tofit.append(cont)
            positions.append(i)

        distance_Matrix=self.calculate_distances(tofit)


        cluslabels = AgglomerativeClustering(metric="precomputed", linkage="average", n_clusters=self.k).fit(distance_Matrix).labels_
        self.labels = cluslabels

        clusters_res={}
        for q,lb in  enumerate(cluslabels):

            if lb not in clusters_res:
                clusters_res[lb]=[tofit[q]]
            else:
                clusters_res[lb].append(tofit[q])

        self.clusters_sets=clusters_res
    def get_representatives(self,number_per_class=3):
        """
        Extract representatives from Context clusters.

        :param number_per_class: How many representatives to exclude from a cluster (random selection)
        :return: A list containing representatives contexts for each cluster
        """
        context_to_return = []
        if self.clusters_sets is None:
            raise ValueError("Clusters set is empty, call fit() first.")

        for key in self.clusters_sets.keys():
            if number_per_class<=len(self.clusters_sets[key]):
                random_indices = random.sample(range(len(self.clusters_sets[key])), min(number_per_class, len(self.clusters_sets[key])))
                random_elements = [self.clusters_sets[key][i] for i in random_indices]
            else:
                random_indices = np.random.choice([i for i in range(len(self.clusters_sets[key]))], number_per_class, replace=True)
                random_elements = [self.clusters_sets[key][i] for i in random_indices]
            context_to_return.extend(random_elements)

        return context_to_return

    def plot(self):
        for q,lb in  enumerate(list(set(self.labels))):
            self.plotsingleclust(q,self.clusters_sets[lb])
        plt.show()


    def plotsingleclust(self,num,clust,desc=None):
        xaxistimes = [cont.timestamp for cont in clust]
        yaxistimes = [num for cont in clust]
        if len(xaxistimes)==0:
            return
        edges = [cont.CR["edges"] for cont in clust]
        if len(edges)!=0:
            edgeplot = edges[len(edges) // 2]
        else:
            edgeplot=[]
        toplot = ""
        seen = []
        for ed in edgeplot:
            if (ed[0], ed[1]) not in seen:
                toplot += f"({ed[0]},{ed[1]}),"
                seen.append((ed[1], ed[0]))
                seen.append((ed[0], ed[1]))
        if desc is None:
            plt.text(xaxistimes[len(xaxistimes) // 2], yaxistimes[len(yaxistimes) // 2] + 0.1, toplot, fontsize=8)
        else:
            plt.text(xaxistimes[len(xaxistimes) // 2], yaxistimes[len(yaxistimes) // 2] + 0.1, desc, fontsize=8)

        plt.scatter(xaxistimes, yaxistimes)
