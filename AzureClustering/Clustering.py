import pickle

import matplotlib.pyplot as plt
import numpy as np
from pandas import Timedelta
from sklearn.cluster import dbscan

import methods
import read_data_inner
from PdmContext.ContextGeneration import ContextGenerator, ContextGeneratorBatch
from PdmContext.utils.causal_discovery_functions import calculate_with_pc
from PdmContext.utils.distances import distance_3D_sbd_jaccard
from PdmContext.utils.structure import Eventpoint

# df = pd.read_csv("wind-turbine-logs.csv",sep=";",header=0)
# df=df[df["Turbine_Identifier"]=="T07"]
# print(df.shape)
# print(df.head())
# print(df.describe())

class ClusteringPipeline:

    def dsitance(self,contexta, contextb):
        similarity, _ = distance_3D_sbd_jaccard(contexta, contextb, a=0)
        return 1 - similarity

    def load_context(self,source = 3,profile_hours = 150):
        with open(f'./context_list_{source}_{profile_hours}.pkl', 'rb') as file:
            loaded_list = pickle.load(file)

        distance_Matrix = np.zeros((len(loaded_list), len(loaded_list)))
        from tqdm import tqdm
        limit = min(10000,len(loaded_list))
        for i in tqdm(range(limit)):
            for j in range(limit):
                if j == i:
                    distance_Matrix[i, j] = 0
                elif i > j:
                    distance_Matrix[i, j] = distance_Matrix[j, i]
                else:
                    distance_Matrix[i, j] = self.dsitance(loaded_list[i], loaded_list[j])
        with open(f'distances_{source}_{profile_hours}.pkl', 'wb') as file:
            pickle.dump(distance_Matrix, file)

    def generate_context(self, source = 3,profile_hours = 150):



        dftelemetry, dfcontext, failures = read_data_inner.AzureDataOneSource(source=source)
        dfs, isfailure = read_data_inner.split_df_with_failures_isfaile(dftelemetry, failures)
        dfs_context, _ = read_data_inner.split_df_with_failures_isfaile(dfcontext, failures)

        list_train, list_test, context_list_dfs, isfailure, all_sources = read_data_inner.Azure_generate_train_test(dfs,
                                                                                                              isfailure,
                                                                                                              dfs_context,
                                                                                                              [source
                                                                                                               for i in
                                                                                                               dfs],
                                                                                                              period_or_count=f"{profile_hours} hours")


        context_gen = ContextGenerator(target="score", context_horizon="1 days", Causalityfunct=calculate_with_pc)
        context_list = []
        first_time=True
        from tqdm import tqdm
        episode=-1
        for trainset,testset,df in zip(list_train, list_test,context_list_dfs):
            episode+=1
            anomaly_scores = methods.ocsvm_semi(trainset.values, testset.values)
            if first_time:

                plt.subplot(211)
                plt.title("Example: Anomaly scores before first failure (close to continue).")
                plt.plot(anomaly_scores,label="Anomaly score")
                plt.legend()
                plt.subplot(212)
                plt.title("Example: Logged events before first failure.")
                for column in df.columns:
                    plt.plot(df[column],label=column)
                plt.legend()
                plt.show()
                first_time = False


            # Iteratevly for all data
            contimestamp = df.index[0]
            print(f"Calculate Contexts Episode {episode}:")
            for counter in tqdm(range(len(testset.index))):
                timestamp = testset.index[counter]
                score = anomaly_scores[counter]
                # row = dfdata.iloc[counter]

                # # collect df data:
                # if add_raw_to_context:
                #     for col in df_data.columns:
                #         context_gen.add_to_buffer(Eventpoint(code=col, details=row[col], source="m", timestamp=timestamp),
                #                                   replace=[])
                # Generate context data:
                contextdata = df.loc[contimestamp:timestamp]
                contimestamp = timestamp
                for index, row in contextdata.iterrows():
                    for col in df.columns:
                        context_gen.add_to_buffer(
                            Eventpoint(code=col, details=row[col], source="m", timestamp=index),
                            replace=[])

                context = context_gen.collect_data(timestamp=timestamp, source="m", name="score", value=score)
                context_list.append(context)

        context_gen.plot(context_list,filteredges=[["", "score", "increase"]])

        with open(f'context_list_{source}_{profile_hours}.pkl', 'wb') as file:
            pickle.dump(context_list, file)




def dsitance(contexta,contextb):
    similarity, _ = distance_3D_sbd_jaccard(contexta, contextb, a=0)
    return 1-similarity

def load_context():
    with open('./context_list.pkl', 'rb') as file:
        loaded_list = pickle.load(file)

    distance_Matrix=np.zeros((len(loaded_list),len(loaded_list) ))
    from tqdm import tqdm
    limit=len(loaded_list)
    limit=10000
    print("Calculate distances:")
    for i in tqdm(range(limit)):
        for j in range(limit):
            if j==i:
                distance_Matrix[i,j]=0
            elif i>j:
                distance_Matrix[i, j] = distance_Matrix[j, i]
            else:
                distance_Matrix[i,j]=dsitance(loaded_list[i],loaded_list[j])
    with open('distances.pkl', 'wb') as file:
        pickle.dump(distance_Matrix, file)



def clustering(distance_measure):
    from sklearn.cluster import DBSCAN
    from sklearn.cluster import AgglomerativeClustering

    predictions = AgglomerativeClustering( metric="precomputed",linkage="average",n_clusters=8).fit(distance_measure).labels_

    return predictions
from matplotlib import rc

# activate latex text rendering
# plt.rcParams.update({'font.size': 16})
#plt.gca().get_xaxis().get_major_formatter().set_useOffset(False)


def cluster_data(source,profile_hours):
    lendata=3000
    dftelemetry, dfcontext, failures = read_data_inner.AzureDataOneSource(source=source)
    dfs, isfailure = read_data_inner.split_df_with_failures_isfaile(dftelemetry, failures)
    dfs_context, _ = read_data_inner.split_df_with_failures_isfaile(dfcontext, failures)

    list_train, list_test, context_list_dfs, isfailure, all_sources = read_data_inner.Azure_generate_train_test(dfs,
                                                                                                                isfailure,
                                                                                                                dfs_context,
                                                                                                                [source
                                                                                                                 for i
                                                                                                                 in
                                                                                                                 dfs],
                                                                                                                period_or_count=f"{profile_hours} hours")

    dates=[]
    for ddf in list_test:
        dates.extend([dt for dt in ddf.index])

    with open(f'distances_{source}_{150}.pkl', 'rb') as file:
        distance_matrix = pickle.load(file)
        distance_matrix=distance_matrix#[:lendata]
        distance_matrix = distance_matrix#[:,:lendata]

    predictions=dbscan(distance_matrix)[1]
    failures=sorted(failures)
    with open(f'./context_list_{source}_{profile_hours}.pkl', 'rb') as file:
        loaded_list = pickle.load(file)
        c=0
        for fail in failures:
            if c==0:
                c+=1
                plt.axvline(fail,color="red",label="failure")
            else:
                plt.axvline(fail,color="red")
        labels = []
        fi=0
        for dt in dates[:len(predictions)]:
            if dt > failures[fi]-Timedelta(hours=96) and dt<failures[fi]:
                labels.append(1)
            else:
                labels.append(0)
            if dt > failures[fi]:
                fi=min(len(failures)-1,fi+1)
        plt.fill_between(dates[:len(predictions)], 0, 10,
                         where=labels, color="red", alpha=0.3, label="Near Failure State")
        # Plot the dendrogram
        colors=["blue","green","magenta","brown","red","orange","grey","black","lightgreen","lightblue","cyan","lightblue","white"]
        for i,pred in enumerate(list(set([pred for pred in predictions]))):
            toplot=[]
            dttoplot=[]
            edges={}
            for qi,dtt, pd in zip(range(len(predictions)),dates[:len(predictions)], predictions):
                if pd==pred:
                    for edge in loaded_list[qi].CR['edges']:
                        ed1 = edge[1].split("@")[0].replace("score","d_0")
                        ed2 = edge[0].split("@")[0].replace("score","d_0")
                        if edge[1]<edge[0]:
                            ed1=edge[0].split("@")[0].replace("score","d_0")
                            ed2=edge[1].split("@")[0].replace("score","d_0")
                        if f"{ed1}-{ed2}" in edges:
                            edges[f"{ed1}-{ed2}"]+=1
                        else:
                            edges[f"{ed1}-{ed2}"]=0
                    dttoplot.append(dtt)
                    toplot.append(pd)
            label_t=""
            c=0
            tups=[]
            for key in edges.keys():
                tups.append((key,edges[key]))
            tups=sorted(tups,key=lambda x: x[1],reverse=True)
            for tup in tups:
                c+=1
                if c>10:
                    break
                # strcount="\\textbf{"+f"{tup[1]}"+"}"
                strcount=f"{tup[1]}"
                label_t+=f"{tup[0]}: {strcount}, "
            plt.scatter(dttoplot,toplot,color=colors[i],label=label_t[:30])
        plt.yticks([0,1,2,3,4,5,6,7,8,9,10])
        plt.legend(facecolor='white', framealpha=1,prop={'size': 13},ncol=2,loc='upper center')
        plt.show()

if __name__ == '__main__':
    clustPipe=ClusteringPipeline()
    # calculate context
    clustPipe.generate_context(source=2,profile_hours=150)
    # calculate distance
    clustPipe.load_context(source=2,profile_hours=150)
    # calculate clustering
    print("Cluster contexts:")
    cluster_data(2,150)