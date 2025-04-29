
import os
import warnings

import pandas as pd
from PdmContext.ContextGeneration import ContextGenerator
from PdmContext.utils.simulate_stream import simulate_from_df


from PdmContext.utils.dbconnector import SQLiteHandler
from PdmContext.Pipelines import ContextAndDatabase
# add absolute src directory to python path to import other project modules
import sys


src_path = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
sys.path.append(src_path)
from explanation.model_dependent.model_dependent_explainers import ModelDependentExplainer

from tqdm import tqdm


def my_causality_MMPC(names,data):
    from cdt.causality.graph import MMPC
    import networkx as nx

    #start=time.time()
    data_Df={}
    for name,dat in zip(names,data):
        data_Df[name]=dat

    df = pd.DataFrame(data_Df)
    df = df.astype(float)
    #print(f"time to df: {time.time()-start}")
    obj = MMPC()
    # obj = PC()

    learned_graph = obj.predict(df)
    MAPPING = {k: n for k, n in zip(range(len(names)), names)}
    learned_graph = nx.relabel_nodes(learned_graph, MAPPING, copy=True)
    edges = learned_graph.edges
    #print(f"time all: {time.time() - start}")
    return edges


class ContextCaus(ModelDependentExplainer):
    """Causality based Local Interpretable Model-agnostic Explanations explanation discovery class.

    For this explainer, the `fit` method must be called before explaining samples.

    See https://arxiv.org/pdf/1602.04938.pdf for more details.
    """
    def __init__(self, ad_model,sequence_size,causality_algorithm=None):
        #super().__init__(args, output_path, ad_model)
        # number of features to report in the explanations
        self.ad_model=ad_model
        # LIME model
        self.lime_model = None
        self.sample_length=sequence_size

        self.small_anomalies_expansion = 'before'
        # coverage policy for anomalies larger than sample length
        self.large_anomalies_coverage = 'center'
        self.causality_algorithm=causality_algorithm
        self.all_importance={}
        if self.causality_algorithm == 'MMPC':
            self.congen=ContextGenerator(target="scores",Causalityfunct=my_causality_MMPC)
        else:
            self.congen = ContextGenerator(target="scores")
    def _count_consecutive_positive(self,numbers):
        consecutive_count = 0
        max_consecutive_count = 0

        for num in numbers:
            if num > 0:
                consecutive_count += 1
                max_consecutive_count = max(max_consecutive_count, consecutive_count)
            else:
                consecutive_count = 0

        return max_consecutive_count
    def _get_single_importance(self,interlist):
        important_fts = []

        for pair in interlist:
            if "scores"==pair[1]:
                important_fts.append(pair[0].split("@")[0])

        returned_fts = sorted(important_fts)
        if len(returned_fts) == 0:
            warnings.warn('No explanation found for the sample.')
            return {'important_fts': []}, 1
        return {'important_fts': returned_fts}, 1

    def run_simulation(self,df,start2 = pd.to_datetime("2023-01-01 00:00:00")):

        maxcon=self._count_consecutive_positive(df["scores"].values)
        timestamps2 = [start2 + pd.Timedelta(hours=i) for i in range(len(df.index))]
        df.index = timestamps2
        if self.causality_algorithm == 'MMPC':
            self.congen = ContextGenerator( target="scores",context_horizon=f"{min(len(df.index),2*maxcon)}",Causalityfunct=my_causality_MMPC)
        else:
            self.congen = ContextGenerator( target="scores",context_horizon=f"{min(len(df.index),2*maxcon)}")
        database = SQLiteHandler(db_name=f"ContextDatabase{self.ad_model.get_name()}.db")
        self.contextpipeline = ContextAndDatabase(context_generator_object=self.congen, databaseStore_object=database)
        stream = simulate_from_df(df, [], "scores")
        source = "ignore"
        print(len(df.index)*len(df.columns))

        allrecords = [record for record in stream]
        for i in tqdm(range(len(allrecords))):
            record = allrecords[i]
            # print(record)
            self.contextpipeline.collect_data(timestamp=record["timestamp"], source=source, name=record["name"],
                                     type=record["type"], value=record["value"])
        return self.congen.contexts

    def _load_from_base(self,test):
        database = SQLiteHandler(db_name=f"ContextDatabase{self.ad_model.get_name()}.db")
        traget_name = "scores"
        counter=0

        notincluded=[(-1.3891040924821167, -0.4560007562141696, -0.45942079792695595, -5.639152619988776, -5.565439766876459, -5.564848744327601, -6.089135357136732, -3.489566220271544, -1.313437836224082, -0.8257094834054192, -1.7480051322185397, -0.8095472908778482, -0.019339582277197065, -5.799428577035019, -6.522026825911004, -6.3157049839892725, -5.908270891616711, -7.1645692977471285, -0.6658202907171984)
                    ,(0.02737599205396835, -0.07747195679913634, -0.011780568903368522, 0.08265890046615143, 0.08160507740236815, 0.09429589314680044, -0.23682215386017408, 0.3611080934888116, -1.3301603442645327, 0.9035040396091958, -0.6742493653274081, 0.5074296142564336, 0.5979577239008859, 0.04733918847799901, 0.05998601309995101, 0.05870453392103729, 0.03530209492399656, 0.043375033465563956, -0.46488923490657774)
                    ,(0.7063617036754537, -0.07599133659129872, 0.5643865167009697, 0.0984433125623956, 0.09657029898909789, 0.07668149510537481, 0.061071820670245434, -0.3970664145870894, 0.6996950967933099, -0.8381195643635777, -1.1554926459889905, -1.3455596568082315, -0.8369865314769606, 0.06805994395120674, 0.05887400183562828, 0.06647199496442066, 0.08607219557334367, 0.06802945529541778, 0.6798894618089181)
                    ,(-3.673956000594546, -0.07850839094462268, -2.782770724675217, -6.763829846281604, -6.743418840793429, -5.527787879766838, -3.037148902455321, -2.286401824719964, 0.6876843646613018, 1.2851826576565328, 1.3030870158365302, 0.6186089705203139, 0.3045377501121404, 0.04740197864609874, 0.062822521042864, 0.061297089512169806, 0.049357891202835784, 0.05669239463843046, 0.3552410027149769)
                    ,(-5.866149353109235, -0.0804331972148116, -5.007527933015077, -10.24034661048032, -10.209099066257652, -11.207062415135306, -3.236510468931572, -10.151595497186415, -1.4297838481275245, -1.377971303403505, -0.49970517026879446, -0.14508732191133805, -0.41762373788847656, 0.06379021252036368, 0.0669376265975301, 0.06661577569015444, 0.07117282821688886, 0.07383749760611256, 0.519988553233156)
                    ]
        contextlist = database.get_all_context_by_target(traget_name)
        if len(contextlist)>0:
            start2 =contextlist[0].timestamp
        else:
            start2 = pd.to_datetime("2023-01-01 00:00:00")
        #print(len(contextlist))
        amm=0
        for period in test:
            amm+=len(period)
        #print(amm)
        get_ready=True
        for qi,period in enumerate(test):
            temp_importances={}
            if get_ready:
                if len(period)==104:
                    continue
                try:
                    for i in range(len(period)):
                        record = period[i]
                        sc, reci = self.ad_model.get_record_score(record)
                        if reci in notincluded:
                            self.all_importance[reci]={'important_fts': []}, 1
                        else:
                            temp_importances[reci] = self._get_single_importance(contextlist[counter].CR["edges"])
                            counter+=1
                    start2 = contextlist[counter-1].timestamp + pd.Timedelta(hours=2)
                except:
                    print(reci)
                    temp_importances, start2 = self.hendle_period(period, start2)
                    get_ready = False
            else:
                temp_importances, start2 = self.hendle_period(period, start2)
                get_ready = False
            for key in temp_importances.keys():
                self.all_importance[key]=temp_importances[key]

    def hendle_period(self,period,start2):
        temp_importances={}
        scores = []
        recs = []
        for i in range(len(period)):
            record = period[i]
            sc, reci = self.ad_model.get_record_score(record)
            recs.append(reci)
            scores.append(sc)
        df = self._produce_df_with_score(period, scores)
        conlist = self.run_simulation(df, start2)
        print(len(conlist))
        start2 = conlist[-1].timestamp + pd.Timedelta(hours=2)
        for rec, cont in zip(recs, conlist):
            temp_importances[rec] = self._get_single_importance(cont.CR["edges"])
        return temp_importances,start2

    def add_importance(self,test):

        self._load_from_base(test)



    def _produce_df(self,sample):
        scores = []
        for record in sample:
            sc,_=self.ad_model.get_record_score(record)
            scores.append(sc)
        df = pd.DataFrame()
        for ftv in range(sample.shape[1]):
            sereis = sample[:, ftv]
            df[f"ft_{ftv}"] = sereis
        df["scores"]=scores
        return df

    def _produce_df_with_score(self,sample,score):

        df = pd.DataFrame()
        for ftv in range(sample.shape[1]):
            sereis = sample[:, ftv]
            df[f"ft_{ftv}"] = sereis
        df["scores"]=score
        return df


    def explain_sample(self, sample, sample_labels=None):
        if len(sample.shape)>1:
            reci=sample[-1]
        else:
            reci=sample
        rec = tuple([kati for kati in reci])
        if rec not in self.all_importance.keys():
            return {'important_fts': []}, 1
        return  self.all_importance[rec]

