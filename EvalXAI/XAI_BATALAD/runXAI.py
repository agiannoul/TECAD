import pickle

import matplotlib.pyplot as plt
import pandas as pd
from PdmContext.ContextGeneration import ContextGenerator
from PdmContext.utils.causal_discovery_functions import calculate_with_pc
from PdmContext.utils.simulate_stream import simulate_stream
from tqdm import tqdm

from Explainers import MYLimeGT, MYShapGT

pd.set_option('display.max_columns', None)
import methods
def BATADAL_all():
    df = pd.read_csv('../../Data/BATADAL/test_1_2.csv', header=0, index_col=0)
    df.index = pd.to_datetime(df.index, format="%d/%m/%y %H")

    cols_values=[col for col in df.columns if "L_" in col or "P_" in col]
    F_cols = [col for col in df.columns if "F_" in col]
    cols_values.extend(F_cols)

    labels=[lb for lb in df['attack']]
    datalabels=[dt for dt in df.index]

    df.drop(["attack"],axis=1,inplace=True)

    df['date'] = [dt for dt in df.index]
    cols_values.append("date")
    dfvalues = df[cols_values]
    target_data = [dfvalues]

    target_sources = ["s"]
    

    date=[]
    source=[]
    description=[]
    type=[]

    for i in range(len(labels)-1):
        if labels[i] == 1 and labels[i+1] == 0:
            date.append(datalabels[i])
            source.append("s")
            description.append("attack")
            type.append("attack")

    event_data = pd.DataFrame(
        {"date": date, "type": type, "source": source, "description": description})
    event_data = event_data.sort_values(by='date')

    dataset={}

    dataset["max_wait_time"] = 120
    #dataset["raw_cols"] = [col for col in F_cols if "date" not in col]
    dataset["raw_cols"] = [col for col in cols_values if "date" not in col]
    dataset["anomaly_ranges"] = True
    dataset["label_dates"] = [datalabels]
    dataset["dates"] = "date"
    dataset["event_data"] = event_data
    dataset["target_data"] = target_data
    dataset["target_sources"] = target_sources
    dataset["historic_data"] = []
    dataset["historic_sources"] = []
    dataset["predictive_horizon"] = [labels]  # there are many episodes of length 1,2,3 ...
    dataset["slide"] = 3
    dataset["lead"] = [[0 for lb in labels]]
    dataset["beta"] = 1
    dataset["min_target_scenario_len"] = min(df.shape[0] for df in target_data)

    return dataset

    

def get_labels_explains(predict_data,context_feats,score,dates_score,dates_labels_dict):
    Timeseries = [("score", score, dates_score)]
    for col in context_feats:
        data_col = predict_data[col].values
        raw_data_triple = (col, [v for v in data_col], dates_score)
        Timeseries.append(raw_data_triple)

    stream = simulate_stream(Timeseries, [], [], "score")

    contextpipeline3 = ContextGenerator(target="score", context_horizon=f"{100} hours",
                                        Causalityfunct=calculate_with_pc)
    source = "s"
    all_records = []
    for record in stream:
        all_records.append(record)

    labels_to_return=[]
    important_features=[]

    for q in tqdm(range(len(all_records))):
        record = all_records[q]
        if isinstance(record["value"], str):
            print(record["name"])
        kati = contextpipeline3.collect_data(timestamp=record["timestamp"], source=source, name=record["name"],
                                             type=record["type"], value=record["value"])
        if kati is not None:
            labels_to_return.append(dates_labels_dict[record["timestamp"]])
            important_features.append(_get_single_importance(kati.CR["edges"]))
    contextpipeline3.plot([["", "score", ""]])
    return labels_to_return,important_features



def _get_single_importance(interlist):
    important_fts = []
    for pair in interlist:
        if "score"==pair[1]:
            important_fts.append(pair[0].split("@")[0])
    returned_fts = sorted(important_fts)
    return returned_fts


def generate_explainable_Context_pickle(filename):
    method = "gt"
    explainations = {
        "name": f"Context_{method}",
        "label": [],
        "important_features": []
    }

    dataset = BATADAL_all()
    labels = dataset["predictive_horizon"][0]
    data = dataset["target_data"][0]
    data.index = data[dataset["dates"]]
    data.drop([dataset["dates"]], axis=1, inplace=True)

    fit_data = data.iloc[:120]
    predict_data = data.iloc[120:]


    # score = methods.distance_based(fit_data.values,predict_data.values)
    if method=="gt":
        score = [1 if lb>0 else 0 for lb in labels[-len(predict_data.index):]]
    else:
        score = methods.distance_based(fit_data.values, predict_data.values)
    dates_score = [dt for dt in predict_data.index]

    dates_labels_dict={}
    for dt,lb in zip(dates_score,labels[-len(predict_data.index):]):
        dates_labels_dict[dt]=lb

    labels_to_return,important_features=get_labels_explains(predict_data, dataset["raw_cols"], score, dates_score, dates_labels_dict)
    explainations["label"]=labels_to_return
    explainations["important_features"]=important_features

    with open(f"Pickles/{filename}", 'wb') as file:
        pickle.dump(explainations, file)


def generate_lime(filename,limek=5):



    method="gt"
    explainations = {
        "name": f"Lime_{method}",
        "label": [],
        "important_features": []
    }

    dataset = BATADAL_all()
    labels = dataset["predictive_horizon"][0]
    data = dataset["target_data"][0]
    data.index = data[dataset["dates"]]
    data.drop([dataset["dates"]], axis=1, inplace=True)

    fit_data = data.iloc[:120]
    predict_data = data.iloc[120:]


    # score = methods.distance_based(fit_data.values,predict_data.values)
    if method=="gt":
        score = [1 if lb>0 else 0 for lb in labels[-len(predict_data.index):]]
    else:
        score = methods.distance_based(fit_data.values, predict_data.values)
    dates_score = [dt for dt in predict_data.index]

    dates_labels_dict={}
    for dt,lb in zip(dates_score,labels[-len(predict_data.index):]):
        dates_labels_dict[dt]=lb

    limeexpl = MYLimeGT(predict_data.values,score,limek)
    all_impostancies=limeexpl.add_importance(predict_data.values,score)
    labels=labels[-len(all_impostancies):]

    explainations["label"] = labels
    explainations["important_features"] = all_impostancies

    with open(f"Pickles/{filename}", 'wb') as file:
        pickle.dump(explainations, file)



def generate_shap(filename,threshold=0.1):
    method = "gt"
    explainations = {
        "name": f"Shap_{method}",
        "label": [],
        "important_features": []
    }

    dataset = BATADAL_all()
    labels = dataset["predictive_horizon"][0]
    data = dataset["target_data"][0]
    data.index = data[dataset["dates"]]
    data.drop([dataset["dates"]], axis=1, inplace=True)

    fit_data = data.iloc[:120]
    predict_data = data.iloc[120:]


    # score = methods.distance_based(fit_data.values,predict_data.values)
    if method=="gt":
        score = [1 if lb>0 else 0 for lb in labels[-len(predict_data.index):]]
    else:
        score = methods.distance_based(fit_data.values, predict_data.values)
    dates_score = [dt for dt in predict_data.index]

    dates_labels_dict={}
    for dt,lb in zip(dates_score,labels[-len(predict_data.index):]):
        dates_labels_dict[dt]=lb

    limeexpl = MYShapGT(predict_data.values,score,threshold)
    all_impostancies=limeexpl.add_importance(predict_data.values)
    labels=labels[-len(all_impostancies):]

    explainations["label"] = labels
    explainations["important_features"] = all_impostancies

    with open(f"Pickles/{filename}", 'wb') as file:
        pickle.dump(explainations, file)

import sys
import argparse

# Initialize the parser
parser = argparse.ArgumentParser(description="A script for calculating the importance of features for each sample in BATADAL dataset using one of SHAP, LIME or TEMPC XAI methods.")

# Add an argument
parser.add_argument("--method", help="XAI method to use",choices=["SHAP", "LIME", "TEMPC"],default="LIME")
parser.add_argument("--filename", help="The file name of pickle file for storing important features of each sample",default="lime.pickle")
parser.add_argument("--shapthreshold", help="Threshold over importnance to use with SHAP method",default=0.1)
parser.add_argument("--limek", help="Number of importance features to return when lime is used",default=5)

# Parse the arguments
args = parser.parse_args()
if __name__ == "__main__":
    # Get the first command-line argument
    method =args.method
    filename = args.filename
    if method=="SHAP":
        generate_shap(filename,float(args.shapthreshold))
    elif method=="LIME":
        generate_lime(filename,int(args.limek))
    elif method=="TEMPC":
        generate_explainable_Context_pickle(filename)
    else:
        print(f" Available XAI methods: SHAP, LIME, TEMPC")
