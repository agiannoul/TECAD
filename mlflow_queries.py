import mlflow
import sys

from triton.language import dtype


def get_experiments_from(dataset_name,urll):
    mlflow.set_tracking_uri(urll)

    # Get a list of experiments
    client = mlflow.tracking.MlflowClient()
    experiments = client.search_experiments(filter_string=f"name LIKE '%{dataset_name}%'")
    return experiments

def get_max_of_metric(name="CMAPSS"):  # [("params.postprocessor","Default")]):
    # Connect to your MLflow tracking server
    experiments1 = get_experiments_from(name, urll="http://127.0.0.1:8080/")
    print("ok")
    dict={}
    Nones = []
    Halfs = []
    Zero = []
    for dataset in ["AZURE","SKAB","BATADAL"]:
        dict[dataset]={}
        indatasets=[exp for exp in experiments1 if dataset in exp.name]

        for classifier in ['xgb','svc']:
            dict[dataset][classifier]={}
            classf = [exp for exp in indatasets if classifier in exp.name]
            for method in ['KNN','IF',"OCSVM","CNN"]:
                dict[dataset][classifier]['KNN'] = {}
                epxmeth = [exp for exp in classf if method in exp.name]
                epxmeth=epxmeth[0]
                expid=epxmeth.experiment_id
                runs = mlflow.search_runs(expid)
                df_cleaned = runs.dropna(subset=['end_time'])
                df_sorted = df_cleaned.sort_values(by='end_time', ascending=False)

                Nonew=df_sorted[df_sorted["params.method_detector_weight"]=='None'].iloc[0]["metrics.AD1_AUC"]
                dict[dataset][classifier]['KNN']["None"] = Nonew
                Nones.append(Nonew)

                hlafw = df_sorted[df_sorted["params.method_detector_weight"] == '0.5'].iloc[0]["metrics.AD1_AUC"]
                dict[dataset][classifier]['KNN']["Halfs"] = hlafw
                Halfs.append(hlafw)

                zerow = df_sorted[df_sorted["params.method_detector_weight"] == '0'].iloc[0]["metrics.AD1_AUC"]
                dict[dataset][classifier]['KNN']["Zero"] = zerow
                Zero.append(zerow)
    from scipy.stats import wilcoxon
    print(dataset)
    print(Nones)
    print(Halfs)
    print(Zero)
    res = wilcoxon(Nones, Zero, alternative='greater')
    wins=sum([1 for NN, z in zip(Nones, Zero) if NN > z])
    loses=len(Nones) - wins
    print(f"None vs Zero: {res[1]}, {wins}, {loses}")

    res = wilcoxon(Nones, Halfs, alternative='greater')
    wins = sum([1 for NN, z in zip(Nones, Halfs) if NN > z])
    loses = len(Nones) - wins
    print(f"None vs Halfs: {res[1]}, {wins}, {loses}")
    print("-------------------------------------")

get_max_of_metric(name="CC-")