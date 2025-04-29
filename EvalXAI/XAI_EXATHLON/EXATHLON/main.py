
import helper
from mydetector import TestExplainer,ModelSpecific
from ContextExplainer import ContextCaus
from Explainers import MYLimeGT,MYLime_MD,MYShapGT,MYShap_MD

import argparse


from pathlib import Path
Path("saved_models/").mkdir(exist_ok=True)
Path("myADoutput/").mkdir(exist_ok=True)

# Initialize the parser
parser = argparse.ArgumentParser(description="A script for calculating the importance of features for each sample in BATADAL dataset using one of SHAP, LIME or TEMPC XAI methods.")

# Add an argument
parser.add_argument("--method", help="XAI method to use",choices=["SHAP", "LIME", "TEMPC"],default="SHAP")

parser.add_argument("--gt", action='store_true', dest='gt',help="use ground truth as anomaly scores")
parser.add_argument('--no-gt', action='store_false', dest='gt', help='not use ground truth as anomaly scores')

parser.add_argument("--admodel", help="The anomaly detector to use if --no-gt",default="ocsvm",choices=["lof", "IF", "ocsvm"])
parser.add_argument("--coverage", help="Type of evaluation",default="center",choices=["all", "center"])

parser.add_argument("--shapthreshold", help="Threshold over importance to use with SHAP method",default=0.5)
parser.add_argument("--limek", help="Number of importance features to return when lime is used",default=5)
parser.add_argument("--causality", help="Which Causality discovery algorithm to use",default=None,choices=["PC", "MMPC", None])

parser.add_argument("--saved", action='store_true', dest='saved', help="use pre calculated explainer")
parser.add_argument('--no-saved', action='store_false',  dest='saved', help='do not use pre calculated explainer')



args = parser.parse_args()


if __name__ == '__main__':
    data= helper.get_test_data()
    datatrain= helper.get_train_data()

    experiment_name=""
    useground_truth=args.gt
    coverage=args.coverage
    causality_algorithm=args.causality
    savedmodel=args.saved
    modelname = args.method
    admodelname= args.admodel# lof, IF, ocsvm

    if useground_truth:
        experiment_name += "_gt"
        model = TestExplainer()
    else:
        # use ground truth
        experiment_name += f"_{admodelname}"
        # use svm as detector
        model = ModelSpecific(admodelname)

    ccc=0
    for kati in data["y_test"]:
        print(len(kati))
        ccc+=1
        #if ccc==2:
        #print(count_consecutive_positive(kati))

    if useground_truth:
        print("GT")
        model.fit_data_labels(data["test"],data["y_test"])
    else:
        model.fit_data(datatrain["train"])


    if modelname=="LIME":
        if useground_truth:
            explaiener=MYLimeGT(ad_model=model,sequence_size=1,features=int(args.limek),load_existing=savedmodel)
        else:
            explaiener=MYLime_MD(ad_model=model,sequence_size=1,features=int(args.limek))
    elif modelname=="SHAP":
        if useground_truth:
            explaiener=MYShapGT(ad_model=model,sequence_size=1,threshold=float(args.shapthreshold),load_existing=savedmodel)
        else:
            explaiener = MYShap_MD(ad_model=model, sequence_size=1,threshold=float(args.shapthreshold))
    elif modelname=="TEMPC":
        # Context Explainer
        explaiener=ContextCaus(ad_model=model,sequence_size=1,causality_algorithm=causality_algorithm)
        ok='ok'
    else:
        raise ValueError("Model name must be either SHAP, LIME or TEMPC")
    experiment_name = modelname+experiment_name


    explaiener.add_importance(data["test"])
    print("=== Calculation context Done======")
    #
    # data['test_scores']=model.getscores(data["test"])
    #
    # data['test_preds'] =model.getpreds(data["test"],th=0.95)
    #
    #
    helper.evaluationSingleTypeED(data, explaiener,args=args, ev_type='model_dependent', experiment_name=experiment_name, scores=False,coverage=coverage)

