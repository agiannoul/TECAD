from scipy.stats import entropy


def calculate_metrics(data,labels,explanations):
    alltypes=list(set(labels))
    results={}
    for an_type in alltypes:
        if an_type==0:
            continue
        important_features=[]
        results[an_type]={}
        for i in range(len(labels)):
            if labels[i]==an_type:
                important_features.append(explanations[i])
        results[an_type]["consinces"]=consinces_of_a_group(important_features)
        results[an_type]["norm_ED2_concistancy"]=norm_ED2_consistency(important_features)
        results[an_type]["prop_ex"]=Pop_Exaplained_of_a_group(important_features)
        results[an_type]["concistancy"]=consistency_of_a_group(important_features)
        print(f'Type {an_type}: {results[an_type]["norm_ED2_concistancy"]:.2f} & {results[an_type]["prop_ex"]:.2f} & {results[an_type]["consinces"]:.2f}')

    important_features=[]
    results["global"]={}
    for i in range(len(labels)):
        if labels[i]>0:
            important_features.append(explanations[i])
    results["global"]["consinces"]=consinces_of_a_group(important_features)
    results["global"]["norm_ED2_concistancy"]=norm_ED2_consistency(important_features)
    results["global"]["prop_ex"]=Pop_Exaplained_of_a_group(important_features)
    results["global"]["concistancy"]=consistency_of_a_group(important_features)


    for key in results:
        print(f"Type {key}")
        for metric in results[key]:
            print(f"\t{metric} : {results[key][metric]}")


    return results

def Pop_Exaplained_of_a_group(important_features):
    c=0
    for list_ft in important_features:
        if len(list_ft)>=1:
            c+=1
    return c/len(important_features)
def consinces_of_a_group(important_features):
    cardinallity=0
    c=0
    for explanation in important_features:
        cardinallity+=len(explanation)
        if len(explanation)>=1:
            c+=1
    cardinallity=cardinallity/c
    return cardinallity

def consistency_of_a_group(important_features):
    features_bag = [ft for features_list in important_features for ft in features_list]
    # unnormalized probability distribution of feature ids
    p_features = []
    for feature_id in set(features_bag):
        p_features.append(features_bag.count(feature_id))
    return entropy(p_features, base=2)

def norm_ED2_consistency(important_features):
    explanations_fts = important_features
    fts_consistency = consistency_of_a_group(explanations_fts)
    #avg_explanation_length = sum([len(fts) for fts in explanations_fts]) / len(important_features)
    avg_explanation_length=consinces_of_a_group(important_features)
    return (2 ** fts_consistency) / avg_explanation_length




import pickle
def evaluate_filemame(filename):
    with open(f'Pickles/{filename}', 'rb') as file:
        explains = pickle.load(file)
        labels=explains["label"]
        print(labels[0])
        important_features=explains["important_features"]
        #important_features=[imp[0]['important_fts'] for imp in important_features]
        print(important_features[0])
        #for q in range(len(important_features)):
        #    if labels[q]>0:
        #        print(important_features[q])
        calculate_metrics([], labels, important_features)



# with open('Pickles/Shap_gt.pickle', 'rb') as file:
#     explains = pickle.load(file)
#     labels=explains["label"]
#     print(labels[0])
#     important_features=explains["important_features"]
#     important_features=[imp[0]['important_fts'] for imp in important_features]
#     print(important_features[0])
#     #for q in range(len(important_features)):
#     #    if labels[q]>0:
#     #        print(important_features[q])
#     calculate_metrics([], labels, important_features)

# Initialize the parser
import argparse


parser = argparse.ArgumentParser(description="A script for calculating ED metrics over calculated importance, loaded from pickle files")

# Add an argument
parser.add_argument("--filename", help="The file name of pickle file for storing important features of each sample",default="lime.pickle")

# Parse the arguments
args = parser.parse_args()
if __name__ == "__main__":
    # Get the first command-line argument
    evaluate_filemame(args.filename)
