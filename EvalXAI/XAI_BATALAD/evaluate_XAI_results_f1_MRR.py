import pickle

import matplotlib.pyplot as plt
import pandas as pd



def Type_7_example(type_l=3):
    Types_feats={
        1:["L_T7","PU10","PU11"],
        3:["L_T1","PU1","PU2"],
        5:["PU7","L_T4"],
        7:["L_T1","PU1","PU2"],
        8:["L_T3","PU4","PU5"],
        9:["L_T2","PU4","PU5"],
        10:["PU3"],
        12:["L_T2","P_J14","P_J422"],
        13:["L_T7","P_J14","P_J422","PU10","PU11"],
        14:["L_T4","L_T6"],
    }

    df = pd.read_csv('../../Data/BATADAL/test_1_2.csv', header=0, index_col=0)
    df.index = pd.to_datetime(df.index, format="%d/%m/%y %H")

    cols_values = [col for col in df.columns if "L_" in col or "P_" in col]
    F_cols = [col for col in df.columns if "F_" in col]
    cols_values.extend(F_cols)
    dfvalues = df[cols_values]
    labels = [lb for lb in df['attack']]
    datalabels = [dt for dt in df.index]

    df.drop(["attack"], axis=1, inplace=True)

    explains=get_explanations(dfvalues.columns)
    pos=0
    for i in range(len(labels)):
        if labels[i]==type_l:
            pos=i
            break
    print(pos)
    #df_normalized = (dfvalues - dfvalues.min()) / (dfvalues.max() - dfvalues.min())
    df_normalized = (dfvalues - dfvalues.mean()) / dfvalues.std()
    #df_normalized = dfvalues

    df_toplot=df_normalized.iloc[pos-100:pos+100]
    lab_toplot=labels[pos-100:pos+100]
    dates=datalabels[pos-100:pos+100]
    ft_imp=explains["shap"]["important_features"][pos-100-120:pos+100-120]

    explain_plot(df_toplot, ft_imp, lab_toplot, dates)

    plt.legend(ncol=5)
    plt.show()

def explain_plot(df_toplot,period_fts,label_period,dates):
    # Generate 20 distinct colors
    colors1 = plt.cm.get_cmap('tab20', 20)  # 'tab20' colormap provides 20 distinct colors

    # Convert colormap to list of colors
    all_colors = [colors1(i%20) for i in range(40)]

    plt.fill_between(dates, df_toplot.min().min(), df_toplot.max().max(), where=label_period,
                     color="red", alpha=0.3, label=f"Attack")

    context_ft = list(
        set([ft for listft in period_fts for ft in listft]))

    positions = {}
    colors = {}
    counter = {}
    for qi, ft in enumerate(context_ft):
        colors[ft] = all_colors[qi]
        positions[ft] = []
        counter[ft]=0
    for i, listft in enumerate(period_fts):
        if label_period[i] > 0:
            print(listft)
        for ft in listft:
            positions[ft].append(i)
    for col in df_toplot:
        if col in context_ft:
            plt.plot(dates, df_toplot[col].values,linewidth=3, color=colors[col], label=col)


    for col in df_toplot:
        if col in context_ft:
            continue
        else:
            plt.plot(dates, df_toplot[col].values, color="grey", label=col,alpha=0.5)

    for col in df_toplot:
        if col in context_ft:
            plt.plot(dates, df_toplot[col].values, color=colors[col])


    tuplelist=[]
    for ft in context_ft:
        vals=[v for v in df_toplot[ft].values]
        v_to_plot=[]
        d_to_plot=[]
        for qi in positions[ft]:
            if label_period[qi] > 0:
                v_to_plot.append(vals[qi])
                d_to_plot.append(dates[qi])
        counter[ft]=len(v_to_plot)
        tuplelist.append((ft,len(v_to_plot)))
        plt.scatter(d_to_plot,v_to_plot,color=colors[ft],edgecolors='black',s=100,label=ft,zorder=2)
    sorted_list = sorted(tuplelist, key=lambda x: x[1], reverse=True)
    for tup in sorted_list:
        print(f"{tup[0]}: {tup[1]}")

def get_explanations(colnames):
    with open('pickle_test/lime_gt_5.pickle', 'rb') as file:
        explains = pickle.load(file)
        labels=explains["label"]
        important_features=explains["important_features"]
        allfts = []
        for ftlist in important_features:
            allfts.append([colnames[int(ft)] for ft in ftlist])
        explanations={"lime":{"labels":labels,"important_features":allfts}}

    with open('pickle_test/Shap_gt.pickle', 'rb') as file:
        explains = pickle.load(file)
        labels=explains["label"]
        important_features=explains["important_features"]
        important_features=[imp[0]['important_fts'] for imp in important_features]
        allfts=[]
        for ftlist in important_features:
            allfts.append([colnames[int(ft)] for ft in ftlist])
        explanations["shap"]= {"labels": labels, "important_features": allfts}
    with open('pickle_test/Context_gt.pickle', 'rb') as file:
        explains = pickle.load(file)
        labels = explains["label"]
        important_features = explains["important_features"]
        explanations["context"]= {"labels": labels, "important_features": important_features}
    return explanations


def precision(predicted_features, true_features):
    """
    Calculate precision.

    Parameters:
    - predicted_features: List or set of predicted features.
    - true_features: List or set of true features.

    Returns:
    - Precision score.
    """
    predicted_set = set(predicted_features)
    true_set = set(true_features)

    true_positive = len(predicted_set.intersection(true_set))
    if len(predicted_set) == 0:
        return 0.0  # Avoid division by zero
    return true_positive / len(predicted_set)


def recall(predicted_features, true_features):
    """
    Calculate recall.

    Parameters:
    - predicted_features: List or set of predicted features.
    - true_features: List or set of true features.

    Returns:
    - Recall score.
    """
    predicted_set = set(predicted_features)
    true_set = set(true_features)

    true_positive = len(predicted_set.intersection(true_set))
    return true_positive / len(true_set) if len(true_set) > 0 else 0.0  # Avoid division by zero


def f1_score(predicted_features, true_features):
    """
    Calculate F1 score.

    Parameters:
    - predicted_features: List or set of predicted features.
    - true_features: List or set of true features.

    Returns:
    - F1 score.
    """
    prec = precision(predicted_features, true_features)
    rec = recall(predicted_features, true_features)

    if (prec + rec) == 0:
        return 0.0  # Avoid division by zero
    return 2 * (prec * rec) / (prec + rec)


def Fscore(type_l=3):
    Types_feats={
        1:["L_T7","PU10","PU11"],
        3:["L_T1","PU1","PU2"],
        5:["PU7","L_T4"],
        7:["L_T1","PU1","PU2"],
        8:["L_T3","PU4","PU5"],
        9:["L_T2","PU4","PU5"],
        10:["PU3"],
        12:["L_T2","P_J14","P_J422"],
        13:["L_T7","P_J14","P_J422","PU10","PU11"],
        14:["L_T4","L_T6"],
    }
    df = pd.read_csv('../../Data/BATADAL/test_1_2.csv', header=0, index_col=0)
    df.index = pd.to_datetime(df.index, format="%d/%m/%y %H")

    cols_values = [col for col in df.columns if "L_" in col or "P_" in col]
    F_cols = [col for col in df.columns if "F_" in col]
    cols_values.extend(F_cols)
    explanations=get_explanations(cols_values)

    results={}
    for type_an in Types_feats.keys():
        results[type_an] = {}
        print(f"Type: {type_an}")
        for method in explanations.keys():
            avg_f1=[]
            for lb,expl in zip(explanations[method]["labels"],explanations[method]["important_features"]):
                if str(lb)!=str(type_an):
                    continue
                prec=precision(expl, Types_feats[type_an])
                rec=recall(expl, Types_feats[type_an])
                f1=f1_score(expl, Types_feats[type_an])
                avg_f1.append(f1)
            results[type_an][method] = sum(avg_f1) / len(avg_f1) if avg_f1 else 0.0
            print(f"{method}: {results[type_an][method]}")


def MRR():
    Types_feats = {
        1: ["L_T7", "PU10", "PU11"],
        3: ["L_T1", "PU1", "PU2"],
        5: ["PU7", "L_T4"],
        7: ["L_T1", "PU1", "PU2"],
        8: ["L_T3", "PU4", "PU5"],
        9: ["L_T2", "PU4", "PU5"],
        10: ["PU3"],
        12: ["L_T2", "P_J14", "P_J422"],
        13: ["L_T7", "P_J14", "P_J422", "PU10", "PU11"],
        14: ["L_T4", "L_T6"],
    }
    df = pd.read_csv('../../Data/BATADAL/test_1_2.csv', header=0, index_col=0)
    df.index = pd.to_datetime(df.index, format="%d/%m/%y %H")

    cols_values = [col for col in df.columns if "L_" in col or "P_" in col]
    F_cols = [col for col in df.columns if "F_" in col]
    cols_values.extend(F_cols)
    explanations = get_explanations(cols_values)

    results = {}
    for type_an in Types_feats.keys():
        results[type_an] = {}
        print(f"T{type_an}",end=" &")
        for method in explanations.keys():
            predicted=[]
            true_feats=[]
            for lb, expl in zip(explanations[method]["labels"], explanations[method]["important_features"]):
                if str(lb) != str(type_an):
                    continue
                predicted.append(expl)
                true_feats.append(Types_feats[type_an])
            results[type_an][method] = mean_reciprocal_rank(predicted, true_feats)
            print(f"{method}: {results[type_an][method]}",end=" &")

def MRR_and_f1():
    Types_feats = {
        1: ["L_T7", "PU10", "PU11"],
        3: ["L_T1", "PU1", "PU2"],
        5: ["PU7", "L_T4"],
        7: ["L_T1", "PU1", "PU2"],
        8: ["L_T3", "PU4", "PU5"],
        9: ["L_T2", "PU4", "PU5"],
        10: ["PU3"],
        12: ["L_T2", "P_J14", "P_J422"],
        13: ["L_T7", "P_J14", "P_J422", "PU10", "PU11"],
        14: ["L_T4", "L_T6"],
    }
    df = pd.read_csv('../../Data/BATADAL/test_1_2.csv', header=0, index_col=0)
    df.index = pd.to_datetime(df.index, format="%d/%m/%y %H")

    cols_values = [col for col in df.columns if "L_" in col or "P_" in col]
    F_cols = [col for col in df.columns if "F_" in col]
    cols_values.extend(F_cols)
    explanations = get_explanations(cols_values)

    results = {}
    for type_an in Types_feats.keys():
        results[type_an] = {}
        print(f"T{type_an}",end=" &")
        for method in ["lime","shap","context"]:
            predicted=[]
            true_feats=[]
            for lb, expl in zip(explanations[method]["labels"], explanations[method]["important_features"]):
                if str(lb) != str(type_an):
                    continue
                predicted.append(expl)
                true_feats.append(Types_feats[type_an])
            avg_f1 = []
            for lb, expl in zip(explanations[method]["labels"], explanations[method]["important_features"]):
                if str(lb) != str(type_an):
                    continue
                prec = precision(expl, Types_feats[type_an])
                rec = recall(expl, Types_feats[type_an])
                f1 = f1_score(expl, Types_feats[type_an])
                avg_f1.append(f1)
            f1 = sum(avg_f1) / len(avg_f1) if avg_f1 else 0.0
            mrr = mean_reciprocal_rank(predicted, true_feats)
            if method == "context":
                print(f"{f1:.8f} & {mrr:.8f}", end="\\\\ \n\\hline\n")
            else:
                print(f"{f1:.8f} & {mrr:.8f}",end=" &")



def MRR_and_f1_perType():
    Types_feats = {
        1: ["L_T7", "PU10", "PU11"],
        3: ["L_T1", "PU1", "PU2"],
        5: ["PU7", "L_T4"],
        7: ["L_T1", "PU1", "PU2"],
        8: ["L_T3", "PU4", "PU5"],
        9: ["L_T2", "PU4", "PU5"],
        10: ["PU3"],
        12: ["L_T2", "P_J14", "P_J422"],
        13: ["L_T7", "P_J14", "P_J422", "PU10", "PU11"],
        14: ["L_T4", "L_T6"],
    }
    df = pd.read_csv('../../Data/BATADAL/test_1_2.csv', header=0, index_col=0)
    df.index = pd.to_datetime(df.index, format="%d/%m/%y %H")

    cols_values = [col for col in df.columns if "L_" in col or "P_" in col]
    F_cols = [col for col in df.columns if "F_" in col]
    cols_values.extend(F_cols)
    explanations = get_explanations(cols_values)

    results = {}
    for type_an in Types_feats.keys():
        results[type_an] = {}
        print(f"T{type_an}",end=" &")
        for method in ["lime","shap","context"]:
            predicted=[]
            true_feats=[]

            all_exp = []
            for lb, expl in zip(explanations[method]["labels"], explanations[method]["important_features"]):
                if str(lb) != str(type_an):
                    continue
                all_exp.extend(expl)
            f1 = f1_score(set(all_exp), Types_feats[type_an])
            count={}
            for ft in all_exp:
                count[ft]=0
            for lb, expl in zip(explanations[method]["labels"], explanations[method]["important_features"]):
                if str(lb) != str(type_an):
                    continue
                for ft in expl:
                    count[ft]+=1
            sorted_keys = sorted(count, key=count.get, reverse=True)
            mrr = mean_reciprocal_rank([sorted_keys], [Types_feats[type_an]])
            if method == "context":
                print(f"{f1:.2f} & {mrr:.2f}", end="\\\\ \n\\hline\n")
            else:
                print(f"{f1:.2f} & {mrr:.2f}",end=" &")

def mean_reciprocal_rank(predicted_list, true_list):
    """
    Calculate Mean Reciprocal Rank (MRR).

    Parameters:
    - predicted_list: List of lists of predicted features ranked for each instance.
    - true_list: List of lists of true features.

    Returns:
    - MRR score.
    """
    reciprocal_ranks = []

    for predicted_features, true_features in zip(predicted_list, true_list):
        predicted_set = set(predicted_features)
        # Find the rank of the first relevant feature
        rank = next((i + 1 for i, feature in enumerate(predicted_features) if feature in true_features), 0)
        reciprocal_ranks.append(1 / rank if rank > 0 else 0)

    # Calculate the average MRR
    return sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0


MRR_and_f1()
MRR_and_f1_perType()
# Type_7_example()