import pandas as pd

def type_value(col,coln,df):
    value=df[col].values[0]
    type = col.split(coln)[0]
    return type,value

def show_results(filename,name):
    df = pd.read_csv(filename,header=0)
    #df = pd.read_csv("all_until_now_and_shap.csv",header=0)
    df=df[df["method"]==name]


    col1="ED1_NORM_CONSISTENCY"
    col2="PROP_EXPLAINED"
    col3="ED2_CONCISENESS"

    granularity="global"


    df=df[df["granularity"]==granularity]
    #print(df.head())

    results={}

    for col in df.columns:
        if col1 in col:
            type,value=type_value(col, col1, df)
            if type not in results:
                results[type]={}
            results[type][col1]=value
        elif col2 in col:
            type,value=type_value(col, col2, df)
            if type not in results:
                results[type]={}
            results[type][col2]=value
        elif col3 in col:
            type,value=type_value(col, col3, df)
            if type not in results:
                results[type]={}
            results[type][col3]=value

    #print(results)
    for key in results:
        if key == "TEST_GLOBAL_":
            continue
        print(f'Type {key}: {results[key][col1]:.2f} & {results[key][col2]:.2f} & {results[key][col3]:.2f}')

import argparse

# Initialize the parser
parser = argparse.ArgumentParser(description="A script for Exathlon results")


parser.add_argument("--method", help="XAI method as stored in the results",default="LIME")
parser.add_argument("--filename", help="Path to the file of Exathlon output",default="ExathlonResult.csv")

args = parser.parse_args()

if args.method == "TEMPC":
    name="Context_gt"
else:
    name=args.method
filename=args.filename
show_results(filename,name)
