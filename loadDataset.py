import os
import sys
import pandas as pd
from pdm_evaluation_types.types import EventPreferences, EventPreferencesTuple


def step(df,cols_values,anomaly_col,source_name):
    labels = [1 if lb > 0 else 0 for lb in df[anomaly_col]]
    datalabels = [dt for dt in df.index]

    df.drop([anomaly_col], axis=1, inplace=True)

    dfvalues = df[cols_values]
    target_data = [dfvalues]
    target_sources = [source_name]

    date = []
    source = []
    description = []
    type = []

    for i in range(len(labels) - 1):
        if labels[i] == 1 and labels[i + 1] == 0:
            date.append(datalabels[i])
            source.append(source_name)
            description.append(anomaly_col)
            type.append(anomaly_col)
    return datalabels,labels,date,source,description,type,target_data,target_sources
def SKAB():
    target_data = []
    target_sources = []
    historic_data = []
    historic_sources = []

    date = []
    source = []
    description = []
    type = []
    labels = []
    cols_values=["datetime","Accelerometer1RMS","Accelerometer2RMS","Current","Pressure","Temperature","Thermocouple","Voltage","Volume Flow RateRMS"]
    anomaly_col="anomaly"
    source_name=0
    folder="./Data/valve1"

    label_dates=[]
    for i in range(16):
        df=pd.read_csv(folder+f"/{i}.csv",sep=";")
        df["datetime"]=pd.to_datetime(df["datetime"])
        df.index=df["datetime"]
        # print(df.head())
        datalabelst,labelst,datet,sourcet,descriptiont,typet,target_dataT,target_sourcesT=step(df, cols_values, anomaly_col, str(source_name))

        labels.append(labelst)
        label_dates.append(datalabelst)

        date.extend(datet)
        source.extend(sourcet)
        description.extend(descriptiont)
        type.extend(typet)

        source_name+=1

        target_data.extend(target_dataT)
        target_sources.extend(target_sourcesT)

    event_data = pd.DataFrame(
        {"date": date, "type": type, "source": source, "description": description})
    event_data=event_data.sort_values(by='date')

    event_preferences: EventPreferences = {
        'failure': [
            EventPreferencesTuple(description='*', type=anomaly_col, source='*', target_sources='=')
        ],
        'reset': [
            EventPreferencesTuple(description='*', type=anomaly_col, source='*', target_sources='='),
        ]
    }


    dataset = {}
    dataset["max_wait_time"] = 200
    dataset["raw_cols"] = [col for col in cols_values if "date" not in col]  # Fcols
    dataset["all_cols"] = [col for col in cols_values if "date" not in col]
    dataset["anomaly_ranges"] = True
    dataset["label_dates"] = label_dates
    dataset["dates"] = "datetime"
    dataset["event_preferences"] = event_preferences
    dataset["event_data"] = event_data
    dataset["target_data"] = target_data
    dataset["target_sources"] = target_sources
    dataset["historic_data"] = []
    dataset["historic_sources"] = []
    dataset["predictive_horizon"] = labels # there are many episodes of length 1,2,3 ...
    dataset["slide"] = 3
    dataset["lead"] = [[0 for lb in labelinner] for labelinner in labels]
    dataset["slide"] = 3
    dataset["beta"] = 1
    dataset["min_historic_scenario_len"] = sys.maxsize
    dataset["min_target_scenario_len"] = min(df.shape[0] for df in target_data)
    return dataset

def azure():
    dftelemetry = pd.read_csv("./Data/azure/PdM_telemetry.csv", header=0)
    dfmainentance = pd.read_csv("./Data/azure/PdM_maint.csv", header=0)
    dferrors = pd.read_csv("./Data/azure/PdM_errors.csv", header=0)
    dffailures = pd.read_csv("./Data/azure/PdM_failures.csv", header=0)
    dftelemetry["machineID"]=[str(m_id) for m_id in dftelemetry["machineID"].values]
    dfmainentance["machineID"]=[str(m_id) for m_id in dfmainentance["machineID"].values]
    dferrors["machineID"]=[str(m_id) for m_id in dferrors["machineID"].values]
    dffailures["machineID"]=[str(m_id) for m_id in dffailures["machineID"].values]

    #print(dferrors.head())
    #print(dfmainentance.head())
    #print(dffailures.head())

    #print(dftelemetry.head())

    event_data = pd.DataFrame(columns=["date", "type", "source", "description"])

    dferrors=dferrors.rename({'datetime': 'date', 'machineID': 'source', 'errorID' : 'description'}, axis=1)
    dferrors['type']=["error" for i in dferrors.index]

    event_data=pd.concat([event_data,dferrors], ignore_index=True)

    dfmainentance = dfmainentance.rename({'datetime': 'date', 'machineID': 'source', 'comp': 'description'}, axis=1)
    dfmainentance['type'] = ["maintenance" for i in dfmainentance.index]

    event_data = pd.concat([event_data, dfmainentance], ignore_index=True)

    dffailures = dffailures.rename({'datetime': 'date', 'machineID': 'source', 'failure': 'description'}, axis=1)
    dffailures['type'] = ["failure" for i in dffailures.index]

    event_data = pd.concat([event_data, dffailures], ignore_index=True)

    event_data['date']=pd.to_datetime(event_data['date'])

    event_preferences: EventPreferences = {
        'failure': [
            EventPreferencesTuple(description='*', type='failure', source='*', target_sources='=')
        ],
        'reset': [
            #EventPreferencesTuple(description='*', type='reset', source='*', target_sources='='),
            EventPreferencesTuple(description='*', type='failure', source='*', target_sources='=')
        ]
    }

    historic_data = []
    historic_sources = []
    episode_lengths =[]
    target_data = []
    target_sources = []
    for machine_idi in range(1,20):
        machine_id=str(machine_idi)
        df = dftelemetry[dftelemetry["machineID"] == machine_id]
        df = df.drop(["machineID"],axis=1)
        df = df.rename({'datetime': 'timestamp'}, axis=1)
        df['timestamp']=pd.to_datetime(df['timestamp'])
        df=df.sort_values(by='timestamp')
        target_data.append(df)
        target_sources.append(machine_id)

        dftemp=df.copy()
        ailures_for_current_source = event_data[event_data["source"]==machine_id].sort_values(by='date')
        for failure_index, failure in ailures_for_current_source.iterrows():
            current_df = dftemp[dftemp['timestamp'] <= failure.date]
            if current_df.shape[0] > 0:
                episode_lengths.append(current_df.shape[0])
            # final_df = pd.concat([final_df, current_df])
            dftemp = dftemp[dftemp['timestamp'] > failure.date]
        #if dftemp.shape[0] > 0: # these are without failure
        #    episode_lengths.append(dftemp.shape[0])
    #print(episode_lengths)
            
    #print(f"Ph + slide: {min(episode_lengths)/3}")
    #print(f"Ph: {min(episode_lengths)/10}")
    #episode_lengths.sort()
    #print(episode_lengths)

    dataset = {}
    dataset["dates"] = "timestamp"
    dataset["event_preferences"] = event_preferences
    dataset["event_data"] = event_data
    dataset["target_data"] = target_data
    dataset["target_sources"] = target_sources
    dataset["historic_data"] = historic_data
    dataset["historic_sources"] = historic_sources
    dataset["predictive_horizon"] = "96 hours" # there are many episodes of length 1,2,3 ...
    dataset["slide"] = 96 
    dataset["lead"] = "2 hours"
    dataset["beta"] = 1
    dataset["min_historic_scenario_len"] = sys.maxsize
    dataset["min_target_scenario_len"] = min(df.shape[0] for df in target_data)
    dataset["max_wait_time"] = 150

    return dataset



def BATADAL():
    df = pd.read_csv('./Data/BATADAL/test_1_2.csv', header=0, index_col=0)
    df.index = pd.to_datetime(df.index, format="%d/%m/%y %H")
    todrop=[]
    for col in df.columns:
        if df[col].unique().shape[0] == 1:
            todrop.append(col)
    df.drop(todrop, axis=1, inplace=True)
    cols_values=[col for col in df.columns if "L_" in col or "P_" in col]
    F_cols = [col for col in df.columns if "F_" in col]
    cols_values.extend(F_cols)


    labels=[1 if lb>0 else 0 for lb in df['attack']]
    datalabels=[dt for dt in df.index]

    df.drop(["attack"],axis=1,inplace=True)

    df['date']=[dt for dt in df.index]
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


    event_preferences: EventPreferences = {
        'failure': [
            EventPreferencesTuple(description='*', type='attack', source='*', target_sources='=')
        ],
        'reset': [
            EventPreferencesTuple(description='*', type='attack', source='*', target_sources='='),
        ]
    }

    dataset={}


    dataset["max_wait_time"] = 200
    dataset["raw_cols"] = [ col for col in cols_values if "date" not in col] # Fcols
    dataset["all_cols"] = [ col for col in cols_values if "date" not in col]
    dataset["anomaly_ranges"] = True
    dataset["label_dates"] = [datalabels]
    dataset["dates"] = "date"
    dataset["event_preferences"] = event_preferences
    dataset["event_data"] = event_data
    dataset["target_data"] = target_data
    dataset["target_sources"] = target_sources
    dataset["historic_data"] = []
    dataset["historic_sources"] = []
    dataset["predictive_horizon"] = [labels]  # there are many episodes of length 1,2,3 ...
    dataset["slide"] = 3
    dataset["lead"] = [[0 for lb in labels]]
    dataset["beta"] = 1
    dataset["min_historic_scenario_len"] = sys.maxsize
    dataset["min_target_scenario_len"] = min(df.shape[0] for df in target_data)

    return dataset



def get_dataset(name="azure"):
    if name == "azure":
        return azure()
    elif name == "BATADAL":
        return BATADAL()
    elif name == "SKAB":
        return SKAB()
    else:
        print(f"No dataset with naem {name}")
