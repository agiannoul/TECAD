import matplotlib.pyplot as plt
import pandas as pd

from tqdm import tqdm

# context imports
from PdmContext.ContextGeneration import ContextGenerator
from PdmContext.utils.causal_discovery_functions import calculate_with_pc

# methods and utils
import methods
from iforest import IForest
from utils import Window, find_length






from sand_core import SAND




def apply_SAND(data):
    slidingWindow = find_length(data)
    clf = SAND(pattern_length=slidingWindow,subsequence_length=4*(slidingWindow))
    x = data


    clf.fit(x,overlaping_rate=int(1.5*slidingWindow))
    X_data = Window(window = slidingWindow).convert(data).to_numpy()
    #x = X_data
    #clf = IForest(n_jobs=1)
    #clf.fit(x)
    score = clf.decision_scores_
    #score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
    return score
def apply_Dist(data):
    slidingWindow = find_length(data)
    X_data = Window(window=slidingWindow).convert(data).to_numpy()
    trainsize=int(0.1*len(X_data))
    score = methods.distance_based(X_data[:trainsize],X_data[trainsize:])
    #score = MinMaxScaler(feature_range=(0, 1)).fit_transform(score.reshape(-1, 1)).ravel()
    return score
def apply_lof(data):
    slidingWindow = find_length(data)
    X_data = Window(window=slidingWindow).convert(data).to_numpy()
    trainsizes = 1500
    trainsizee = 2200
    train = X_data[trainsizes:trainsizee]
    score = methods.lof_semi(train,X_data)
    #score = MinMaxScaler(feature_range=(0, 1)).fit_transform(score.reshape(-1, 1)).ravel()
    return score

def apply_ocsvm(data):
    slidingWindow = find_length(data)
    X_data = Window(window=slidingWindow).convert(data).to_numpy()
    trainsizes=1500
    trainsizee=2200
    train=X_data[trainsizes:trainsizee]
    score = methods.ocsvm_semi(train,X_data)
    #score = MinMaxScaler(feature_range=(0, 1)).fit_transform(score.reshape(-1, 1)).ravel()
    return score
def apply_isolation_fores_semi(data):
    slidingWindow = find_length(data)
    X_data = Window(window=slidingWindow).convert(data).to_numpy()
    trainsizes = 1500
    trainsizee = 2200
    train = X_data[trainsizes:trainsizee]
    score = methods.isolation_fores_semi(train, X_data)
    #score = MinMaxScaler(feature_range=(0, 1)).fit_transform(score.reshape(-1, 1)).ravel()
    return score


def read_data():
    dftemp=pd.read_csv("temp_data.csv",header=0,index_col=1)
    dftemp.index= pd.to_datetime(dftemp.index)
    dftemp=dftemp[["Mean.TemperatureF"]]

    dftemp.plot()


    print(dftemp.head())

    tempdata=dftemp["Mean.TemperatureF"].values.astype(float)
    extrem_weather=[1 if tm>90 or tm<25 else 0 for tm in tempdata]
    extrem_weather=[0]+[1 if extrem_weather[i-1]!=extrem_weather[i]  else 0 for i in range(1,len(extrem_weather))]

    plt.plot(dftemp.index,extrem_weather)


    tempdata=extrem_weather

    df1 = pd.read_csv("nyc_taxi.csv",index_col=0)
    df1.index= pd.to_datetime(df1.index)


    df1.plot()
    plt.show()

    df=df1.to_numpy()

    data = df[:, 0].astype(float)
    # independance, thanks giving, christamas, 1st januarry,
    holidays=[pd.to_datetime("2014-07-04 00:00:00"),pd.to_datetime("2014-11-27 00:00:00"),pd.to_datetime("2014-12-24 00:00:00"),pd.to_datetime("2015-01-01 00:00:00")]
    dates = pd.date_range(start="2024-01-01", end="2024-12-31", freq='D')

    # List of official holidays (for example, US public holidays in 2024)
    holidays = [
        "2014-07-04",  # Independence Day
        "2014-09-02",  # Labor Day
        "2014-10-14",  # Columbus Day
        "2014-11-11",  # Veterans Day
        "2014-11-28",  # Thanksgiving
        "2014-12-25",   # Christmas Day
        "2015-01-01",  # New Year's Day
        "2015-01-15",  # Martin Luther King Jr. Day
    ]
    holidays = pd.to_datetime(holidays)
    plot=True
    if plot:
        fig, ax = plt.subplots(figsize=(10, 5))

        # Plot all dates with a thin line
        #ax.plot(dates, [1] * len(dates), color='gray', linewidth=1, label='Regular Days')

        # Highlight the holidays with thicker red lines
        for holiday in holidays:
            ax.axvline(holiday, color='red', linewidth=4, label='Holiday' if holiday == holidays[0] else "")

        # Formatting the plot
        ax.set_title("US Holidays in 2024", fontsize=14)
        ax.set_xlabel("Date", fontsize=12)
        ax.set_yticks([])  # Hide the y-axis
        ax.legend(loc='upper left')

        # Rotate the x-axis labels for better readability
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Show the plot
        plt.show()


    date_range = pd.date_range(start=dftemp.index[0], end=dftemp.index[-1], freq='D')



    # this may be the marathon
    change_hours=[pd.to_datetime("2014-11-01 23:00:00")]
    return df1,holidays,change_hours,data,dftemp,tempdata,extrem_weather


#plt.subplot(211)
#plt.plot(df1.index,x)
#plt.subplot(212)
#plt.plot(df1.index,score)
#plt.show()

def context_results(df1,holidays,change_hours,extrem_weather,dftemp,score):
    from PdmContext.utils.simulate_stream import simulate_stream

    eventconf2 = ("holidays", holidays, "isolated")
    spiketuples2 = ("change_hours", change_hours, "isolated")
    spiketuples3 = ("extreme", [dt for dt, ext in zip(dftemp.index, extrem_weather) if ext == 1], "isolated")

    anomaly1tuples2 = ("anomaly1", score, df1.index)

    stream = simulate_stream([anomaly1tuples2], [eventconf2, spiketuples2, spiketuples3],[], "anomaly1")

    contextpipeline3 = ContextGenerator(target="anomaly1", context_horizon=f"{7 * 24} hours",
                                        Causalityfunct=calculate_with_pc)
    source = "taxi"

    c = 0
    all_records=[]
    for record in stream:
        all_records.append(record)
    for q in tqdm(range(len(all_records))):
        record=all_records[q]
        kati=contextpipeline3.collect_data(timestamp=record["timestamp"], source=source, name=record["name"],
                                      type=record["type"], value=record["value"])
        # if record["name"]=="anomaly1" and q>5000:
        #     if ("holidays@taxi","anomaly1") in kati.CR["edges"]:
        #         kati.plot()
    contextpipeline3.plot([["", "anomaly1", "increase"]])
    contextpipeline3.plot([["", "anomaly1", ""]])
    contextpipeline3.plot_interpretation([["", "anomaly1", "increase"]])
    contextpipeline3.plot_interpretation([["", "", ""]])


def context_results_days(df1,holidays,change_hours,extrem_weather,dftemp,score):
    from PdmContext.utils.simulate_stream import simulate_stream

    eventconf2 = ("holidays", holidays, "isolated")
    spiketuples2 = ("change_hours", change_hours, "isolated")
    spiketuples3 = ("extreme", [dt for dt, ext in zip(dftemp.index, extrem_weather) if ext == 1], "isolated")

    anomaly1tuples2 = ("anomaly1", score, df1.index)

    date_range = pd.date_range(start=dftemp.index[0], end=dftemp.index[-1], freq='D')

    # Convert to a list of Timestamps
    timestampsdays = list(date_range)
    #categorical_hours=("day",[dt for dt in timestampsdays],[ times.day_name() for times in timestampsdays],"categorical")
    #stream = simulate_stream([anomaly1tuples2], [eventconf2, spiketuples2, spiketuples3],[categorical_hours] ,"anomaly1")

    stream = simulate_stream([anomaly1tuples2], [eventconf2, spiketuples2, spiketuples3],[] ,"anomaly1")

    contextpipeline3 = ContextGenerator(target="anomaly1", context_horizon=f"{7 * 24} hours",
                                        Causalityfunct=calculate_with_pc)
    source = "taxi"

    c = 0
    all_records=[]
    for record in stream:
        all_records.append(record)
    for q in tqdm(range(len(all_records))):
        c+=1
        record=all_records[q]
        kati=contextpipeline3.collect_data(timestamp=record["timestamp"], source=source, name=record["name"],
                                      type=record["type"], value=record["value"])
        if record["name"]=="anomaly1" and c>5000:
            if ("holidays@taxi","anomaly1") in kati["edges"]:
                kati.plot()
                plt.show()
    contextpipeline3.plot([["", "anomaly1", "increase"]])
    contextpipeline3.plot([["", "anomaly1", ""]])
    contextpipeline3.plot_interpretation([["", "anomaly1", "increase"]])
    contextpipeline3.plot_interpretation([["", "", ""]])

def main():
    df1,holidays,change_hours,data,dftemp,tempdata,extrem_weather=read_data()

    score=apply_SAND(data)
    # score=apply_Dist(data)
    # score=apply_lof(data)
    # score=apply_isolation_fores_semi(data)
    #score=apply_ocsvm(data)

    context_results(df1,holidays, change_hours, extrem_weather, dftemp, score)
    #context_results_days(df1, holidays, change_hours, extrem_weather, dftemp, score)



main()
