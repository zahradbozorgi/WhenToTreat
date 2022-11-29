import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras.utils import pad_sequences


case_id_col = "Case ID orig"
activity_col = "Activity"
resource_col = "org:resource"
timestamp_col = "time:timestamp"

treatment = 'treatment'
outcome = 'outcome'

dynamic_cat_cols = ["Activity", 'org:resource', 'Action', 'EventOrigin', 'lifecycle:transition']
static_cat_cols = ['ApplicationType', 'LoanGoal']
dynamic_num_cols = ['FirstWithdrawalAmount', 'MonthlyCost', 'NumberOfTerms', 'OfferedAmount', "open_cases", "month", "weekday", "hour",
                    "timesincelastevent", "timesincecasestart", "timesincemidnight",'t_started']
static_num_cols = ['RequestedAmount', 'CreditScore', 'timesincefirstcase', 'treatment', 'outcome','time_of_treatment']

cat_cols = dynamic_cat_cols + static_cat_cols
num_cols = dynamic_num_cols + static_num_cols

not_for_lstm = ['treatment', 'case_length', 'Case ID orig', 't_started', 'time_of_treatment']
not_for_catboost = ['treatment', 'event_nr','outcome', 't_started', 'time_of_treatment', 'ite', 'y0','y1','upper','lower']


def split_data(data, train_ratio):
    # split into train and test using temporal split
    # data["Case ID orig"] = data.index
    # data["Case ID orig"] = data["Case ID orig"].apply(lambda x: x.split("_")[1])

    grouped = data.groupby('Case ID orig')
    start_timestamps = grouped[['timesincecasestart', 'timesincefirstcase']].min().reset_index()
    start_timestamps = start_timestamps.sort_values('timesincefirstcase', ascending=True, kind='mergesort')
    train_ids = list(start_timestamps['Case ID orig'])[:int(train_ratio * len(start_timestamps))]
    train = data[data['Case ID orig'].isin(train_ids)]
    test = data[~data['Case ID orig'].isin(train_ids)]
    train = train.sort_values(by=["Case ID orig", 'timesincecasestart'], kind='mergesort')
    test = test.sort_values(by=["Case ID orig", 'timesincecasestart'], kind='mergesort')
    # train = data[data['Case ID orig'].isin(train_ids)].groupby(by=['Case ID orig']).apply(lambda x: x.sort_values('timesincecasestart', ascending=True, kind='mergesort'))
    # test = data[~data['Case ID orig'].isin(train_ids)].groupby(by=['Case ID orig']).apply(lambda x: x.sort_values('timesincecasestart', ascending=True, kind='mergesort'))

    train.drop('Case ID orig', axis=1, inplace=True)
    test.drop('Case ID orig', axis=1, inplace=True)

    return (train, test)

def get_id_nr(df):
    df["Case ID orig"] = df.index
    df.rename(columns={'case_length': 'event_nr'}, inplace=True)
    # df.loc[df.case_length == 1, 'Case ID orig'] = df[df['case_length'] == 1]['Case ID orig'].apply(lambda x: x + str('_1'))
    df["Case ID orig"] = df["Case ID orig"].apply(lambda x: x.split("_")[1])
    df['case_length'] = df.groupby(by=['Case ID orig'])['event_nr'].transform(max)
    # df["event_nr"] = df["Case ID orig"].apply(lambda x: x.split("_")[2])
    # df = df.astype({"event_nr": int})

    return df

def prep_data_catboost(df):
    # X_train = df.drop(not_for_catboost, axis=1)
    X_train = df.drop(['outcome','treatment', 'Case ID','y1', 'y0', 'upper', 'lower'],axis=1, errors='ignore') #, 'Case ID'
    Y_train = df["outcome"]

    return X_train, Y_train


def prep_data_lstm(df):
    scaler = MinMaxScaler()
    dt_scaled = pd.DataFrame(scaler.fit_transform(df[num_cols]), index=df.index, columns=num_cols)
    df.loc[:, num_cols] = dt_scaled[num_cols]

    df["Case ID orig"] = df.index
    df["Case ID orig"] = df["Case ID orig"].apply(lambda x: x.split("_")[1])

    max_len = df['case_length'].max()

    data_list = []
    Y = []
    for id in df['Case ID orig'].unique():
        sample = df[df['Case ID orig'] == id].sort_values(by=['timesincecasestart'], ascending=True)
        sample.drop(not_for_lstm, axis=1, inplace=True)
        Y.append(sample[outcome].mean())
        sample.drop([outcome], axis=1, inplace=True)
        data_list.append(sample)

    Y = np.array(Y)
    Y = Y.astype(int)
    X = np.array([df_.to_numpy() for df_ in data_list])

    for idx, item in enumerate(X):
        X[idx] = pad_sequences(item.transpose(), maxlen=max_len, dtype=np.float64, value=-1).transpose()

    X = np.stack(X, axis=0)

    return X, Y

def chunks(lst, n,trim=True):
    """Yield successive n-sized chunks from lst.
    trim means that each chunk is exactly the same size.
    """
    size=len(lst)
    for i in range(0, size, n):
        if trim and (i + n)>size: #TODO: MAYBE SHOULD BE >=
            break
        else:
            yield lst[i:i + n]

def create_bootstrapped_dataset(data,label,n):
    index = np.random.choice(data.shape[0], n, replace=True)

    x_random = data[index]
    y_random = label[index]

    return x_random, y_random

def create_bootstrapped_dataframe(data,label,n):
    index = np.random.choice(data.shape[0], n, replace=True)

    x_random = data.iloc[index]
    y_random = label.iloc[index]

    return x_random, y_random