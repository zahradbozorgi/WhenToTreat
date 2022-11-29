import pandas as pd
import numpy as np
import argparse
from sys import argv
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from causal_estimators.wtt_estimator import wtt_estimator
from Predictive_models.utils import *

case_id_col = "Case ID"
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


def get_args():
    parser = argparse.ArgumentParser(description="when_to_treat")

    # dataset
    # parser.add_argument("--debug", type=str_to_bool, default=False)
    parser.add_argument("--data", type=str, default="bpic17",choices=["bpic17","bpic19"])
    parser.add_argument("--estimator", type=str, default='CausalForest')
    parser.add_argument("--propensity_model", type=str, default='LogisticRegression')
    parser.add_argument("--outcome_model", type=str, default='RandomForestClassifier')
    parser.add_argument("--conf_thresh", type=float, default=0.1)
    # parser.add_argument("--is_forest", type=bool, default=False)

    return parser


def main(args):
    print('Reading data')
    df_train = pd.read_pickle('../data_12_multiple_offers_train_newRatio.pkl')
    # df_test = pd.read_pickle('../data_17_multiple_offers_test_newRatio.pkl')
    df_test = pd.read_csv('../Log_withCounterFacs_12.csv')
    # df_train = df_train.head(2000)
    # df_train, df_thresh = split_data(df_train, train_ratio=0.8)
    # df_thresh = df_thresh.head(10)
    # df_test = df_test.head(10)

    print('preparing data')
    features = df_train.drop([outcome, treatment, case_id_col], axis=1)
    # features_thresh = df_thresh.drop([outcome, treatment, 'time_of_treatment'], axis=1)
    features_test = df_test.drop([outcome, treatment, case_id_col, 'y1', 'y0'], axis=1)

    # df_train = pd.read_csv('../generated_log.csv')
    Y = df_train[outcome].to_numpy()
    T = df_train[treatment].to_numpy()
    # features = df_train.drop([outcome, treatment, 'time_of_treatment'], axis=1)


    ###### Standardisation ######
    scaler = MinMaxScaler()
    W = scaler.fit_transform(features[[c for c in features.columns]].to_numpy())
    X = scaler.fit_transform(features[[c for c in features.columns]].to_numpy())
    # X_thresh = scaler.fit_transform(features_thresh[[c for c in features_thresh.columns]].to_numpy())
    X_test = scaler.fit_transform(features_test[[c for c in features_test.columns]].to_numpy())

    # df_thresh["Case ID orig"] = df_thresh.index
    # df_thresh["Case ID orig"] = df_thresh["Case ID orig"].apply(lambda x: x.split("_")[1])
    # df_results = df_thresh[['Case ID orig','case_length']]
    df_results = df_test
    # df_results = df_results.rename(columns={'case_length': 'event_nr'})

    wtt = wtt_estimator(args)
    wtt.initialize_forest()
    wtt.fit_forest(X,T,Y)
    # te = wtt.get_te(X_thresh)
    lower,upper = wtt.get_te_withCI(X_test)
    # df_results['treatment_effects'] = te
    # CILen = wtt.get_CI_length()
    # df_results['interval'] = CILen
    df_results['upper'] = upper
    df_results['lower'] = lower
    wtt.save_results(df_results)
    # wtt.evaluate_model_cost(df_results)

    # best_params = wtt.find_opt_thresh()
    # print('the best threshold is:', best_params)
    # df = df_results[['Case ID', 'lower', 'upper']]
    df_results.to_csv('resultsCF_enhanced_log_12_counterfacs.csv', index=False)

    print('@report: Ended')


if __name__ == '__main__':
    print("@report: Started")
    main(get_args().parse_args())