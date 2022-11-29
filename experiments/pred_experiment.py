import pandas as pd
import numpy as np
import argparse
from sys import argv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from Predictive_models.outcome_pred import *
from Predictive_models.utils import *
from tqdm import tqdm

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

not_for_lstm = ['treatment', 'case_length', 'Case ID orig', 't_started', 'time_of_treatment']


def get_args():
    parser = argparse.ArgumentParser(description="when_to_treat")

    # dataset
    # parser.add_argument("--debug", type=str_to_bool, default=False)
    parser.add_argument("--data", type=str, default="bpic17",choices=["bpic17","bpic19"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--num_models", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--eval_metric", type=str, default="Accuracy")
    # parser.add_argument("--is_forest", type=bool, default=False)

    return parser

def main(args):
    print('Reading data')
    df_train = pd.read_pickle('../data_12_multiple_offers_train_newRatio.pkl')
    # df = pd.read_csv('results_te_interval.csv')
    # df = pd.read_csv('test_19.csv')
    # df_test = pd.read_pickle('../data_17_multiple_offers_test_newRatio.pkl')
    df_test = pd.read_csv('resultsCF_enhanced_log_12_counterfacs.csv')

    # df_train, df_val = split_data(df, train_ratio=0.8)
    # df_train, df_val = train_test_split(df, test_size=0.8, shuffle=False)
    # df_train.drop('Case ID orig', axis=1, inplace=True)
    # df_val.drop('Case ID orig', axis=1, inplace=True)

    # df_train = pd.read_csv('train.csv')
    # df_val = pd.read_csv('test.csv')

    # print('preparing data')
    # X, Y = prep_data_lstm(df_train)
    # X_val, Y_val = prep_data_lstm(df_val)

    # df_train = df_train.head(500)
    # df_test = df_test.head(500)
    print('preparing data')
    # df_result = get_id_nr(df)
    df_result = df_test.copy()
    # df.drop('Case ID orig', axis=1, inplace=True)
    # df_result['y0'] = df_result['outcome']
    # df_result['y1'] = df_result['outcome']
    # df_result['ite'] = 0
    # X, Y = prep_data_catboost(df)
    X_train, Y_train = prep_data_catboost(df_train)
    X_test, Y_test = prep_data_catboost(df_test)

    # np.save('train_data', X)
    # np.save('train_label', Y)
    # np.save('val_data', X_val)
    # np.save('val_label', Y_val)
    # X_train = np.load('train_data.npy')
    # Y_train = np.load('train_label.npy')
    # X_val = np.load('val_data.npy')
    # Y_val = np.load('val_label.npy')

    ensemble = catboost_ensemble(args)
    ensemble.create_ensemble(X_train, Y_train, X_test, Y_test)
    # ensemble.create_single_model(X_train, Y_train, X_test, Y_test)

    # preds, probs = ensemble.get_preds_onePredictor(X_test, 'model_complete')
    preds, probs = ensemble.get_preds(X_test)
    # probs = probs.squeeze()

    # averaged = ensemble.agg_average(preds)
    # av_acc = accuracy_score(averaged, Y_val)
    # print('average aggregation accuracy is', av_acc)
    votes = ensemble.agg_vote(preds)
    avg_probs = ensemble.agg_average(probs)
    # avg_probs = [avg_probs[i][0] for i in range(len(avg_probs))]
    df_result['probability_0'] = avg_probs[:, 0]
    df_result['probability_1'] = avg_probs[:, 1]
    vote_acc = accuracy_score(votes, Y_test)
    print('voting aggregation accuracy is', vote_acc)
    df_result['preds2'] = votes
    # df_result['probability'] = avg_probs

    reliability = ensemble.get_reliability(X_test, Y_test)
    df_result['reliability'] = reliability
    df_result['case_length'] = df_result.groupby(by=['Case ID'])['event_nr'].transform(max)
    df_result.to_csv('results_forConformal_counterfacs_12.csv', index=False)
    print('voting aggregation accuracy is', vote_acc)
    print('finished')


    # for epoch in range(args.num_epochs):
        # dset = list(zip(X, Y))
        # Xes = chunks(dset, args.batch_size, trim=True)
        # np.random.shuffle(dset)
        # n = len(X) / args.batch_size + 1
        #
        # for d in tqdm(Xes, total=n):
        #     if len(d) != args.batch_size:  # ignoring the tail end data cause need same size each time.
        #         print(f"skipped {len(d)}")
        #         continue
        #
        # x_batch, y_batch = list(zip(*d))
        # if len(x_batch) != args.batch_size:
        #     print(f"BATCH SIZE NOT CONSISTENT. MAY CAUSE RECOMPILE OF GRAPH")



if __name__ == '__main__':
    print("@report: Started")
    main(get_args().parse_args())