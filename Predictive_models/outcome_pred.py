import os

import numpy as np
import scipy
from catboost import CatBoostClassifier, CatBoostRegressor
import tensorflow as tf
from tensorflow.keras.optimizers import Adam, Nadam
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LSTM, Conv1D, ConvLSTM1D, Flatten, Masking
from tensorflow.keras.layers import BatchNormalization, Reshape, Input, TimeDistributed
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.callbacks import ModelCheckpoint
tf.keras.backend.clear_session()

from Predictive_models.utils import *

# cat_features = ['(case)_Item_Type', '(case)_Spend_area_text',
# '(case)_Sub_spend_area_text', '(case)_Vendor',
# '(case)_GR-Based_Inv__Verif_', '(case)_Spend_classification_text',
# '(case)_Item_Category', '(case)_Company',
# '(case)_Name', '(case)_Document_Type']

case_id_col = "Case ID"
activity_col = "Activity"
resource_col = "org:resource"
timestamp_col = "time:timestamp"

treatment = 'treatment'
outcome = 'outcome'

dynamic_cat_cols = ["Activity", 'org:resource', 'Action', 'EventOrigin', 'lifecycle:transition']
static_cat_cols = ['ApplicationType', 'LoanGoal']
dynamic_num_cols = ['FirstWithdrawalAmount', 'MonthlyCost', 'NumberOfTerms', 'OfferedAmount', "open_cases", "month", "weekday", "hour",
                    "timesincelastevent", "timesincecasestart", "timesincemidnight"] #,'t_started'
static_num_cols = ['RequestedAmount', 'CreditScore', 'timesincefirstcase', 'treatment', 'outcome'] #,'time_of_treatment'

cat_cols = dynamic_cat_cols + static_cat_cols
num_cols = dynamic_num_cols + static_num_cols

not_for_lstm = ['treatment', 'case_length', 'Case ID orig', 't_started', 'time_of_treatment']


def make_lstm_model(num_features=1):
    # tf.keras.backend.set_image_data_format(data_format='channels_first')
    model = Sequential()
    model.add(Masking(mask_value=-1, input_shape=(None, num_features)))
    model.add(LSTM(64,input_shape=(None,num_features),return_sequences=True,activation='tanh')) #
    # model.add(LeakyReLU(alpha=0.05))
    # model.add(BatchNormalization())
    model.add(Dropout(0.1))

    # model.add(LSTM(64, return_sequences=True))
    # model.add(LeakyReLU(alpha=0.05))
    # model.add(Dropout(0.2))

    # model.add(LSTM(128,return_sequences=True))
    # model.add(LeakyReLU(alpha=0.05))
    # model.add(Dropout(0.1))

    model.add(LSTM(50, activation='tanh'))
    # model.add(LeakyReLU(alpha=0.05))
    model.add(Dropout(0.2))
    # model.add(BatchNormalization())

    model.add(Dense(1, activation='sigmoid'))
    lr_schedule = ExponentialDecay(initial_learning_rate=1e-2,decay_steps=10000,decay_rate=0.9)
    opt = Adam(learning_rate=lr_schedule)
    model.compile(loss = 'binary_crossentropy', optimizer = 'Nadam', metrics = ['accuracy'])
    # model.summary()
    return model

class lstm_ensemble:

    def __init__(self, args):
        self.args=args
        self.batch_size = args.batch_size
        self.num_epochs = args.num_epochs
        self.num_models = args.num_models
        self.checkpoints_dir = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'results/lstms'))



    def create_ensemble(self, X_train, Y_train, X_val, Y_val):
        for m in range(self.num_models):
            X, Y = create_bootstrapped_dataset(X_train, Y_train, 10000)
            checkpoint = ModelCheckpoint('../results/lstms/model_' + f"{m}.h5", monitor='accuracy', save_best_only=True, save_freq='epoch')
            classifier = make_lstm_model(num_features=X.shape[2])
            print('starting training model number', m)
            classifier.fit(X, Y, batch_size=self.batch_size, epochs=self.num_epochs, validation_data=(X_val, Y_val), callbacks=[checkpoint])

    def get_preds(self, X_test):
        preds = []
        for filename in os.listdir(self.checkpoints_dir):
            if filename != '.ipynb_checkpoints':
                classifier = tf.keras.models.load_model(self.checkpoints_dir +'/'+ filename)
                print(self.checkpoints_dir +'/'+ filename)
                preds.append(classifier.predict(X_test))

        preds = np.array(preds)

        return preds, preds

    def agg_average(self,preds):
        average = np.mean(preds, axis=0)
        average[average <= 0.5] = 0
        average[average > 0.5] = 1
        average = average.squeeze()

        return average

    def agg_vote(self, preds):
        preds[preds <= 0.5] = 0
        preds[preds > 0.5] = 1
        maj_vote = scipy.stats.mode(preds, axis=0)[0]
        maj_vote = maj_vote.squeeze()

        return maj_vote


class catboost_ensemble:

    def __init__(self, args):
        self.args=args
        self.iterations = args.iterations
        self.eval_metric = args.eval_metric
        self.num_models = args.num_models
        self.checkpoints_dir = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'results/catboosts'))



    def create_ensemble(self, X_train, Y_train, X_val, Y_val):
        for m in range(self.num_models):
            X, Y = create_bootstrapped_dataframe(X_train, Y_train, int(X_train.shape[0]*1))
            classifier = CatBoostClassifier(iterations=self.iterations, eval_metric=self.eval_metric, random_state=1234) #eval_metric=self.eval_metric ,
            print('starting training model number', m)
            classifier.fit(X, Y, plot=False, eval_set=(X_val, Y_val), silent=True)
            classifier.save_model('../results/catboosts/model_'+f"{m}", format="cbm")

    def create_single_model(self, X_train, Y_train, X_val, Y_val):
        classifier = CatBoostClassifier(iterations=self.iterations, eval_metric=self.eval_metric, learning_rate=0.001,
                                        random_state=4321)  # eval_metric=self.eval_metric ,
        print('starting training single model')
        classifier.fit(X_train, Y_train, plot=False, eval_set=(X_val, Y_val), silent=True)
        classifier.save_model('../results/catboosts/model_complete', format="cbm")

    def get_preds(self, X_test):
        preds = []
        probs = []
        for filename in os.listdir(self.checkpoints_dir):
            if filename != '.ipynb_checkpoints':
                classifier = CatBoostClassifier()
                classifier.load_model(self.checkpoints_dir +'/'+ filename)
                print(self.checkpoints_dir +'/'+ filename)
                preds.append(classifier.predict(X_test))
                probs.append(classifier.predict_proba(X_test))
        preds = np.array(preds)
        probs = np.array(probs)

        return preds, probs

    def get_preds_onePredictor(self, X_test, filename):

        classifier = CatBoostClassifier()
        classifier.load_model(self.checkpoints_dir +'/'+ filename)
        print(filename)
        preds = classifier.predict(X_test)
        probs = classifier.predict_proba(X_test)

        preds = np.array(preds)
        probs = np.array(probs)

        return preds, probs

    def agg_average(self,preds):
        average = np.mean(preds, axis=0)
        # average[average <= 0.5] = 0
        # average[average > 0.5] = 1
        average = average.squeeze()

        return average

    def agg_vote(self, preds):
        # preds[preds <= 0.5] = 0
        # preds[preds > 0.5] = 1
        maj_vote = scipy.stats.mode(preds, axis=0)[0]
        maj_vote = maj_vote.squeeze()

        return maj_vote

    def get_reliability(self, X, Y):
        preds, probs = self.get_preds(X)
        # preds = np.transpose(preds)
        deviation = (1 - preds)/1
        reliability = np.count_nonzero(np.transpose(deviation), axis=1)/deviation.shape[0]

        return reliability