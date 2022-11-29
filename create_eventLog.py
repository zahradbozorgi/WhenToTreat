import pandas as pd
import numpy as np
from loading import load_from_folder

df = pd.read_pickle('data_12_multiple_offers_test_newRatio.pkl')
gen_model, args = load_from_folder(dataset='bpic12', checkpoint_path="GenModelCkpts")
# bpic17_ite = gen_model.ite(noisy=True).squeeze()

cols = list(df.columns)
cols.remove('treatment')
cols.remove('outcome')
# cols.remove('time:timestamp')

w, t, (y0, y1) = gen_model.sample(seed=123, ret_counterfactuals=True)
df_w = pd.DataFrame(w, columns=cols)
df_w['y0'] = y0
df_w['y1'] = y1
df_w['treatment']= t
# df_w['ite'] = bpic17_ite
df_w['outcome'] = 0
df_w.loc[df_w.treatment==0, 'outcome'] = df_w[df_w['treatment']==0]['y0']
df_w.loc[df_w.treatment==1, 'outcome'] = df_w[df_w['treatment']==1]['y1']

df_w.to_csv('enhanced_log_12.csv', index=False)