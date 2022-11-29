# Learning when to treat
 

## Instructions
To train the causal forest model, execute the following script:

```bash
python experiments/wtt_experiment.py
```

and the following for training the predictive model:

```bash
python experiments/pred_experiment.py
```

After training the predictive model, load the results to the Jupyter notebook below and execute the cells.

```bash
conformal_prediction.ipynb
```

The scripts for running the policy learning experiments are the following:

```bash
python ppo_temp_cost_reward_noPred.py
```

```bash
python ppo_temp_cost_reward_conformal.py
```
