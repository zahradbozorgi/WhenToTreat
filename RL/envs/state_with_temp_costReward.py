import numpy as np
from gym import spaces
from RL.envs.baselineEnv import BaseEnv


class StatewithTempCostReward(BaseEnv):
    metadata = {'render.modes': ['human']}
    summary_writer = None

    def __init__(self):
        super().__init__()
        #################################
        # Parameter fuer das Environment
        #################################
        self.state = None
        self.action_space = spaces.Discrete(2)  # set action space to adaption true or false
        # Hier dimensionen des state-arrays anpassen:
        #        low_array = np.array([0, 0, 0])
        #        high_array = np.array([np.finfo(np.float32).max, np.finfo(np.float32).max, np.finfo(np.float32).max])

        # state = rel. position, reliability, pred. deviation, TE lower bound, TE upper bound, timesincecasestart, timesincemidnight, month,weekday, hour,open_cases
        low_array = np.array([0., 0., np.finfo(np.float32).min, np.finfo(np.float32).min, np.finfo(np.float32).min]) #, 0,1,1,0,0,0
        high_array = np.array([1., np.finfo(np.float32).max, np.finfo(np.float32).max, np.finfo(np.float32).max, np.finfo(np.float32).max]) #np.finfo(np.float32).max, np.finfo(np.float32).max,12,6,23, np.finfo(np.float32).max
        self.observation_space = spaces.Box(low=low_array, high=high_array)

        self.upper = None
        self.lower = None
        self.timesincecasestart = None
        self.timesincemidnight = None
        self.month = None
        self.weekday = None
        self.hour = None
        self.open_cases = None

    def step(self, action=None):
        if self.data.finished:
            self.close()
            raise SystemExit('Out of data!')
        if action is None:
            action = 0

        self.send_action(int(action))
        self.action_value = action

        self.receive_reward_and_state()

        self.do_logging(action)

        info = {}
        self.state = self._create_state()

        return self.state, self.reward, self.done, info

    def compute_reward(self, adapted, cost, gain, done, predicted_outcome, planned_outcome, reliability, position,
                       process_length, actual_outcome=0., true_effect=0, median=0):
        # if not done:
        #     reward = 0
        # else:
            # alpha = ((1. - 0.5) / process_length) * position
        violation = actual_outcome != planned_outcome
        if adapted:
            # if violation and (true_effect > 0):
            #     reward = (gain * true_effect) - cost
            # elif violation and (true_effect <= 0):
            #     # reward = (gain * true_effect) - cost
            #     reward = -1 * cost - gain
            #     self.true = False
            # elif not violation:
            #     reward = -1 * cost
            #     self.true = False
            if true_effect > 0:
                reward = (gain * true_effect) - cost
                self.true = True
            elif true_effect == 0:
                reward = -1 * cost
                self.true = False
            elif true_effect < 0:
                reward = -1 * cost - gain
                self.true = False
        else:
            # if violation and (true_effect > 0):
            #     # reward = -1 * gain * true_effect
            #     reward = -1 * gain - (true_effect * 1)
            #     self.true = False
            # elif violation and (true_effect <= 0):
            #     # reward = cost
            #     reward = -1 * gain
            # elif not violation:
            #     reward = gain
            if true_effect > 0:
                reward = -gain
                self.true = False
            elif true_effect == 0:
                if actual_outcome==1:
                    reward = gain
                    self.true = True
                else:
                    reward = 0
                    self.true = True
            elif true_effect<0:
                reward = gain
                self.true = True
        return reward

    def reset(self):
        # self.send_action(-1)
        self.receive_reward_and_state()

        self.state = self._create_state()

        return self.state

    def _create_state(self):
        relative_position = self.position / self.process_length
        prediction_deviation = (self.planned_outcome - self.predicted_outcome)/self.planned_outcome
        self.state = np.array(
            [relative_position, self.reliability, prediction_deviation, self.lower, self.upper]) #, self.timesincecasestart, self.timesincemidnight, self.month, self.weekday, self.hour, self.open_cases
        return self.state

    def render(self, mode='human'):
        # we don't need this
        return

    def receive_reward_and_state(self):
        # print("receiving reward parameters...")

        adapted = self.recommend_treatment
        adapted = True if adapted == 1 else False
        if adapted:
            self.treat_counter += 1

        cost = 25
        gain = 50
        done = self.data.done
        event = self.data.get_event()
        done = True if done == 1 else False
        # if done:
        #     true = event['true'][0]
        #     true = True if true == 'true' else False
        #     self.true = true


        case_id = event['Case ID orig'].iloc[0]
        predicted_outcome = event['preds'].iloc[0]
        predicted_outcome = float(predicted_outcome)
        planned_outcome = 1
        planned_outcome = float(planned_outcome)
        reliability = event['reliability'].iloc[0]
        reliability = float(reliability)
        position = event['event_nr'].iloc[0]
        position = float(position)
        process_length = event['case_length'].iloc[0]
        process_length = float(process_length)

        upper = event['upper']
        upper = float(upper)
        lower = event['lower']
        lower = float(lower)

        timesincecasestart = event['timesincecasestart'].iloc[0]
        timesincemidnight = event['timesincemidnight'].iloc[0]
        month = event['month'].iloc[0]
        weekday = event['weekday'].iloc[0]
        hour = event['hour'].iloc[0]
        open_cases = event['open_cases'].iloc[0]

        y0 = event['y0'].iloc[0]
        y1 = event['y1'].iloc[0]
        # ite = event['ite'].iloc[0]
        ite = y1 - y0

        actual_outcome = y1 if adapted else y0
        actual_outcome = float(actual_outcome)

        median_ite = self.data.median_ite

        # compute the reward
        reward = self.compute_reward(adapted, cost, gain, done, predicted_outcome, planned_outcome, reliability, position,
                                     process_length, actual_outcome=actual_outcome, true_effect=ite, median=median_ite)

        self.adapted = adapted
        self.cost = cost
        self.gain = gain
        self.done = done
        self.case_id = case_id
        self.actual_outcome = y1 if self.adapted else y0
        self.predicted_outcome = predicted_outcome
        self.planned_outcome = planned_outcome
        self.reliability = reliability
        self.reward = reward
        self.upper = upper
        self.lower = lower
        self.timesincecasestart = timesincecasestart
        self.timesincemidnight = timesincemidnight
        self.month = month
        self.weekday = weekday
        self.hour = hour
        self.open_cases = open_cases
        if adapted:
            done=True
            self.data.done = 1
            self.done=True

        if position != 0.:
            self.position = position
            self.process_length = process_length

        return 0
