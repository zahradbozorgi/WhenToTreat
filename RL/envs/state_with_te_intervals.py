import numpy as np
from gym import spaces
from RL.envs.baselineEnv import BaseEnv


class StatewithTEinterval(BaseEnv):
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

        # state = rel. position, reliability, pred. deviation, TE lower bound, TE upper bound
        low_array = np.array([0., 0., np.finfo(np.float32).min, np.finfo(np.float32).min, np.finfo(np.float32).min])
        high_array = np.array([1., np.finfo(np.float32).max, np.finfo(np.float32).max, np.finfo(np.float32).max, np.finfo(np.float32).max])
        self.observation_space = spaces.Box(low=low_array, high=high_array)

        self.upper = None
        self.lower = None

    def step(self, action=None):
        if action is None:
            action = 0

        self.send_action(int(action))
        self.action_value = action

        self.receive_reward_and_state()

        self.do_logging(action)

        info = {}
        self.state = self._create_state()

        return self.state, self.reward, self.done, info

    def compute_reward(self, adapted, cost, done, predicted_outcome, planned_outcome, reliability, position,
                       process_length, actual_outcome=0.):
        if not done:
            reward = 0
        else:
            alpha = ((1. - 0.5) / process_length) * position
            violation = actual_outcome != planned_outcome
            if adapted:
                if violation:
                    reward = alpha
                else:
                    reward = -0.5 - alpha * 0.5
            else:
                if violation:
                    reward = -1.
                else:
                    reward = 1.
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
            [relative_position, self.reliability, prediction_deviation, self.lower, self.upper])
        return self.state

    def render(self, mode='human'):
        # we don't need this
        return

    def receive_reward_and_state(self):
        # print("receiving reward parameters...")

        adapted = self.recommend_treatment
        adapted = True if adapted == 1 else False

        cost = 0

        done = self.data.done
        event = self.data.get_event()
        done = True if done == 1 else False
        # if done:
        #     true = event['true'][0]
        #     true = True if true == 'true' else False
        #     self.true = true


        case_id = event['Case ID orig'].iloc[0]
        actual_outcome = event['outcome'].iloc[0]
        actual_outcome = float(actual_outcome)
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

        # compute the reward
        reward = self.compute_reward(adapted, cost, done, predicted_outcome, planned_outcome, reliability, position,
                                     process_length, actual_outcome=actual_outcome)

        self.adapted = adapted
        self.cost = cost
        self.done = done
        self.case_id = case_id
        self.actual_outcome = actual_outcome
        self.predicted_outcome = predicted_outcome
        self.planned_outcome = planned_outcome
        self.reliability = reliability
        self.reward = reward
        self.upper = upper
        self.lower = lower

        if position != 0.:
            self.position = position
            self.process_length = process_length

        return reward, done, predicted_outcome, planned_outcome, reliability, position, process_length, cost, adapted
