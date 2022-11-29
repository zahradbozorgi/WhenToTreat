import numpy as np
from gym import spaces
from RL.envs.baselineEnv import BaseEnv, get_average_last_entries_from_numeric_list


class StateCostRewardMultTreat(BaseEnv):
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
        low_array = np.array([0., 0., np.finfo(np.float32).min, np.finfo(np.float32).min, np.finfo(np.float32).min, 0,1,1,0,0,0])
        high_array = np.array([1., np.finfo(np.float32).max, np.finfo(np.float32).max, np.finfo(np.float32).max, np.finfo(np.float32).max,
                               np.finfo(np.float32).max, np.finfo(np.float32).max,12,6,23, np.finfo(np.float32).max])
        self.observation_space = spaces.Box(low=low_array, high=high_array)

        self.upper = None
        self.lower = None
        self.timesincecasestart = None
        self.timesincemidnight = None
        self.month = None
        self.weekday = None
        self.hour = None
        self.open_cases = None
        self.num_adapt_in_ep = []

    def step(self, action=None):
        if self.data.finished:
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
                       process_length, actual_outcome=0., true_effect=0):
        # if not done:
        #     reward = 0
        # else:
            # alpha = ((1. - 0.5) / process_length) * position
        violation = actual_outcome != planned_outcome
        if adapted:
            if violation and (true_effect > 0):
                reward = (gain * true_effect) - (cost*(self.treat_counter))
            elif violation and (true_effect <= 0):
                reward = (gain * true_effect) - (cost*(self.treat_counter))
                self.true = False
            elif not violation:
                reward = -1 * (cost*(self.treat_counter))
                self.true = False
        else:
            if violation and (true_effect > 0):
                reward = -1 * gain
                self.true = False
            elif violation and (true_effect <= 0):
                reward = 0
            elif not violation:
                reward = gain
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
            [relative_position, self.reliability, prediction_deviation, self.lower, self.upper, self.timesincecasestart,
             self.timesincemidnight, self.month, self.weekday, self.hour, self.open_cases])
        return self.state

    def render(self, mode='human'):
        # we don't need this
        return

    def receive_reward_and_state(self):
        # print("receiving reward parameters...")

        adapted = self.recommend_treatment
        adapted = True if adapted == 1 else False
        self.recommend_treatment = True if self.recommend_treatment == 1 else False
        if self.recommend_treatment:
            self.treat_counter += 1

        cost = 1
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

        # compute the reward
        reward = self.compute_reward(self.recommend_treatment, cost, gain, done, predicted_outcome, planned_outcome, reliability, position,
                                     process_length, actual_outcome=actual_outcome, true_effect=ite)

        if adapted:
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
        # if adapted:
        #     done=True
        #     self.data.done = 1
        #     self.done=True

        if position != 0.:
            self.position = position
            self.process_length = process_length

        return reward, done, predicted_outcome, planned_outcome, reliability, position, process_length, cost, adapted

    def do_logging(self, action):
        # Reward, Kosten und Aktion in Gesamtliste speichern:
        self.rewards.append(self.reward)
        self.costs.append(self.cost)
        self.actions.append(int(action))
        if self.predicted_outcome < self.planned_outcome:
            # pred_violation = 1
            adap_no_pred = 0
        else:
            # pred_violation = 0
            if int(action) == 1:
                adap_no_pred = 1
            else:
                adap_no_pred = 0

        # self.violation_predicted.append(pred_violation)
        self.adapted_no_violation.append(adap_no_pred)
        # Reward, Kosten und Aktion in temporaeren Listen fuer Episode speichern:
        self.tmp_cumul_cost_per_ep.append(self.cost)
        self.tmp_cumul_reward_per_ep.append(self.reward)
        # self.tmp_avg_action_per_ep.append(int(action))

        # self.log_with_tensorboard(tag='custom/reward', simple_value=self.reward,
        #                           step=self.total_steps)
        # self.log_with_tensorboard(tag='custom/action', simple_value=self.action_value,
        #                           step=self.total_steps)
        # self.log_with_tensorboard(tag='custom/cost', simple_value=-self.cost,
        #                           step=self.total_steps)
        # self.log_with_tensorboard(tag='custom/violation_predicted', simple_value=pred_violation,
        #                           step=self.total_steps)
        self.log_with_tensorboard(tag='custom/adapted_though_no_violation_predicted', simple_value=adap_no_pred,
                                  step=self.total_steps)

        self.total_steps += 1
        if self.total_steps % 1000 == 0:
            if len(self.percentage_true_last_100) > 0:
                print("Step " + str(self.total_steps) + " in episode " + str(self.episode_count) + " with " + str(
                    self.percentage_true_last_100[-1]) + " true decisions")
            else:
                print("Step " + str(self.total_steps) + " in episode " + str(self.episode_count))

        if self.recommend_treatment:
            self.position_of_adaptation_per_episode.append(self.position)
            self.earliness.append(self.position / self.process_length)
            self.log_with_tensorboard(tag='episode_result/earliness',
                                  simple_value=self.earliness[-1],
                                  step=self.episode_count)

        if self.done:  # Ende einer Episode
            self.case_id_list.append(self.case_id)
            # Speichern der kumulativen und durschnittlichen Episodenkosten
            cumul_episode_cost = sum(self.tmp_cumul_cost_per_ep)
            self.cumul_cost_per_ep.append(cumul_episode_cost)
            # avg_episode_cost = cumul_episode_cost / len(self.tmp_cumul_cost_per_ep)
            # self.avg_cost_per_ep.append(avg_episode_cost)
            self.tmp_cumul_cost_per_ep = []
            ep_gain = (self.gain*self.actual_outcome) - (self.treat_counter*self.cost)
            self.log_with_tensorboard(tag='episode_reward/episode_net_gain*',simple_value=ep_gain,
                                      step=self.episode_count)

            # Speichern des kumulativen und durschnittlichen Rewards pro Episode
            cumul_episode_reward = sum(self.tmp_cumul_reward_per_ep)

            # avg_episode_reward = cumul_episode_reward / len(self.tmp_cumul_reward_per_ep)
            self.cumul_reward_per_ep.append(cumul_episode_reward)
            # self.avg_reward_per_ep.append(avg_episode_reward)
            self.tmp_cumul_reward_per_ep = []

            # Speichern der durschnittlichen Aktionen einer Episode und ob adaptiert wurde
            # avg_actions = sum(self.tmp_avg_action_per_ep) / len(self.tmp_avg_action_per_ep)
            # self.avg_action_per_ep.append(avg_actions)
            # elf.tmp_avg_action_per_ep = []

            true_negative_status = self.true
            self.true_per_ep.append(true_negative_status)
            if self.adapted:
                self.adapt_in_ep.append(1)
                # self.true_positive_per_positive.append(true_negative_status)
                self.case_length_per_episode.append(self.process_length)
                self.num_adapt_in_ep.append(self.treat_counter)
                self.log_with_tensorboard(tag='episode_reward/treat_counts*', simple_value=self.treat_counter,
                                          step=self.episode_count)
            else:
                self.adapt_in_ep.append(0)
                # self.position_of_adaptation_per_episode.append(-1)
                # self.earliness.append(-1)

            self.case_length_per_episode.append(self.process_length)
            # self.num_adapt_in_ep.append(self.treat_counter)
            # self.log_with_tensorboard(tag='episode_reward/treat_counts*', simple_value=self.treat_counter,
            #                           step=self.episode_count)
            self.treat_counter = 0
            self.adapted = False
            self.true = True

            # self.true_negative_per_negative.append(true_negative_status)
            avg_adapted_100_value = sum(self.adapt_in_ep[-100:]) / 100
            self.avg_adapted_100.append(avg_adapted_100_value)

            percentage_true_last_100_value = get_average_last_entries_from_numeric_list(self.true_per_ep, 100)
            self.percentage_true_last_100.append(percentage_true_last_100_value)
            # percentage_true_positive_100_value = get_average_last_entries_from_numeric_list(
            #    self.true_positive_per_positive, 100)
            # self.percentage_true_positive_100.append(percentage_true_positive_100_value)
            # percentage_true_negative_100_value = get_average_last_entries_from_numeric_list(
            #    self.true_negative_per_negative, 100)
            # self.percentage_true_negative_100.append(percentage_true_negative_100_value)

            # self.log_with_tensorboard(tag='episode_reward/episode_cost', simple_value=-cumul_episode_cost,
            #                           step=self.episode_count)
            self.log_with_tensorboard(tag='episode_reward/episode_reward*', simple_value=cumul_episode_reward,
                                      step=self.episode_count)
            # self.log_with_tensorboard(tag='episode_result/adapted', simple_value=self.adapted,
            #                           step=self.episode_count)
            self.log_with_tensorboard(tag='episode_result/average_adapted_last_100_episodes',
                                      simple_value=avg_adapted_100_value,
                                      step=self.episode_count)
            # self.log_with_tensorboard(tag='episode_result/average_action', simple_value=avg_actions,
            #                           step=self.episode_count)
            # self.log_with_tensorboard(tag='episode_result/ended_correctly', simple_value=true_negative_status,
            #                           step=self.episode_count)
            # if self.adapted:
            #     self.log_with_tensorboard(tag='episode_result/positives_ended_correctly',
            #                               simple_value=true_negative_status,
            #                               step=self.adapted_count)
            # else:
            #     self.log_with_tensorboard(tag='episode_result/negatives_ended_correctly',
            #                               simple_value=true_negative_status,
            #                               step=self.episode_count - self.adapted_count)
            self.log_with_tensorboard(tag='episode_result/percentage_of_trues_among_last_100_episodes',
                                      simple_value=percentage_true_last_100_value,
                                      step=self.episode_count)
            # self.log_with_tensorboard(tag='episode_result/percentage_of_trues_among_last_100_positives',
            #                           simple_value=percentage_true_positive_100_value,
            #                           step=self.episode_count)
            # self.log_with_tensorboard(tag='episode_result/percentage_of_trues_among_last_100_negatives',
            #                           simple_value=percentage_true_negative_100_value,
            #                           step=self.episode_count)
            #
            # self.log_with_tensorboard(tag='episode_result/percentage_Correct_Adaptation_Decisions',
            #                           simple_value=(self.true_per_ep[-1000:].count(1) / 1000) * 100,
            #                           step=self.episode_count)
            # self.log_with_tensorboard(tag='episode_result/adapt_in_ep',
            #                           simple_value=(self.adapt_in_ep.count(1) / self.adapt_in_ep.__len__()) * 100,
            #                           step=self.episode_count)
            self.episode_count += 1
            if self.recommend_treatment:
                self.adapted_count += 1
            if self.data.finished != True:
                self.data.get_new_case()
