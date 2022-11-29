import pandas as pd


case_id_col = "Case ID"
directory = "/home/zdashtbozorg/when_to_treat/experiments/"
filename = "results_adaptive_counterfacs.csv"

class envManager():

    def __init__(self):
        self.current_case_id = 0
        self.finished = False

        self.load_cases()
        self.get_num_of_cases()
        self.get_cases_list()


    def load_cases(self):
        self.df = pd.read_csv(directory+filename)

    def get_num_of_cases(self):
        self.num_cases = self.df[case_id_col].nunique()

    def get_cases_list(self):
        self.list_cases = list(self.df[case_id_col].unique())

    def get_new_case(self):
        if len(self.list_cases) != 0:
            self.current_case_id = self.list_cases.pop(0)
            self.current_df = self.df.loc[self.df[case_id_col]==self.current_case_id]
            self.current_df.sort_values(by='event_nr', inplace=True)
            self.done = 0
            self.index = 0
        else:
            self.finished = True

    def get_event(self):
        if self.index < self.current_df.shape[0]:
            self.current_event = self.current_df.iloc[[self.index]]
            self.index += 1
            if self.index == self.current_df.shape[0] - 1:
                self.done = 1
        else:
            self.done = 1

        return self.current_event


