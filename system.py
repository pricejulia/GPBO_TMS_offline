import os
import pickle as p
from itertools import product
import numpy as np
import Tools.filters as f
import Tools.helper as h
import Tools.objective as ob

class System:
    def __init__(self,config):

        self.config = config

        if not os.path.exists(config.temp_data_storage_path):
            raise FileNotFoundError(
                f"Preprocessed dataset not found: {config.temp_data_storage_path}"
            )

        with open(config.temp_data_storage_path, 'rb') as file:
            disk_data_dict = p.load(file)

        self.search_space = np.array([i for i in product(*[ [j for j in range(len(i))] for i in self.config.search_space.values()])])

        self.Y = disk_data_dict['Y']

        self.data = disk_data_dict['data']

        if self.config.exp_type not in ['ISI_only','Cint_only','Both']:
            raise NotImplementedError
        
        self.compute_ground_truth()

        self.stim_counter = [0 for i in range(self.search_space.shape[0])]

        self.random_trial_order = np.stack([np.random.permutation(np.arange(self.config.NUM_TRIALS)) for i in range(self.search_space.shape[0])])

        self.emg_responses = np.empty((0,self.data.shape[-1]))

        self.obj = ob.Objective(self.config).func()

        self.filt = f.filter_emg()

        self.X_it = []
        self.Y_it = []

        self.max_response = -1*np.inf
        self.min_response = np.inf

    
    def compute_ground_truth(self):

        temp_ = self.Y.reshape(-1,self.config.NUM_TRIALS)

        self.ground_truth = np.array([np.nanmean(temp_,axis=1),np.nanstd(temp_,axis=1)])
        

    def get_response(self, query):
    
        if self.config.exp_type in ['ISI_only', 'Cint_only']:

            response = self.data[
                self.search_space[query][0],
                self.random_trial_order[query, self.stim_counter[query]]
            ]

            if (self.stim_counter[query] + 1) % self.config.NUM_TRIALS == 0:
                self.random_trial_order[query] = np.random.permutation(
                    np.arange(self.config.NUM_TRIALS)
                )

            self.stim_counter[query] = (
                self.stim_counter[query] + 1
            ) % self.config.NUM_TRIALS

            self.X_it.append([self.search_space[query][0]])

        elif self.config.exp_type == 'Both':

            response = np.nan

            while np.isnan(response).all():

                response = self.data[
                    self.search_space[query][0],
                    self.search_space[query][1],
                    self.random_trial_order[query, self.stim_counter[query]]
                ]

                if (self.stim_counter[query] + 1) % self.config.NUM_TRIALS == 0:
                    self.random_trial_order[query] = np.random.permutation(
                        np.arange(self.config.NUM_TRIALS)
                    )

                self.stim_counter[query] = (
                    self.stim_counter[query] + 1
                ) % self.config.NUM_TRIALS

            self.X_it.append([self.search_space[query][0]])

        else:
            raise NotImplementedError

        self.emg_responses = np.vstack((self.emg_responses, response))

        if self.config.exp_type in ['ISI_only', 'Both']:
            isis_ms = [
                int(self.config.search_space['ISI'][int(i[0])])
                for i in self.X_it
            ]
        else:
            isis_ms = [10 for _ in self.X_it]

        filtered_emgs = self.filt(
            self.emg_responses,
            np.array(isis_ms)
        )

        y = np.apply_along_axis(
            self.obj,
            axis=-1,
            arr=filtered_emgs
        )

        self.Y_it = y

        self.max_response = max(self.max_response, y.max())
        self.min_response = min(self.min_response, y.min())

        if self.config.random_pts['default'] <= len(self.Y_it):
            self.Y_it = (
                (self.Y_it - self.min_response)
                / (self.max_response - self.min_response)
            )

            return self.Y_it[:, None]

        return None


    def run_exhaustive(self,overwrite_repeats=None,overwrite_max_it=None):

        return h.run_random_search(self,self.config,overwrite_repeats=overwrite_repeats,overwrite_max_it=overwrite_max_it)
    
    
    def run_greedy(self, overwrite_repeats=None):
    
        assert 'Both' in self.config.exp_type
        return h.run_greedy_search(self,self.config,overwrite_repeats=overwrite_repeats)

    def instantiate_random_trial_order(self):

        self.random_trial_order = np.stack([np.random.permutation(np.arange(self.config.NUM_TRIALS)) for i in range(self.search_space.shape[0])])

        