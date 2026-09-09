import numpy as np
import copy
import Tools.train_sequence as seq


def phase_non_hierarchical(system_5,config_5,run_exhaustive = True):

    if config_5.offline:

        system_5,config_5 = copy.deepcopy(system_5),copy.deepcopy(config_5)

        system_5.instantiate_random_trial_order()

    model_5 = seq.train_model(system_5,config_5)

    if not config_5.offline:

        model_5.exploration_performance = {'ground_truth':None,'curve':np.array(system_5.Y_it)}

    model_5.log(id='5')

    if run_exhaustive:
        
        if 'Both' in config_5.exp_type:

            model_5.greedy_performance = system_5.run_greedy(overwrite_repeats=1)

        model_5.exhaustive_performance = system_5.run_exhaustive(overwrite_repeats=1)

    return [model_5]
