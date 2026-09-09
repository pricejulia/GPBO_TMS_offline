import pickle as p
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import glob 
import os
import json

import Tools.helper as h
import Tools.procedures as pcd

import config as conf
import system as syst

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'


def calibrate_hyperparameter_non_hier(system_5,config_5, Calibrate):

    calibration_results = {}
    raw_results = {}
    run_exhaustive = True

    for hyperparameter in ['kappa','rho_min','rho_max','noise_min','noise_max','random_pts']:
    
        hyperparameter_setup = getattr(config_5, hyperparameter)

        if hyperparameter_setup['calibrate']:

            per_hyperparameter_results = []

            per_hyperparameter_raw = []

            default_value = hyperparameter_setup['default']

            for value in tqdm(hyperparameter_setup['values']):

                hyperparameter_setup['default'] = value

                procedure_1 = lambda: pcd.phase_non_hierarchical(system_5,config_5,run_exhaustive=run_exhaustive)

                stats_models,stacked_models = h.model_population(procedure_1, config_5)

                per_hyperparameter_results.append(stats_models['gpbo_models'][-1][0])

                per_hyperparameter_raw.append(stacked_models)
            
                titles_gpbo = ["GPBO"]
                titles_exhaustive = ["Exhaustive"]
                
                if 'exhaustive_models' in stats_models and 'greedy_models' in stats_models :  

                    learning_curves = [stats_models['exhaustive_models'],stats_models['greedy_models'],stats_models['gpbo_models']]

                    titles = [titles_exhaustive,["Greedy"],titles_gpbo]
                
                elif 'exhaustive_models' in stats_models:    

                    learning_curves = [stats_models['exhaustive_models'],stats_models['gpbo_models']]

                    titles = [titles_exhaustive,titles_gpbo]
            
                else:

                    learning_curves = [stats_models['gpbo_models']]

                    titles = [titles_exhaustive,titles_gpbo]

                if Calibrate:

                    h.plot_non_hierarchical(learning_curves,titles,config_5,hyperparameter=hyperparameter,hyperparameter_value=value)
                
            hyperparameter_setup['default'] = default_value

            ## For random_pts calibration, where learning curve lengths vary
            def left_pad_stack(arrs, value=0):
                """Right-align 1D arrays and left-pad with `value`, returning a 2D matrix."""
                arrs = [np.asarray(a) for a in arrs]
                maxlen = max(a.shape[0] for a in arrs)
                out = np.full((len(arrs), maxlen), value, dtype=np.result_type(*arrs))
                for i, a in enumerate(arrs):
                    out[i, -a.shape[0]:] = a
                return out 
            
            if hyperparameter == 'random_pts':
                
                calibration_results[hyperparameter] = left_pad_stack(per_hyperparameter_results)

                cumulate = []
                for i in per_hyperparameter_raw:

                    cumulate = cumulate + [j for j in i['gpbo_models'][0]]

                padded_cumulate = left_pad_stack(cumulate)

                padded_cumulate = padded_cumulate.reshape(-1,config_5.repeats,padded_cumulate.shape[-1])

                for idx,i in enumerate(per_hyperparameter_raw):

                    per_hyperparameter_raw[idx]['gpbo_models'][0] = padded_cumulate[idx]

                raw_results[hyperparameter] = per_hyperparameter_raw

            else:

                calibration_results[hyperparameter] = np.array(per_hyperparameter_results)

                raw_results[hyperparameter] = per_hyperparameter_raw

            if Calibrate:

                if 'exhaustive_models' in stats_models:    

                    h.plot_hyperparameter_calibration(calibration_results[hyperparameter],config_5,hyperparameter,exh_mu=stats_models['exhaustive_models'][-1][0])

                else:

                    h.plot_hyperparameter_calibration(calibration_results[hyperparameter],config_5,hyperparameter,exh_mu=None)

            p.dump({'GPBO_non_hier':raw_results},open(config_5.save_folder+'/raw_results.p','wb'))

    config_5.log_config()


def aggregate_curves(path_, maximize_MEP = True):

    subfolders = glob.glob(os.path.join(path_, "*/"))

    subfolders = [i for i in subfolders if 'aggregate' not in i]

    path_aggregate = path_ + '/aggregate'

    if not os.path.exists(path_aggregate):
        os.makedirs(path_aggregate)
        print(f"Folder created at: {path_aggregate}")
    else:
        print(f"Folder already exists at: {path_aggregate}")

    more = [p.load(open(i+"raw_results.p",'rb')) for i in subfolders]

    if 'exhaustive' in more[0].keys():

        # Unpacking models
        models = [m['exhaustive'] for m in more]

        # Looping over systems models
        for mod_idx in range(len(models[0])):
            
            exhaustive_aggregate = []

            for mod in models:

                exhaustive_aggregate.append(mod[mod_idx]['curve'])

            exhaustive_aggregate = np.stack(exhaustive_aggregate).mean(0).mean(0)

            plt.plot(exhaustive_aggregate)
            plt.title('System Exhaustive')
            plt.xlabel('Queries')
            plt.ylabel('Performance')

            ax = plt.gca()
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            plt.savefig(path_aggregate+f'/aggregate_{mod_idx}.png',dpi=500)
            plt.close('all')
    
    elif 'GPBO_non_hier'in more[0].keys() :

        more = [m['GPBO_non_hier'] for m in more]

        # identify hyperarameter calibrated 
        hyper = list(more[0])[0]

        # Looping over the systems or models
        for model_idx in range(len(more[0][hyper][0]['gpbo_models'])):

            gpbo_aggregate = [[] for i in range(len(more[0][hyper]))]
            exhaustive_aggregate = []
            greedy_aggregate = []

            # Looping over datasets
            for m in more:

                for hyper_value in range(len(m[hyper])):

                    gpbo_aggregate[hyper_value].append(m[hyper][hyper_value]['gpbo_models'][model_idx])

                if 'exhaustive_models' in more[0][hyper][0].keys():
                    exhaustive_aggregate.append(m[hyper][-1]['exhaustive_models'][model_idx])

                if 'greedy_models' in more[0][hyper][0].keys():
                    greedy_aggregate.append(m[hyper][-1]['greedy_models'][model_idx])

            gpbo_aggregate = np.stack(gpbo_aggregate)
            n_hyper, n_datasets, n_repeats, n_queries = gpbo_aggregate.shape

            if maximize_MEP:
                gpbo_aggregate = np.maximum(gpbo_aggregate, 0)
            else:
                gpbo_aggregate = np.minimum(gpbo_aggregate, 1)
                gpbo_aggregate = gpbo_aggregate - 1

            gpbo_per_participant = np.mean(gpbo_aggregate, axis=2)
            gpbo_mean = np.mean(gpbo_per_participant, axis=1)

            if n_datasets > 1:
                gpbo_sem = (
                    np.std(gpbo_per_participant, axis=1, ddof=1)
                    / np.sqrt(n_datasets)
                )
            else:
                gpbo_sem = (
                    np.std(gpbo_aggregate, axis=2, ddof=1)
                    / np.sqrt(n_repeats)
                )[:, 0, :]
            
            with open(subfolders[0]+f'log_config_{1}.json', 'r') as file:
                config_data = json.load(file)

            hyper_values = config_data[hyper]['values']
            
            if not config_data[hyper]['calibrate']:
                hyper_values = [f"{config_data[hyper]['default']}"] * gpbo_mean.shape[0]

            plt.rcParams['figure.facecolor'] = 'white'
            plt.rcParams['axes.facecolor'] = 'white'
            plt.rcParams['savefig.facecolor'] = 'white'
            
            plt.figure(figsize=(3.5,5), dpi=300)

            if 'exhaustive_models' in more[0][hyper][0].keys():
      
                exhaustive_aggregate = np.stack(exhaustive_aggregate)

                if maximize_MEP:
                    exhaustive_aggregate = np.maximum(exhaustive_aggregate, 0)
                else:
                    exhaustive_aggregate = np.minimum(exhaustive_aggregate, 1)
                    exhaustive_aggregate = exhaustive_aggregate - 1

                exhaustive_per_participant = np.mean(exhaustive_aggregate, axis=1)

                exhaustive_aggregate_mean = np.mean(exhaustive_per_participant,axis=0)

                if n_datasets > 1:
                    exhaustive_aggregate_sem = (np.std(exhaustive_per_participant, axis=0, ddof=1) / np.sqrt(n_datasets))
                else:
                    exhaustive_aggregate_sem = (np.std(exhaustive_aggregate[0], axis=0, ddof=1) / np.sqrt(n_repeats))

                plt.plot(exhaustive_aggregate_mean,label='Exhaustive', color='#de6866', linewidth=3)
                plt.fill_between(np.arange(len(exhaustive_aggregate_mean)),
                        exhaustive_aggregate_mean - exhaustive_aggregate_sem,
                        exhaustive_aggregate_mean + exhaustive_aggregate_sem,
                        color='#de6866', alpha=0.3, linewidth=0)
                
            if 'greedy_models' in more[0][hyper][0].keys():

                greedy_aggregate = np.stack(greedy_aggregate)

                if maximize_MEP:
                    greedy_aggregate = np.maximum(greedy_aggregate, 0)
                else:
                    greedy_aggregate = np.minimum(greedy_aggregate, 1)
                    greedy_aggregate = greedy_aggregate - 1

                greedy_per_participant = np.mean(greedy_aggregate, axis=1)

                greedy_aggregate_mean = np.mean(greedy_per_participant,axis=0)

                if n_datasets > 1:
                    greedy_aggregate_sem = (np.std(greedy_per_participant, axis=0, ddof=1) / np.sqrt(n_datasets))
                else:
                    greedy_aggregate_sem = (np.std(greedy_aggregate[0], axis=0, ddof=1) / np.sqrt(n_repeats))

                plt.plot(greedy_aggregate_mean,label='Greedy', color='#f6dc8a', linewidth=3)
                plt.fill_between(np.arange(len(greedy_aggregate_mean)),
                        greedy_aggregate_mean - greedy_aggregate_sem,
                        greedy_aggregate_mean + greedy_aggregate_sem,
                        color='#f6dc8a', alpha=0.3, linewidth=0)

            values = [ str(i) for i in hyper_values]
            last_per_value = gpbo_mean[:,-1]

            # plot only the best trace
            if maximize_MEP:
                idx_best = last_per_value.argmax()
            else:
                idx_best = last_per_value.argmin()

            plt.plot(gpbo_mean[idx_best],label='GPBO',color='#5799d1',linewidth=3)
            plt.fill_between(np.arange(gpbo_mean.shape[1]),
                        gpbo_mean[idx_best] - gpbo_sem[idx_best],
                        gpbo_mean[idx_best] + gpbo_sem[idx_best],
                        color='#5799d1', alpha=0.3, linewidth=0)
            legend = plt.legend(
                title_fontsize=10,
                fontsize=8,
                loc='lower right',
                frameon=False,
            )

            if maximize_MEP:
                plt.ylim(0, 1)
            else:
                plt.ylim(-1, 0)
            plt.title(f'Exploration performance {hyper} \n ' f'Best {values[idx_best]} with performance of {last_per_value[idx_best]:.3f}', fontsize=10)
            plt.xlabel("Queries", fontsize=10)
            plt.ylabel("Performance", fontsize=10)
            plt.xticks(fontsize=8)
            plt.yticks(fontsize=8)
            plt.setp(legend.get_title(), fontweight='bold')
            ax = plt.gca()
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            plt.tight_layout()
            plt.savefig(path_aggregate+f'/aggregate_{model_idx}.png',dpi=300)
            plt.close('all')

            if len(hyper_values) > 1:
            # generate calibration summary figure

                mean_per_value = gpbo_mean.mean(1)

                plt.plot(values,mean_per_value,label='Mean',c='orange')
                plt.plot(values,last_per_value,label='Last',c='purple')

                if maximize_MEP:
                    best_idx = last_per_value.argmax()
                    print(f'Best (argmax) value of {hyper} is {values[best_idx]}'
                        f' with performance of {last_per_value[best_idx]}')
                else:
                    best_idx = last_per_value.argmin()
                    print(f'Best (argmin) value of {hyper} is {values[best_idx]}'
                        f' with performance of {last_per_value[best_idx]}')

                if 'exhaustive_models' in more[0][hyper][0].keys():

                    plt.plot(values,[exhaustive_aggregate_mean.mean()]*len(values),label='Exhaustive Mean',c='red')
                    plt.plot(values,[exhaustive_aggregate_mean[-1]]*len(values),label='Exhaustive Last',c='blue')

                if 'greedy_models' in more[0][hyper][0].keys():

                    plt.plot(values,[greedy_aggregate_mean.mean()]*len(values),label='Greedy Mean',c='green')
                    plt.plot(values,[greedy_aggregate_mean[-1]]*len(values),label='Greedy Last',c='gray')

                plt.legend(frameon=False)
                plt.xlabel('Values')
                if maximize_MEP:
                    plt.ylim(0, 1)
                else:
                    plt.ylim(-1, 0)
                plt.ylabel('Performance')
                plt.title(f'Exploration performance {hyper} \n 'f'Best {values[best_idx]} with performance of {last_per_value[best_idx]}')
                plt.grid(True)

                from matplotlib.ticker import MultipleLocator
                ax = plt.gca() 
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.xaxis.set_major_locator(MultipleLocator(1))  
                ax.xaxis.set_minor_locator(MultipleLocator(0.2)) 
                ax.yaxis.set_major_locator(MultipleLocator(0.1))  
                ax.yaxis.set_minor_locator(MultipleLocator(0.02)) 

                plt.grid(which='both', linestyle='--', linewidth=0.5)
                plt.minorticks_on()
                plt.savefig(path_aggregate+f'/summary_{model_idx}_{hyper}.png', dpi=1000)
                plt.close('all')




if __name__ == '__main__':
    
    # ============================================================
    # USER SETTINGS
    # ============================================================

    # Experiment:
    #   'ISI_only'  -> optimize ISI only
    #   'Cint_only' -> optimize conditioning-stimulus intensity only
    #   'Both'      -> optimize ISI and conditioning-stimulus intensity
    exp_type_ = 'ISI_only'

    # Optimization objective:
    #   True  -> find parameters that maximize MEP (facilitation search)
    #   False -> find parameters that minimize MEP (inhibition search)
    maximize_MEP = False

    # Run mode:
    #   True  -> sequentially calibrate hyperparameters (see CALIBRATION SETTINGS below)
    #   False -> run validation using previously calibrated hyperparameters
    Calibrate = False

    # Participants to include
    #   When 'ISI_only' or 'Cint_only' -> ['P1', 'P2', 'P3', 'P4', 'P5']
    #   When         'Both'            -> ['P1', 'P2', 'P3', 'P4']
    #    To validate one participant, set potential_experiments = ['P1'] or ['P2'] or ['P3'] or ['P4'] or ['P5']
    potential_experiments = ['P1', 'P2', 'P3', 'P4', 'P5']

    # Total number of TMS queries
    # When calibrate = False:
    #   1D (ISI_only/Cint_only): 30, 40, or 50
    #   2D (Both):               60, 80, or 100
    total_queries = 30

    # Number of stochastic repetitions per participant
    temp_repeats = 100

    # ============================================================
    # END OF USER SETTINGS
    # ============================================================


    if maximize_MEP:
        common_path = h.setup_folder('Results/Hyper_opt/Maximization')
    else:
        common_path = h.setup_folder('Results/Hyper_opt/Minimization')

    for exp_idx_, PID in enumerate(tqdm(potential_experiments,desc="Experiments", total=len(potential_experiments))):
        
        experiment_id = f'dataset_{PID}_exp_type_{exp_type_}'
        overwrite_folder = f'{common_path}/{experiment_id}'

        if not os.path.exists(overwrite_folder):
            os.makedirs(overwrite_folder)

        if exp_type_ == 'ISI_only':

            config_1 = conf.Config(f'./Config/ISI_only/hyperparameter_calibration_datasets/{PID}/ISI_only_config_1.json',overwrite_folder)
        
        elif exp_type_ == 'Cint_only':

            config_1 = conf.Config(f'./Config/Cint_only/hyperparameter_calibration_datasets/{PID}/Cint_only_config_1.json',overwrite_folder)

        elif exp_type_ == 'Both':

            config_1 = conf.Config(f'./Config/Both/hyperparameter_calibration_datasets/{PID}/Both_config_1.json',overwrite_folder)

        config_1.temp_data_storage_path = os.path.join('Data',f"{experiment_id}.p")

        config_1.repeats = temp_repeats

        config_1.verbose = True

        config_1.maximization = maximize_MEP

        config_1.max_it = total_queries

        if Calibrate:
            
            # ============================================================
            # CALIBRATION SETTINGS
            # ============================================================
            # Set 'calibrate = True' for ONE hyperparameter at a time.
            # After each calibration step, replace its default value with
            # the optimized value before calibrating the next hyperparameter.   
                     
            config_1.kappa['default'] = 3 ; config_1.kappa['calibrate'] = True ; config_1.kappa['values'] = [ 3,  5, 9, 13, 15, 19, 23]

            config_1.rho_min['default'] = 0.001 ; config_1.rho_min['calibrate'] = False ; config_1.rho_min['values'] = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1]

            config_1.rho_max['default'] = 1.1 ; config_1.rho_max['calibrate'] = False ; config_1.rho_max['values'] = [1.1, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

            config_1.noise_min['default'] = 0.0001 ; config_1.noise_min['calibrate'] = False ; config_1.noise_min['values'] = [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 0.7]

            config_1.noise_max['default'] = 0.8 ; config_1.noise_max['calibrate'] = False ; config_1.noise_max['values'] = [ 0.8, 1, 1.5, 2, 5, 7, 10]

            config_1.random_pts['default'] = 2 ; config_1.random_pts['calibrate'] = False ; config_1.random_pts['values'] = [2, 5, 10, 15, 20] 
        
        else:

            if maximize_MEP:
                objective_folder = "MEP_maximization"
            else:
                objective_folder = "MEP_minimization"

            exp_type_folder = {
                "ISI_only": "ISI_only",
                "Cint_only": "CSI_only",
                "Both": "Both",
            }[exp_type_]

            calibrated_config_path = os.path.join(
                "calibrated_hyperparameters",
                objective_folder,
                exp_type_folder,
                f"{config_1.max_it}_queries",
                f"dataset_{PID}.json",
            )

            with open(calibrated_config_path, "r") as f:
                calibrated_config = json.load(f)

            print(f"Loaded calibrated hyperparameters from:")
            print(calibrated_config_path)

            config_1.kappa["default"] = calibrated_config["kappa"] ; config_1.kappa["calibrate"] = True ; config_1.kappa["values"] = [calibrated_config["kappa"]]

            config_1.rho_min["default"] = calibrated_config["rho_min"] ; config_1.rho_min["calibrate"] = False

            config_1.rho_max["default"] = calibrated_config["rho_max"] ; config_1.rho_max["calibrate"] = False
            
            config_1.noise_min["default"] = calibrated_config["noise_min"] ; config_1.noise_min["calibrate"] = False

            config_1.noise_max["default"] = calibrated_config["noise_max"] ; config_1.noise_max["calibrate"] = False

            config_1.random_pts["default"] = calibrated_config["random_pts"] ; config_1.random_pts["calibrate"] = False


        system_1 = syst.System(config_1)

        plt.plot(system_1.ground_truth[0])
        plt.savefig(f'{config_1.save_folder}/ground_truth.png',dpi=300)
        plt.close('all')

        ## Run GP-BO
        calibrate_hyperparameter_non_hier(system_1, config_1, Calibrate)

    ## Aggregating performance over all datasets evaluated
    aggregate_curves(common_path, maximize_MEP=maximize_MEP)
