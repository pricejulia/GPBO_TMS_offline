import numpy as np
import os
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm
from datetime import datetime

def datestamp():
    return datetime.now().strftime("%Y_%m_%d_%H_%M_%S")


def elec_to_mean_value(mean_ground_truth):
    
    def elec_to_mean_value_(index, mean=mean_ground_truth):
        return mean[index]
    
    return np.vectorize(elec_to_mean_value_)


def random_search(system,config,max_it):
    def get_chunk(index, chunk_size, vector):

        start = index * chunk_size
        end = start + chunk_size
        return vector[start:end]

    uni = np.arange(system.search_space.shape[0])

    stim_counter = [0 for i in range(uni.shape[0])]

    random_trial_order = np.stack([np.random.permutation(np.arange(config.NUM_TRIALS)) for i in range(system.search_space.shape[0])])

    model = [[] for i in range(uni.shape[0])]

    queries = uni.tolist() * config.NUM_TRIALS

    np.random.shuffle(queries)
    exploration_performance = []

    if len(queries) < max_it:

        queries = queries + queries

    for query in queries[:max_it]:

        y = np.nan

        while np.isnan(y).all():
            
            y = get_chunk(query,config.NUM_TRIALS,system.Y)[random_trial_order[query,stim_counter[query]]]

            stim_counter[query] = (stim_counter[query] + 1) % config.NUM_TRIALS

        model[query].append(y)

        if config.maximization:

            model_ = [np.mean(i) if len(i) >= 1 else 0 for i in model]

            exploration_performance.append(np.argmax(model_))
        
        else: 

            model_ = [np.mean(i) if len(i) >= 1 else 1 for i in model]

            exploration_performance.append(np.argmin(model_))

    return exploration_performance, model_


def run_random_search(system,config,overwrite_repeats = None,overwrite_max_it=None):

    if overwrite_repeats is None:

        repeats = config.repeats
    
    else:

        repeats = overwrite_repeats
    
    if overwrite_max_it is None:

        max_it = config.max_it

    else:
        
        max_it = overwrite_max_it

    elec_to_mean = elec_to_mean_value(system.ground_truth[0])

    random_results = [random_search(system,config,max_it) for i in range(repeats)]

    exhau = np.stack([i[0] for i in random_results])

    models = np.stack([i[1] for i in random_results])

    exhau = elec_to_mean(exhau)

    if exhau.shape[0]==1: #making sure that when single model is returned, single vector is returned
        exhau = exhau[0,:]

    return {'curve' : exhau, 'ground_truth':system.ground_truth[0], 'model':models }


def greedy_search(system,config):

    model = np.full((system.search_space[-1][0]+1,system.search_space[-1][1]+1, config.max_it), np.nan)

    uni = np.arange(system.search_space.shape[0])

    stim_counter = [0 for i in range(uni.shape[0])]

    random_trial_order = np.stack([np.random.permutation(np.arange(config.NUM_TRIALS)) for i in range(system.search_space.shape[0])])

    def get_chunk(index, chunk_size, vector):

        start = index * chunk_size
        end = start + chunk_size
        return vector[start:end]

    def invert(x):
        return 1 - x

    size_axis = [system.search_space[-1][0]+1,system.search_space[-1][1]+1]

    intersection = [np.random.randint(0, size_axis[0]), np.random.randint(0, size_axis[1])]

    # Pick order at random (axis_1,axis_2)
    
    order = np.random.permutation([0, 1])

    query_count = []
    axis_identity = []
    i = 0

    while np.sum(query_count) <= config.max_it:

        query_count.append(size_axis[order[i%2]])
        axis_identity.append(order[i%2])

        i+=1

    query_count[-1] = config.max_it - np.sum(query_count[:-1])

    itt = 0 
    
    for idx,ax_idx in enumerate(axis_identity):

        queries_slice = np.zeros((query_count[idx],2))

        queries_slice[:,invert(ax_idx)] = intersection[invert(ax_idx)]

        queries_slice[:,ax_idx] = np.random.choice(size_axis[ax_idx], size = query_count[idx], replace=False)
     
        for qu in queries_slice:
            
            query = np.ravel_multi_index((int(qu[0]),int(qu[1])), size_axis)

            y = np.nan

            while np.isnan(y).all():
                
                y = get_chunk(query,config.NUM_TRIALS,system.Y)[random_trial_order[query,stim_counter[query]]]

                stim_counter[query] = (stim_counter[query] + 1) % config.NUM_TRIALS

            model[int(qu[0]),int(qu[1]),itt] = y

            itt += 1
    
        if ax_idx == 0:

            sliced = model[:, intersection[invert(ax_idx)], :]

        else:

            sliced = model[intersection[invert(ax_idx)], :, :]

        if config.maximization:
        
            intersection[ax_idx] =  np.nanargmax(np.nanmean(sliced, axis=-1))

        else:
            
            intersection[ax_idx] =  np.nanargmin(np.nanmean(sliced, axis=-1))

    exploration_performance = []

    for i in range(1,config.max_it+1):

        if config.maximization:

            model_ = np.nanmean(model[:,:,:i],axis=-1).flatten()

            exploration_performance.append(np.nanargmax(model_))

        else:

            model_ = np.nanmean(model[:,:,:i],axis=-1).flatten()
            
            exploration_performance.append(np.nanargmin(model_))

    return exploration_performance, model_


def run_greedy_search(system, config, overwrite_repeats=None):
    
    if overwrite_repeats is None:
        repeats = config.repeats
    else:
        repeats = overwrite_repeats

    greedy_results = [
        greedy_search(system, config)
        for _ in range(repeats)
    ]

    greedy = np.stack([i[0] for i in greedy_results])
    models = np.stack([i[1] for i in greedy_results])

    elec_to_mean = elec_to_mean_value(system.ground_truth[0])
    greedy = elec_to_mean(greedy)

    if greedy.shape[0] == 1:
        greedy = greedy[0, :]

    return {
        'curve': greedy,
        'ground_truth': system.ground_truth[0],
        'model': models
    }


def model_population(procedure,config): 

    models = []

    for i in tqdm(range(config.repeats)):

        models.append(procedure())

    def process_curves(curves):
        curves = np.stack(curves)
        curves_mu = curves.mean(0)
        curves_std = curves.std(0) / np.sqrt(curves.shape[0])

        return [curves_mu,curves_std]

    def stack_models(models):

        stacked_models = []
        for i in range(len(models[0])):
            temp_1 = []
            for j in range(config.repeats):

                temp_1.append(models[j][i])

            stacked_models.append(np.stack(temp_1))

        return stacked_models
    
    model_types  = ['gpbo_models']

    gpbo_models = stack_models([[submodel.exploration_performance['curve'] for submodel in m] for m in models])

    ground_truths = [[submodel.exploration_performance['ground_truth'] for submodel in m] for m in models]

    for idx,curves in enumerate(gpbo_models): 

        for rep,curve in enumerate(curves): 
            
            if config.maximization:

                gpbo_models[idx][rep] = (curve - ground_truths[rep][idx].mean()) / (ground_truths[rep][idx].max() - ground_truths[rep][idx].mean())

            else:

                gpbo_models[idx][rep] = 1 + ((curve - ground_truths[rep][idx].mean()) / (ground_truths[rep][idx].mean() - ground_truths[rep][idx].min()))

    gpbo_models_stats = [process_curves(i) for i in gpbo_models]

    if hasattr(models[0][0],'exhaustive_performance'):

        exhaustive_models = stack_models([[submodel.exhaustive_performance['curve'] for submodel in m] for m in models])

        ground_truths = [[submodel.exhaustive_performance['ground_truth'] for submodel in m] for m in models]

        for idx,curves in enumerate(exhaustive_models):
            
            for rep,curve in enumerate(curves): 

                if config.maximization:

                    exhaustive_models[idx][rep] = (curve - ground_truths[rep][idx].mean()) / (ground_truths[rep][idx].max() - ground_truths[rep][idx].mean()) 

                else:

                    exhaustive_models[idx][rep] = 1 + ((curve - ground_truths[rep][idx].mean()) / (ground_truths[rep][idx].mean() - ground_truths[rep][idx].min()))

        exhaustive_models_stats = [process_curves(i) for i in exhaustive_models]

        model_types  = ['gpbo_models','exhaustive_models']

        if hasattr(models[0][0],'greedy_performance'):

            greedy_models = stack_models([[submodel.greedy_performance['curve'] for submodel in m] for m in models])

            ground_truths = [[submodel.greedy_performance['ground_truth'] for submodel in m] for m in models]

            for idx,curves in enumerate(greedy_models):
                
                for rep,curve in enumerate(curves): #looping over replicates

                    if config.maximization:

                        greedy_models[idx][rep] = (curve - ground_truths[rep][idx].mean()) / (ground_truths[rep][idx].max() - ground_truths[rep][idx].mean()) 

                    else:

                        greedy_models[idx][rep] = 1 + ((curve - ground_truths[rep][idx].mean()) / (ground_truths[rep][idx].mean() - ground_truths[rep][idx].min()))

            greedy_models_stats = [process_curves(i) for i in greedy_models]

            model_types  = ['gpbo_models','exhaustive_models','greedy_models']

    stats_models = {}
    stacked_models = {} 
    for name in model_types:
        stacked_models[name] = locals()[name]
        stats_models[name] = locals()[name+'_stats']

    return stats_models,stacked_models


def plot_hyperparameter_calibration(learning_curves,config,hyperparameter,exh_mu=None):

    values = [ str(i) for i in getattr(config, hyperparameter)['values']]

    plt.plot(values,np.nanmean(np.where(learning_curves == 0, np.nan, learning_curves),axis=1),label='Mean',c='orange')
    plt.plot(values,learning_curves[:,-1],label='Last',c='purple')

    if exh_mu is not None:
        plt.plot(values,[exh_mu.mean()]*len(values),label='Exhaustive Mean',c='red')
        plt.plot(values,[exh_mu[-1]]*len(values),label='Exhaustive Last',c='blue')

    plt.legend()
    plt.xlabel('Values')
    plt.ylim(0,1)
    plt.ylabel('Performance')
    plt.title(f'Exploration performance {hyperparameter}')
    plt.grid(True)

    from matplotlib.ticker import MultipleLocator
    ax = plt.gca()
    ax.xaxis.set_major_locator(MultipleLocator(1)) 
    ax.xaxis.set_minor_locator(MultipleLocator(0.2))  
    ax.yaxis.set_major_locator(MultipleLocator(0.1))  
    ax.yaxis.set_minor_locator(MultipleLocator(0.02)) 

    plt.grid(which='both', linestyle='--', linewidth=0.5)
    plt.minorticks_on()
    plt.savefig(f'{config.save_folder}/learning_curves_{hyperparameter}_{datestamp()}.png', dpi=1000)
    plt.close('all')


def plot_non_hierarchical(learning_curves,labels,config,hyperparameter=None,hyperparameter_value=None):

    xsize = 8
    ysize = 5

    fig, ax = plt.subplots(figsize=(xsize, ysize))
    title = config.exp_type

    method_colors = {"Exhaustive": "#de6866","GPBO": "#5799d1","Greedy": "#f6dc8a",}

    max_num_queries = max(
        model[0][0].shape[0]
        for model in learning_curves
    )

    for model_type_idx, model in enumerate(learning_curves):

        label = labels[model_type_idx][0]
        color = method_colors[label]

        data_mu = model[0][0]
        data_std = model[0][1]

        x_ = np.arange(max_num_queries - data_mu.shape[0],max_num_queries)

        ax.plot(x_,data_mu,label=label,color=color)

        ax.fill_between(x_,data_mu - data_std,data_mu + data_std,color=color,alpha=0.3)

    ax.set_xlabel("Queries")
    ax.set_ylabel("Exploration Performance")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(title)
    ax.legend(frameon=False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.savefig(
        f'{config.save_folder}/learning_curves_{hyperparameter}_{hyperparameter_value}.png',
        dpi=500
    )

    plt.close(fig)


def setup_folder(parent_folder):
    
    temp_folders = glob.glob(f'{parent_folder}/temp*')

    temp_folders_numbering = [
        i.split('temp')[1]
        for i in temp_folders
    ]

    if len(temp_folders_numbering) == 0:
        temp_string = f'{parent_folder}/temp0'
    else:
        next_number = (
            np.array(temp_folders_numbering, dtype=np.int32).max() + 1
        )
        temp_string = f'{parent_folder}/temp{next_number}'

    os.makedirs(temp_string)

    return temp_string
