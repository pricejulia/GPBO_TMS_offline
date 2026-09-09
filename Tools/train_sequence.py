import numpy as np

def header():
    print(f'\nIteration ---- Query ------- Response range ------- Response ------ Normalized Response')

def print_it_TM_info(iteration, search_space, query, y, response_range):
    trailing = lambda x: str(x).zfill(2) if len(str(x)) < 2 else x
    format_floats = lambda x: [f"{i:.2f}" for i in x] if isinstance(x, (list,np.ndarray)) else f"{x:.2f}"

    print(
        f'    {trailing(iteration)}   ----   {str(search_space[query])} ----  {format_floats(response_range)}   ----     {format_floats(((((y)))*(response_range[1]-response_range[0]))+response_range[0])}     ---    {format_floats(y)}')
    
def train_model(system,config):

    if config.GP_library != 'GPy_torch':
        raise ValueError('Invalid GP library')

    import Model.GPy_torch as GP

    X_it = np.empty((0, system.search_space.shape[1]))

    Y_it = np.empty((0, 1))

    for query in np.random.choice(range(system.search_space.shape[0]), config.random_pts['default'], replace=True):

        X_it = np.concatenate([X_it,system.search_space[query][None,:]]) 

        if config.offline:

            Y_it = system.get_response(query)

            # compute ground_truth for random queries

            def average_by_unique_rows(X, Y):
                unique_rows, inverse = np.unique(X, axis=0, return_inverse=True)
                avg_Y = np.array([Y[inverse == i].mean() for i in range(len(unique_rows))])
                return unique_rows, avg_Y
            
            if Y_it is not None:

                temp_exploration_performance = []

                for i in range(len(Y_it)):

                    unique_rows, avg_Y = average_by_unique_rows(X_it[:i+1], Y_it[:i+1])

                    if config.maximization:

                        solution_idx = (system.search_space == unique_rows[avg_Y.argmax()]).sum(1).argmax()

                    else:

                        solution_idx = (system.search_space == unique_rows[avg_Y.argmin()]).sum(1).argmax()

                    temp_exploration_performance.append(system.ground_truth[0][solution_idx])

                system.random_exploration_performance = temp_exploration_performance
                
        else:

            Y_it = system.get_response(query)

    gp = GP.GP_MODEL(X_it, Y_it,system,config)

    if config.verbose:
        header()

    def step_function(iteration, X_it, Y_it,gp, system,config):

        query = gp.nextX()

        X_it = np.concatenate([X_it,system.search_space[query][None,:]]) 

        Y_it = system.get_response(query)

        if config.verbose:

            print_it_TM_info(iteration, system.search_space, query, Y_it[-1], [system.min_response,system.max_response])
            
        gp.step(X_it,Y_it)

        return X_it, Y_it, gp

    for i in range(config.max_it - config.random_pts['default']):

        X_it, Y_it, gp = step_function(i, X_it, Y_it, gp, system,config)

    return gp