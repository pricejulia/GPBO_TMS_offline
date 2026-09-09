import gpytorch
import torch
import math
import numpy as np
from gpytorch.means import Mean


class OneMean(Mean):
    def __init__(self, batch_shape=torch.Size(), **kwargs):
        super(OneMean, self).__init__()
        self.batch_shape = batch_shape

    def forward(self, input):
        mean = torch.ones(*self.batch_shape, 1, dtype=input.dtype, device=input.device)
        if input.shape[:-2] == self.batch_shape:
            return mean.expand(input.shape[:-1])
        else:
            return mean.expand(torch.broadcast_shapes(input.shape[:-1], mean.shape))
        

class GP(gpytorch.models.ExactGP):
    """Class that extends ExactGP for GPU-based training of Gaussian Process models"""
    def __init__(self, train_x, train_y, likelihood, kernel, maximization=True):
        super(GP, self).__init__(train_x, train_y, likelihood)

        if maximization:

            self.mean_module = gpytorch.means.ZeroMean()

        else: #minimization

            self.mean_module = OneMean()

        # Initialize covariance module
        self.covariance_module = kernel

    def forward(self, x):

        mean_x = self.mean_module(x)
        covariance_x = self.covariance_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covariance_x)
    

def optimize(model, likelihood, training_iter, train_x, train_y, learning_rate=0.01):
    """Optimizes GP model with the Adam optimizer and marginal log likelihood loss"""
    # Setup optimizer which will optimize GaussianLikelihood parameters
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Loss criterion for GPs is chosen as the marginal log likelihood (MLL)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()

        # Forward pass to get output from model
        output = model(train_x)

        # Calculate loss and backpropagate gradients
        loss = -mll(output, train_y).mean()
        loss.backward()
        optimizer.step()

    return model, likelihood



class GP_MODEL:
    def __init__(self,X,Y,system,config):
        
        self.config = config

        self.search_space = system.search_space

        self.n_dim  = system.search_space.shape[1]

        self.predictions = []

        self.search_space_preddictions = torch.tensor(self.search_space, dtype=torch.float).to(self.config.torch_device)

        # Define the kernel and set priors on its parameters

        self.priorbox = gpytorch.priors.SmoothedBoxPrior(a=math.log(config.rho_min['default']),
                                                    b=math.log(config.rho_max['default']),
                                                    sigma=0.01)
        self.priorbox2 = gpytorch.priors.SmoothedBoxPrior(a=math.log(0.01 ** 2),
                                                        b=math.log(100.0 ** 2),
                                                        sigma=0.01)
        kernel = gpytorch.kernels.MaternKernel(nu=2.5,
                                                ard_num_dims=self.n_dim,
                                                lengthscale_prior=self.priorbox)
        
        self.kernel = gpytorch.kernels.ScaleKernel(kernel, outputscale_prior=self.priorbox2)

        self.kernel.base_kernel.lengthscale = [1.0] * self.n_dim
        self.kernel.outputscale = [1.0]

        self.prior_likelihood = gpytorch.priors.SmoothedBoxPrior(a=config.noise_min['default'] ** 2,
                                                            b=config.noise_max['default'] ** 2,
                                                            sigma=0.01)
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_prior=self.prior_likelihood)
        self.likelihood.noise = [1.0]

        self.model = GP(train_x=torch.tensor(X, dtype=torch.float).to(self.config.torch_device),
                    train_y=torch.tensor(Y, dtype=torch.float).squeeze(-1).to(self.config.torch_device),
                    likelihood=self.likelihood,
                    kernel=self.kernel,
                    maximization=config.maximization)
        
        self.model = self.model.to(self.config.torch_device)

        self.model.covariance_module.base_kernel.lengthscale = [self.model.covariance_module.base_kernel.lengthscale[0][i].item() for i in range(self.n_dim)]
        self.model.covariance_module.base_kernel.outputscale = self.model.covariance_module.outputscale.item()
        self.model.likelihood.noise = self.model.likelihood.noise[0].item()

        if self.config.offline:
            
            self.ground_truth = system.ground_truth

            self.exploration_performance = {'ground_truth':self.ground_truth[0],'curve':[]}

            if system.random_exploration_performance is not None:

                self.exploration_performance['curve'] = system.random_exploration_performance
        
    def step(self,X,Y):

        self.model.set_train_data(inputs=torch.tensor(X, dtype=torch.float).to(self.config.torch_device),
                                targets=torch.tensor(Y, dtype=torch.float).squeeze(-1).to(self.config.torch_device),
                                strict=False)

        self.model.train()
        self.likelihood.train()
        
        self.model, self.likelihood = optimize(model=self.model,
                                        likelihood=self.likelihood,
                                        training_iter=10,
                                        train_x=torch.tensor(X, dtype=torch.float).to(self.config.torch_device),
                                        train_y=torch.tensor(Y, dtype=torch.float).squeeze(-1).to(self.config.torch_device))

    # Predict 
    def compute_predictions(self):

        self.model.eval()
        self.likelihood.eval()
        
        with torch.no_grad():

            y = self.likelihood(self.model(self.search_space_preddictions))
            y_mu = y.mean.cpu().numpy()
            y_var = y.variance.cpu().numpy()

        return y_mu, y_var

    def nextX(self):

        y_mu, y_var = self.compute_predictions()

        if self.config.maximization:

            self.exploration_performance['curve'].append(self.ground_truth[0][y_mu.argmax()])

            self.predictions.append([y_mu, y_var])

            ucb = y_mu + self.config.kappa['default'] * np.nan_to_num(np.sqrt(y_var))

            return ucb.flatten().argmax()

        else: #minimization

            self.exploration_performance['curve'].append(self.ground_truth[0][y_mu.argmin()])

            self.predictions.append([y_mu, y_var])

            ucb = y_mu - self.config.kappa['default'] * np.nan_to_num(np.sqrt(y_var))

            return ucb.flatten().argmin()

        
    def log(self,id=''):

        filename = self.config.save_folder+f'/model_{id}_solutions.md'

        if self.config.maximization:

            solution_idx = self.predictions[-1][0].argmax()

        else:

            solution_idx = self.predictions[-1][0].argmin()

        value_ = (self.ground_truth[0][solution_idx ] - self.ground_truth[0].mean()) / (self.ground_truth[0].max() - self.ground_truth[0].mean())

        text_to_append = f'Solution is { self.search_space[solution_idx]} with ground truth of {value_ }'

        try:
            with open(filename, "a") as file: 
                file.write(text_to_append + "\n")  
        except Exception as e:
            print(f"An error occurred: {e}")
