"""

"""
import torch
import torch.nn as nn
from torchdiffeq import odeint, odeint_adjoint
from torch.nn.utils import spectral_norm
import torch.nn.utils as utils
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sph_harm_torch import *
from distributions import *

import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import numpy as np



class ODEVF(nn.Module):
    """
    Architecture
    ----------------
    Use a sequence of residual blocks where each 
    block is parameterized by a neural network that defines the 
    vector field for the ODE.

    Methods
    ----------------
    `__init__`: Initializies the layers of the network. `input_dim` is the dimension of our data and `hidden_dim`
    defines the size of the hidden layers

    `forward`: Defines how the data flows through the network. The `forward` method outputs the vector field at
    any given time $t$ for the input data $x$
    """
    def __init__(self, 
                 input_dim: int, 
                 hidden_dim: int):
        super(ODEVF, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, t, x):
        return self.net(x)
    
class SpectralODEVF(nn.Module):
    """
    Architecture
    ----------------
    Basically just ODEVF, but with spectrally normalized layers
    
    Spectral normalization limits the largest singular value of the 
    weight matrices in the neural network layers, effectively 
    controlling the maximum amount by which the input can be scaled. 
    This helps ensure that the transformation remains stable and does 
    not exhibit extreme sensitivity to small changes in the input.

    Methods
    ----------------
    `__init__`: 
    
    `forward`:
    """
    def __init__(self, 
                 input_dim: int, 
                 hidden_dim: int):
        super(SpectralODEVF, self).__init__()
        self.net = nn.Sequential(
            utils.spectral_norm(nn.Linear(input_dim, hidden_dim)),
            nn.ReLU(),
            utils.spectral_norm(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
            utils.spectral_norm(nn.Linear(hidden_dim, input_dim))
        )

    def forward(self, t, x):
        return self.net(x)

class ResidualBlock(nn.Module):
    """
    Architecture
    ----------------
    Residual connections allow the model to learn perturbations 
    around the identity map, which inherently encourages the model 
    to produce small, incremental transformations rather than 
    drastic changes. This can act as a form of implicit regularization, 
    leading to smoother, more stable transformations.

    Methods
    ----------------
    `__init__`: 
    
    `forward`:
    """
    def __init__(self, 
                 input_dim: int):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.fc2 = nn.Linear(input_dim, input_dim)

    def forward(self, t, x):
        identity = x
        out = torch.relu(self.fc1(x))
        out = self.fc2(out)
        return identity + out

class CNF(nn.Module):
    """
    Architecture
    ----------------
    
    Methods
    ----------------
    `__init__`: Initializes the vector field
    
    `forward`: Solves the ODE from the initial time $t_0$ to the final time $T$ (typically [0,1]), 
    transforming the input data $x$. The `odeint` function from `torchdiffeq` performs the 
    numerical integration
    """
    def __init__(self, vector_field, solver_method='rk4', t_span=torch.tensor([0.0, 1.0])):
        """
        dim: number of dimensions of the data
        """
        super(CNF, self).__init__()
        self.vector_field = vector_field
        self.solver_method = solver_method
        self.t_span = t_span

    def forward(self, x0):
        """
        Integrates the vector field over the time span
        """
        x = odeint(self.vector_field, x0, self.t_span, method=self.solver_method)
        return x[-1]
    



# Define the ODE function for the CNF
def odefunc(t, x, vector_field) -> torch.Tensor:
    return vector_field(x, t)

def flow_matching_loss(x_0:torch.Tensor, x_t:torch.Tensor, vector_field:nn.Module, t_span:torch.Tensor=torch.tensor([0.0, 1.0])) -> torch.Tensor:
    """
    Computes the flow matching loss by comparing the final state of the ODE with target samples.
    
    Parameters:
    ---------------
    x_0 : torch.Tensor
        Initial state samples
    x_t : torch.Tensor
        Target state samples
    vector_field : nn.Module
        The vector field model used in the ODE

    Returns:
    ---------------
    torch.Tensor
        Mean squared error between predicted and target samples
    """

    x_pred = odeint(odefunc, x_0, t_span, args=(vector_field,))[-1]
    return ((x_pred - x_t) ** 2).mean()



def compute_log_likelihood(
    u: torch.Tensor, 
    vector_field: nn.Module, 
    base_log_prob_func: callable, 
    log_prob_func: callable,
    t_span: torch.Tensor = torch.tensor([0.0, 1.0]),
    logl_args: list = []
) -> torch.Tensor:
    """
    Computes log-likelihood using change of variables formula:

    logp_X(x) = logp_T(z) + logp_U(u) - log|det(J)|
    
    Parameters
    ----------------
    base_log_prob_func : callable
        Computes log-probability of base distribution
    log_prob_func : callable
        Custom log-likelihood function - assuming that this is not NLL, but just LL
    """
    batch_size, input_dim = u.size()

    # Compute the ODE solution (forward integration)
    z_t, log_jac_det = odeint_adjoint(odefunc, (u, torch.zeros(batch_size,)), t_span, args=(vector_field,))
    
    z = z_t[-1]  # Final state
    log_det_J = log_jac_det[-1]  # Accumulated log-det of the Jacobian
    log_p_U = base_log_prob_func(u)
    log_p_T = log_prob_func(z, *logl_args)
    
    nll = - (log_p_U + log_p_T - log_det_J)
    return nll




flat_dist = GeneralDistribution(25, "flat")
x0_samples = flat_dist.sample(1000)

x1_samples = torch.randn(1000, 25)

# Create DataLoader
dataset = TensorDataset(x0_samples, x1_samples)
data_loader = DataLoader(dataset, batch_size=64, shuffle=True)


def centering_penalty(z, target_mean=0.0):
    return ((z.mean(dim=0) - target_mean) ** 2).sum()

def variance_penalty(z:torch.Tensor, target_var=1.0):
    return ((z.var(dim=0) - target_var) ** 2).sum()

def gradient_penalty(vector_field, x:torch.Tensor):
    x.requires_grad_(True)
    y = vector_field(0, x).sum()
    grad = torch.autograd.grad(outputs=y, inputs=x, create_graph=True)[0]
    return ((grad.norm(2, dim=1) - 1) ** 2).mean()

# Define the ODE function for the CNF
def odefunc(t, x, vector_field) -> torch.Tensor:
    return vector_field(x, t)

def flow_matching_loss(x_0:torch.Tensor, 
                       x_t:torch.Tensor, 
                       vector_field:nn.Module, 
                       t_span:torch.Tensor=torch.tensor([0.0, 1.0]), 
                       reg_lambda=1e-5,
                       penalty_lambda=10) -> torch.Tensor:
    """
    Computes the flow matching loss by comparing the final state of the ODE with target samples.
    
    Parameters:
    ---------------
    x_0 : torch.Tensor
        Initial state samples
    x_t : torch.Tensor
        Target state samples
    vector_field : nn.Module
        The vector field model used in the ODE

    Returns:
    ---------------
    torch.Tensor
        Mean squared error between predicted and target samples
    """
    
    x_pred = odeint(vector_field, x_0, t_span)[-1]

    mse_loss = ((x_pred - x_t) ** 2).mean()
    kde = TorchKDE(x_t.T)
    kde_loss = kde(x_pred.T)

    # L2 regularization
    # l2_reg = sum(param.pow(2.0).sum() for param in vector_field.parameters())
    
    # gp = gradient_penalty(vector_field, x_0)
    penalty_lambda2 = 1e-3
    # penalty = variance_penalty(x_pred)
    penalty = centering_penalty(x_pred)
    
    # Total loss
    # loss = mse_loss + reg_lambda * l2_reg + penalty_lambda * gp + penalty_lambda2 * penalty
    # loss = mse_loss #+ reg_lambda * l2_reg + penalty_lambda2 * penalty
    # loss = mse_loss + penalty_lambda2 * penalty
    loss = -torch.log(torch.mean(kde_loss))
    return loss


def fml_kde(x_0:torch.Tensor, 
            x_t:torch.Tensor, 
            vector_field:nn.Module, 
            t_span:torch.Tensor = torch.tensor([0.0, 1.0])) -> torch.Tensor:
    """
    Flow matching loss with kde loss function
    """
    x_pred = odeint(vector_field, x_0, t_span)[-1]
    kde = TorchKDE(x_t.T, bandwidth="scott")
    kde_loss = kde(x_pred.T)
    loss = -torch.log(torch.mean(kde_loss))
    return loss

def fml_mse(x_0:torch.Tensor, 
            x_t:torch.Tensor, 
            vector_field:nn.Module, 
            t_span:torch.Tensor = torch.tensor([0.0, 1.0])) -> torch.Tensor:
    """
    Flow matching loss with mse loss function
    """
    x_pred = odeint(vector_field, x_0, t_span)[-1]
    mse_loss = ((x_pred - x_t) ** 2).mean()
    loss = mse_loss
    return loss

def gaussian_kernel(x, y, sigma=1.0):
    dist = torch.cdist(x, y) ** 2
    return torch.exp(-dist / (2 * sigma ** 2))

def fml_mmd(x_0:torch.Tensor, 
            x_t:torch.Tensor, 
            vector_field:nn.Module, 
            t_span:torch.Tensor = torch.tensor([0.0, 1.0]),
            kernel=gaussian_kernel) -> torch.Tensor:

    x_pred = odeint(vector_field, x_0, t_span)[-1]
    xx = kernel(x_pred, x_pred)
    yy = kernel(x_t, x_t)
    xy = kernel(x_pred, x_t)

    return torch.mean(xx) + torch.mean(yy) - 2 * torch.mean(xy)

def kde_kl_divergence(x_0:torch.Tensor, 
                      x_t:torch.Tensor, 
                      vector_field:nn.Module, 
                      t_span:torch.Tensor = torch.tensor([0.0, 1.0])):
    x_pred = odeint(vector_field, x_0, t_span)[-1]
    Tu_kde = TorchKDE(x_pred.T, bandwidth="scott")
    x_kde = TorchKDE(x_t.T, bandwidth="scott")
    return torch.mean(Tu_kde(x_pred.T) - x_kde(x_pred.T))

def compute_log_likelihood(
    u: torch.Tensor, 
    vector_field: nn.Module, 
    base_log_prob_func: callable, 
    log_prob_func: callable,
    t_span: torch.Tensor = torch.tensor([0.0, 1.0]),
    logl_args: list = []) -> torch.Tensor:
    """
    Computes log-likelihood using change of variables formula:

    logp_X(x) = logp_T(z) + logp_U(u) - log|det(J)|
    
    Parameters
    ----------------
    base_log_prob_func : callable
        Computes log-probability of base distribution
    log_prob_func : callable
        Custom log-likelihood function - assuming that this is not NLL, but just LL
    """
    batch_size, input_dim = u.size()

    # Compute the ODE solution (forward integration)
    z_t, log_jac_det = odeint_adjoint(odefunc, (u, torch.zeros(batch_size,)), t_span, args=(vector_field,))
    
    z = z_t[-1]  # Final state
    log_det_J = log_jac_det[-1]  # Accumulated log-det of the Jacobian
    log_p_U = base_log_prob_func(u)
    log_p_T = log_prob_func(z, *logl_args)
    
    nll = - (log_p_U + log_p_T - log_det_J)
    return nll

