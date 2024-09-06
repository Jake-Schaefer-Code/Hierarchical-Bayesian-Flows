"""

"""
import torch
import math
import numpy as np
from scipy.stats import norm

from torchdyn.datasets import generate_moons



# ===================== #
# Probability Functions #
# ===================== #

def gaussian_log_prob(u: torch.Tensor, mu: torch.Tensor):
    """
    For a normal distribution, the log-probability is given by
    `log p_U(u)=-\frac{1}{2}u^Tu-\frac{d}{2}log(2 pi)`
    """
    input_dim = u.size(1)
    return -0.5 * ((u - mu) ** 2).sum(dim=1) - 0.5 * input_dim * torch.log(torch.tensor(2 * np.pi))

def uniform_log_prob(u: torch.Tensor):
    """
    For a uniform distribution over $[0, 1]^d$, the log-probability is constant
    `log p_U(u)=0`
    """
    return torch.zeros(u.size(0))


def logprob_moons(X: np.ndarray, n_samples: int = 100, noise: float = 1e-4) -> np.ndarray:
    """
    Computes the log-probability of each point in X under the moons distribution.
    
    :param X: array of shape (n_samples, 2), where each row is a point in 2D space
    :param n_samples: number of dataset points in the generated dataset
    :param noise: standard deviation of noise added to each dataset point
    :return: array of log-probabilities of each point in X
    """
    n_samples_out = n_samples // 2
    n_samples_in = n_samples - n_samples_out
    
    # Create the original moons without noise
    outer_circ_x = np.cos(np.linspace(0, np.pi, n_samples_out))
    outer_circ_y = np.sin(np.linspace(0, np.pi, n_samples_out))
    inner_circ_x = 1 - np.cos(np.linspace(0, np.pi, n_samples_in))
    inner_circ_y = 1 - np.sin(np.linspace(0, np.pi, n_samples_in)) - 0.5

    # Stack the points into the moons structure
    X_clean = np.vstack([np.append(outer_circ_x, inner_circ_x),
                         np.append(outer_circ_y, inner_circ_y)]).T
    
    # Compute log-probabilities of each point
    log_probs = np.zeros(X.shape[0])
    
    for i, x in enumerate(X):
        # Find the closest point in the moons structure to the sample
        dist_to_modes = np.linalg.norm(X_clean - x, axis=1)
        closest_mode_idx = np.argmin(dist_to_modes)
        closest_mode = X_clean[closest_mode_idx]
        
        # Assuming each point is Gaussian around its mode
        log_prob_x = norm.logpdf(x[0], loc=closest_mode[0], scale=noise)
        log_prob_y = norm.logpdf(x[1], loc=closest_mode[1], scale=noise)
        
        log_probs[i] = log_prob_x + log_prob_y
    
    return log_probs




class Likelihood(object):
    """
    """
    def log_likelihood(self, theta):
        raise NotImplementedError("log_likelihood() should be implemented in subclass.")

    def __call__(self, theta):
        return self.log_likelihood(theta)
    
class LogProbFactory:
    @staticmethod
    def get_log_prob(distribution_type: str):
        if distribution_type == "gaussian":
            return gaussian_log_prob
        elif distribution_type == "uniform":
            return uniform_log_prob
        # Add other distributions here
        else:
            raise ValueError(f"Unsupported distribution type: {distribution_type}")
        
