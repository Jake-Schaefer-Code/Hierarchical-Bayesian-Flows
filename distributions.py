"""

"""
import torch
import math
import numpy as np
from scipy.stats import norm

from torchdyn.datasets import generate_moons
from scipy.stats import gaussian_kde

# Sampling Functions

def eight_normal_sample(n, dim, scale=1, var=1, to_numpy=False):
    """
    Sample drawing method
    """
    
    m = torch.distributions.multivariate_normal.MultivariateNormal(
        torch.zeros(dim), math.sqrt(var) * torch.eye(dim)
    )
    centers = [
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1.0 / math.sqrt(2), 1.0 / math.sqrt(2)),
        (1.0 / math.sqrt(2), -1.0 / math.sqrt(2)),
        (-1.0 / math.sqrt(2), 1.0 / math.sqrt(2)),
        (-1.0 / math.sqrt(2), -1.0 / math.sqrt(2)),
    ]
    centers = torch.tensor(centers) * scale
    noise = m.sample((n,))
    multi = torch.multinomial(torch.ones(8), n, replacement=True)
    data = centers[multi] + noise
    if to_numpy:
        return data.detach().numpy()
    return data

def sample_multimodal(nsamples: int, 
                      ndim: int, 
                      scale: float = 1,
                      var: float = 1, 
                      centers:torch.Tensor = torch.tensor([(0,0)]), 
                      to_numpy=False):
    """
    Params
    var = variance - should be int or float?
    centers: ArrayLike
    """
    m = torch.distributions.multivariate_normal.MultivariateNormal(
        torch.zeros(ndim), math.sqrt(var) * torch.eye(ndim)
    )
    centers = torch.tensor(centers) * scale
    noise = m.sample((nsamples,))
    multi = torch.multinomial(torch.ones(centers.size(0)), nsamples, replacement=True)
    data = centers[multi] + noise
    if to_numpy:
        return data.detach().numpy()
    return data

def sample_8gaussians(n):
    return eight_normal_sample(n, 2, scale=5, var=0.1).float()

def sample_moons(n):
    x0, _ = generate_moons(n, noise=0.2)
    return x0 * 3 - 1


# ===================== #
# Probability Functions #
# ===================== #

def gaussian_log_prob(u: torch.Tensor, mu: torch.Tensor=0, variance: float=0):
    """
    For a normal distribution, the log-probability is given by
    `log p_U(u)=-\frac{1}{2}u^Tu-\frac{d}{2}log(2 pi)`
    """
    input_dim = u.size(1)
    return -0.5 * ((u - mu) ** 2).sum(dim=1) / variance - 0.5 * input_dim * np.log(2 * np.pi * variance)

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

"""
    # registry = []
    def __init_subclass__(cls, **kwargs) -> None:
        pass
        # super().__init_subclass__(**kwargs)
        # if not hasattr(cls, 'required_method'):
        #     raise TypeError(f"{cls.__name__} must define 'required_method'")

        # cls.registry.append(cls)
"""

class Distribution(object):
    """
    """
    def sample(self, batch_size=1, *args, **kwargs) -> torch.Tensor:
        """
        """
        raise NotImplementedError("sample() should be implemented in subclass.")
    
    def log_prob(self, u: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        """
        raise NotImplementedError("log_prob() should be implemented in subclass.")


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
        


class UniformDistribution(Distribution):
    """
    Uniform distribution over [low_0, high_0] x ... x [low_n, high_n], where n is the dimension
    """
    def __init__(self, dim: int, low: torch.Tensor = 0, high: torch.Tensor = 1):
        """
        """
        self.low = torch.tensor(low) if not isinstance(low, torch.Tensor) else low
        self.high = torch.tensor(high) if not isinstance(high, torch.Tensor) else high
        self.dim = dim

        if self.low.size() != torch.Size([dim]):
            raise ValueError(f"Low bounds must have the same size as dim: {dim}")
        if self.high.size() != torch.Size([dim]):
            raise ValueError(f"High bounds must have the same size as dim: {dim}")

    def sample(self, batch_size=1):
        """
        """
        return torch.rand(batch_size, self.dim) * (self.high - self.low) + self.low

    def log_prob(self, u: torch.Tensor):
        """
        """
        inside_bounds = ((u >= self.low) & (u <= self.high)).all(dim=1)
        log_prob_value = torch.zeros_like(u[:, 0])  # Initialize log-prob as zeros
        log_prob_value[~inside_bounds] = -torch.inf  # Set log-prob to -inf for out-of-bound samples
        return log_prob_value
    
    def __call__(self, batch_size=1):
        """
        """
        return self.sample(batch_size)

class GaussianDistribution(Distribution):
    """
    """
    def __init__(self, dim, mean=0, variance=1):
        """
        variance = sigma^2
        """
        self.mean = mean
        self.variance = torch.tensor(variance)
        self.dim = dim

    def sample(self, batch_size=1):
        """
        """
        return torch.randn(batch_size, self.dim) * torch.sqrt(self.variance) + self.mean

    def log_prob(self, u: torch.Tensor):
        """
        """
        return gaussian_log_prob(u, self.mean, self.variance)
    
    def __call__(self, batch_size=1):
        """
        """
        return self.sample(batch_size)

class MultimodalDistribution(Distribution):
    """
    """
    def __init__(self, 
                 centers: torch.Tensor, 
                 variance: torch.Tensor = 1, 
                 scale: float = 1):
        """
        """
        self.centers = torch.tensor(centers) * scale
        self.variance = torch.tensor(variance)
        self.n_modes, self.dim = centers.size(0), centers.size(1)

    def sample(self, nsamples: int):
        """
        """
        noise = torch.randn(nsamples, self.dim) * torch.sqrt(self.variance)
        multi = torch.multinomial(torch.ones(self.n_modes), nsamples, replacement=True)
        return self.centers[multi] + noise

    def log_prob(self, u: torch.Tensor):
        """
        """
        log_probs = []
        for center in self.centers:
            dist = -0.5 * ((u - center) ** 2).sum(dim=1) / self.variance
            log_probs.append(dist)
        return torch.logsumexp(torch.stack(log_probs, dim=1), dim=1)
    
    def __call__(self, batch_size=1):
        """
        """
        return self.sample(batch_size)

class PosteriorDistribution(Distribution):
    """
    """
    def __init__(self, 
                 posterior_samples: torch.Tensor, 
                 weights: torch.Tensor = None):
        """
        """
        self.posterior_samples = torch.tensor(posterior_samples)
        self.num_samples = len(self.posterior_samples)
        self._weights = torch.ones(self.num_samples) / self.num_samples
        if weights is not None: 
            if weights.size(0) != self.num_samples:
                raise ValueError("must be same number of weights as samples")
            self._weights = weights
            self._weights /= sum(self._weights)
                
    def sample(self, batch_size):
        """
        """
        # indices = torch.randint(0, self.posterior_samples.size(0), (batch_size,))
        indices = torch.multinomial(self.weights, batch_size, replacement=True)
        return self.posterior_samples[indices]
    
    
    def add_samples(self, new_samples: torch.Tensor, new_weights: torch.Tensor = None):
        """
        NOTE: if adding new samples, unless new_weights are provided that are the size of the concatenated
        array, the weights will be uniform
        """
        self.posterior_samples = torch.cat([self.posterior_samples, new_samples], dim=0)
        self.num_samples = len(self.posterior_samples)
        if new_weights is not None:
            self._update_weights(new_weights)
        else:
            self._weights = torch.ones(self.num_samples) / self.num_samples


    def _update_weights(self, new_weights: torch.Tensor):
        """
        """
        new_weights /= new_weights.sum()
        self._weights = torch.cat([self._weights * (len(self._weights) / self.num_samples), 
                                   new_weights * (len(new_weights) / self.num_samples)], dim=0)

    def reweight(self, new_weights: torch.Tensor):
        """
        """
        if new_weights.size(0) != self.posterior_samples.size(0):
            raise ValueError("New weights must have the same size as the number of posterior samples.")
        self._weights = new_weights
        self._weights /= self._weights.sum()

    def log_prob(self, u: torch.Tensor):
        """
        Compute the log-probability for the posterior samples.

        Note: This is optional and depends on how the posterior distribution is modeled.
        If you're not modeling explicit likelihoods for the posterior, this can raise an error.
        """
        raise NotImplementedError("Log probability is not defined for this posterior distribution.")
    
    def __call__(self, batch_size=1):
        """
        """
        return self.sample(batch_size)

    @property
    def weights(self):
        """
        """
        return self._weights


class CustomDistribution(Distribution):
    def __init__(self, sample_func, log_prob_func=None):
        """
        Custom distribution allowing users to define their own sampling and log-prob functions.

        Parameters:
        - sample_func: callable, must return samples with shape `(batch_size, input_dim)`
        - log_prob_func: callable, must return log-probabilities given the samples (optional)
        """
        if not callable(sample_func):
            raise ValueError("`sample_func` must be a callable")
        self.sample_func = sample_func
        self.log_prob_func = log_prob_func

    def log_prob(self, u):
        if self.log_prob_func is None:
            raise NotImplementedError("Log probability function is not provided for this custom distribution")
        return self.log_prob_func(u)

    def sample(self, batch_size):
        return self.sample_func(batch_size)
    
    def __call__(self, batch_size=1):
        """
        """
        return self.sample(batch_size)


class GeneralDistribution:
    """
    Methods
    ----------------
    `__init__`:

    `sample`:

    `log_prob`:
    """
    _accepted_types = ["gaussian", "normal", "uniform", "flat", "custom", "posterior"]
    _distribution_map = {
        "gaussian": GaussianDistribution,
        "uniform": UniformDistribution,
        "posterior": PosteriorDistribution
    }
    def __init__(self, 
                 input_dim:int, 
                 distribution_type:str="gaussian", 
                 prior_t=None, 
                 sample_func=None, 
                 log_prob_func=None, 
                 posterior_samples=None,
                 sample_args=[],
                 **kwargs):
        """
        Parameters
        ----------------
        sample_func : callable
            Should take parameters `*shape` and return a distribution of that shape

        prob_func : callable
            Should take `u` as a parameter and evaluate its log-likelihood
        """
        self.distribution_type = distribution_type.lower()
        if self.distribution_type not in self._accepted_types:
            raise ValueError(f"Unknown distribution type: {self.distribution_type}")
        if self.distribution_type in ["gaussian", "normal"]:
            self.sample_func = torch.randn
            self.log_prob_func = gaussian_log_prob
        elif self.distribution_type in ["uniform", "flat"]:
            self.sample_func = torch.rand
            self.log_prob_func = uniform_log_prob
        elif self.distribution_type == "custom":
            if sample_func is None:
                raise ValueError("sample_func must be provided for custom distribution.")    
            self.sample_func = sample_func
            self.log_prob_func = log_prob_func
        elif self.distribution_type == "posterior":
            if posterior_samples is None:
                raise ValueError("Posterior samples must be provided for a posterior distribution.")
            self.posterior_samples = posterior_samples
            self.sample_func = self._sample_posterior
            self.log_prob_func = None  # This may vary depending on how you want to handle log-probabilities for poster
        self.input_dim = input_dim
        self.prior_t = prior_t
        # TODO change this to kwargs
        self.sample_args = sample_args
        self.kwargs = kwargs
        # self.distribution = Distribution.create(distribution_type, dim=input_dim, **kwargs)

    def sample(self, batch_size:int):
        """
        Parameters
        ----------------
        batch_size : int

        """
        return self.sample_func(batch_size, self.input_dim, *self.sample_args, **self.kwargs)
    
    def sample_model(self, batch_size):
        choice = torch.random()

    def log_prob(self, u:torch.Tensor, **kwargs) -> torch.Tensor:
        """
        For a normal distribution, the log-probability is given by
        `log p_U(u)=-\frac{1}{2}u^Tu-\frac{d}{2}log(2 pi)`

        For a uniform distribution over $[0, 1]^d$, the log-probability is constant
        `log p_U(u)=0`

        Parameters
        ----------------
        z : torch.Tensor
        """
        if self.log_prob_func is None:
            raise ValueError("Cannot call `log_prob` if no log prob func provided at class initialization")
        return self.log_prob_func(u, **kwargs)

    def log_prob2(self, u: torch.Tensor):
        return LogProbFactory.get_log_prob(self.distribution_type)(u)
    
    def add_posterior_samples(self, new_samples:torch.Tensor):
        """
        Adds new posterior samples to the distribution.

        Parameters
        ----------------
        new_samples : torch.Tensor
            New posterior samples to be added.

        """
        if self.distribution_type != "posterior":
            raise ValueError("Can only add posterior samples to a posterior distribution.")
        self.posterior_samples = torch.cat([self.posterior_samples, new_samples], dim=0)

    def _sample_posterior(self, batch_size:int):
        """
        Samples from the stored posterior samples.

        Parameters
        ----------------
        batch_size : int
            Number of samples to draw.
        input_dim : int
            Dimensionality of the samples.

        Returns
        ----------------
        torch.Tensor
            Randomly sampled posterior samples.
        """
        indices = torch.randint(0, self.posterior_samples.size(0), (batch_size,))
        return self.posterior_samples[indices]

    def __call__(self, batch_size: int):
        """
        """
        return self.sample(batch_size)
    

