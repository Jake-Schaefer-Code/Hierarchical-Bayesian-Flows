"""

"""
import torch

__all__ = ["spherical_harmonics", "TorchKDE"]

# TODO add more bandwidth methods
def scott(n, d):
    return n**(-1/(d+4))

def silverman(n, d):
    return (n * (d + 2) / 4.) ** (-1 / (d + 4))

_bandwidth_methods = {
    "scott": scott,
    "silverman": silverman
}

class TorchKDE:
    """
    Methods
    ----------------
    `__init__`: 
    
    `evaluate`:
    """
    _allowed_bw_methods = ["scott", "silverman"]
    def __init__(self, data:torch.Tensor, weights=None, bandwidth=1.0):
        """
        Parameters
        ----------------
        """
        self.data = data
        self.weights = weights if weights is not None else torch.ones(data.shape[1])
        self.weights = self.weights / torch.sum(self.weights)
        self.d, self.n = data.size()

        if isinstance(bandwidth, str):
            if bandwidth not in self._allowed_bw_methods:
                raise ValueError(f"No such implemented bandwidth method: {bandwidth}")
            self.bandwidth = _bandwidth_methods[bandwidth](self.n, self.d)
        else:
            self.bandwidth = bandwidth

    def evaluate(self, points: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------------
        """
        points = points.T[:, None, :]
        diff = points - self.data.T[None, :, :]
        exponents = -0.5 * torch.sum((diff / self.bandwidth) ** 2, dim=-1)  # Sum over dimensions
        densities = torch.exp(exponents) @ self.weights  # Weighted sum
        normalization = (2 * torch.pi * self.bandwidth ** 2) ** (self.d / 2)
        return densities / normalization

    def __call__(self, points):
        return self.evaluate(points)

def factorial(n):
    r"""
    For positive integers, the gamma function satisfies:
    $\Gamma(n+1) = n!$
    Torch lgamma suited for large numbers
    The gamma function generalizes the factorial function to non-integer values, allowing for more flexibility in some mathematical contexts.
    also returns floating point value
    """
    return torch.exp(torch.lgamma(n + 1.0))

def normalization_constant(l, m):
    """
    N^l_m
    """
    return torch.sqrt((2 * l + 1) / (4 * torch.pi) * factorial(l - m) / factorial(l + m))

def associated_legendre_polynomial(l, m, x):
    """
    Computes the associated Legendre polynomial P_l^m(x) using recurrence relations
    """
    abs_m = torch.abs(m)
    sign = torch.where(m < 0, (-1)**abs_m*factorial(l-abs_m)/factorial(l+abs_m), torch.ones_like(m))
    # P_m^m(x) = (-1)^m (2m-1)!! (1-x^2)^(m/2)
    pmm = torch.ones_like(x)
    if abs_m.max() > 0:
        somx2 = torch.sqrt(1 - x**2)
        for i in range(abs_m.max()):
            pmm = torch.where(abs_m > i, pmm * (-1) * (2 * i + 1) * somx2, pmm)
    # P_{m+1}^m(x) = x(2m+1)P_m^m(x)
    pmmp1 = torch.where(l != abs_m, x * (2 * abs_m + 1) * pmm, pmm)
    # Recurrence relation for l > m+1
    pll = torch.zeros_like(x)
    
    pout = pmmp1
    pout = torch.where(abs_m == 0, torch.where(l == 0, pmm, pmmp1), pmmp1)
    # pmmp1 = torch.where(abs_m == 0, torch.where(l == 0, pmm, pmmp1), pmmp1)
    for ll in range(abs_m.max() + 2, l.max() + 1):
        # pll = torch.where(ll == l, ((2 * ll - 1) * x * pmmp1 - (ll + abs_m - 1) * pmm) / (ll - abs_m), pll)
        pll = ((2 * ll - 1) * x * pmmp1 - (ll + abs_m - 1) * pmm) / (ll - abs_m)
        pmm = pmmp1
        pmmp1 = pll
        pout = torch.where(l==ll, pll, pout)
        
    if abs_m.eq(0).any():
        for ll in range(2, l.max().item() + 1):
            pll = ((2 * ll - 1) * x * pout - (ll - 1) * torch.ones_like(x)) / ll
            pout = torch.where((abs_m == 0) & (l == ll), pll, pout)
    # print(sign*pout)
    # print(lpmv(m.cpu().numpy(), l.cpu().numpy(), x.cpu().numpy()))
    return sign*pout

def spherical_harmonics(l, m, theta, phi):    
    """
    """
    N_lm = normalization_constant(l, m)
    # print(torch.sin(theta), theta)
    P_lm = associated_legendre_polynomial(l, m, torch.cos(theta))
    Y_lm = N_lm * P_lm * torch.exp(1j * m * phi)
    Y_lm = torch.where(m<0, Y_lm*torch.pow(-1, torch.abs(m)), Y_lm)
    return Y_lm
