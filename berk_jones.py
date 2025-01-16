import crossprob
from bisect import bisect_left
from dataclasses import asdict, dataclass
import numpy.typing as npt
from typing import Callable
import scipy
import numpy as np

@dataclass(frozen=True)
class OrderStatsBound:
    """Class for representing an CDF lower bounds based on order statistics."""
    q: npt.ArrayLike
    delta: float # probability of failure
        
    @property
    def n(self) -> int:
        return self.q.shape[0]
        
    def __post_init__(self):
        assert self.q.ndim == 1, "q must be 1d"
        assert (0 <= np.min(self.q)) and (np.max(self.q) <= 1.0), "elements of q must be in [0, 1]"
        assert np.all((self.q[1:] - self.q[:-1]) >= 0.0), "elements of q must be monotone nondecreasing"
        assert 0 <= self.delta <= 1, "delta must be in [0, 1]"


def bisect_proposal(proposal : Callable, delta : float):
    def f(c):
        return crossprob.ecdf1_new_b(proposal(c)) - (1 - delta)
    
    c_opt = scipy.optimize.bisect(f, 0.0, 1.0)
    
    return OrderStatsBound(proposal(c_opt), delta)

def berk_jones(n : int, delta : float) -> OrderStatsBound:
    def proposal(c : float):
        i = np.arange(1, n + 1)
        return scipy.special.betaincinv(i, n - i + 1, c)
    
    return bisect_proposal(proposal, delta).q

def berk_jones_trunc(n : int, delta : float, k : int) -> OrderStatsBound:
    def proposal(c : float):
        i = np.arange(1, n + 1)
        b = scipy.special.betaincinv(i, n - i + 1, c)
        b[:k] = 0.0
        return b
    
    return bisect_proposal(proposal, delta)