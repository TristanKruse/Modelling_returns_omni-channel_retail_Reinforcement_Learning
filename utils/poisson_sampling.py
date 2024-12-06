import numpy as np
from scipy.stats import poisson


def adjusted_poisson_pmf(mu, max_value):
    pmf = poisson.pmf(np.arange(max_value + 1), mu)
    pmf /= pmf.sum()
    return pmf


def sample_from_adjusted_poisson(pmf, size=1):
    return np.random.choice(len(pmf), size=size, p=pmf)
