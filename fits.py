import numpy as np
from scipy.special import erf

def box_gaussian(x, A, x0, w, sigma, c):
    s = np.sqrt(2) * sigma
    return 0.5 * A * (erf((x - x0 + w/2) / s) - erf((x - x0 - w/2) / s)) + c

def edge(x, A, x0, sigma, c):
    return 0.5 * A * (1 + erf((x - x0) / (np.sqrt(2) * sigma))) + c
