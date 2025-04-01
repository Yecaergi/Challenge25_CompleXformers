import numpy as np

def check_finite(arr):
    if not np.all(np.isfinite(arr)):
        raise ValueError('La entrada contiene valores no finitos (NaN o Inf).')

def Shannon(P):
    if np.sum(~np.isfinite(P)):
        raise ValueError('The input contains non-finite values!')
    check_finite(P)
    # Shannon entropy
    S = -np.sum(P * np.log(P), axis=0)
    # Ensure that 0 * log(0) = 0
    S[np.isnan(S)] = 0
    
    return S



