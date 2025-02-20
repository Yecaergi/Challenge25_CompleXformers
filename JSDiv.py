import numpy as np
from Shannon import Shannon


def JSDiv(P, Q):
    """
    Jensen-Shannon divergence of two probability distributions.
    P and Q are automatically normalized to have the sum to one on each column.
    P: n x n bins
    Q: 1 x 1 uniform distribution Pe = 1/N
    jsd: 1 x n
    """
    # Normalize P so that the sum of each column is 1
    P = P / np.sum(P, axis=0)
    
    # Calculate the JSD according to equations (7) and (8) from Lamberti, 2004
    # and (16) from Martin 2006
    M = 0.5 * (P + Q)
    jsd = Shannon(M) - 0.5 * Shannon(P) - 0.5 * np.log(1/Q)
    
    return jsd



# Ejemplo de uso
#P = np.array([[0.1, 0.4, 0.5, 0.3, 0.3, 0.4]])
#Q = 1/3
#jsd = JSDiv(P, Q)


