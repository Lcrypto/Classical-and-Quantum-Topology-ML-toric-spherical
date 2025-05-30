import numpy as np
from functools import partial

def f(β, J_edge_list, c, phi):
     """Function to find the root of: c * phi * mean(tanh²(β * J)) - 1 = 0."""
    return c * phi * np.mean(np.tanh(β * J_edge_list) ** 2) - 1

def find_beta_sg_dichotomy(J_edge_list, c, phi, tol=1e-6, max_iter=100):
    """
    Finds β_SG using the bisection method.

    Parameters:
        J_edge_list (np.ndarray): Array of interaction strengths.
        c (float): Average vertex degree.
        phi (float): Parameter.
        tol (float): Tolerance threshold.
        max_iter (int): Maximum number of iterations.

    Returns:
        float: Estimated β_SG value.
    """
    f_partial = partial(f, J_edge_list=J_edge_list, c=c, phi=phi)
    
    # Initial values β
    beta_low = 0.0  # f(0) = c * phi * 0 - 1 = -1 < 0
    beta_high = 1.0  # Begin from 1 and increase, until f > 0
    
    # Increase beta_high, until f(beta_high) < 0
    while f_partial(beta_high) < 0:
        beta_high *= 2
    
    # Check, that f(beta_low) < 0 и f(beta_high) > 0
    assert f_partial(beta_low) < 0, "Starting lower bound must f < 0"
    assert f_partial(beta_high) > 0, "Starting upper bound must f > 0"
    
    # Dichotomy (Bisection) method
    for _ in range(max_iter):
        beta_mid = (beta_low + beta_high) / 2
        f_mid = f_partial(beta_mid)
        
        if abs(f_mid) < tol:
            return beta_mid
        
        if f_mid < 0:
            beta_low = beta_mid
        else:
            beta_high = beta_mid
    
    # If not converge after max_iter iterations, return to middle of last calculation
    return (beta_low + beta_high) / 2