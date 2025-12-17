
import numpy as np


def analytical_concentration(x_arr, t, p, n_terms=50):
    """
    Computes the exact solution c(x,t) = u(x) + v(x,t).
    
    Args:
        x_arr (np.array): Spatial grid points.
        t (float): Time to evaluate.
        p (dict): Parameters dictionary.
        n_terms (int): Number of terms in the Fourier series summation.
    """
    L, D, k, c0 = p['L'], p['D'], p['k'], p['c0']
    
    # 1. Steady State Solution u(x)
    gamma = np.sqrt(k / D)
    
    # Using hyperbolic identity for stability: sinh(A-B)
    # u(x) = c0 * sinh(gamma*(L-x)) / sinh(gamma*L)
    c_infinity = c0 * np.sinh(gamma * (L - x_arr)) / np.sinh(gamma * L)
    
    # 2. Transient Solution v(x,t)
    c_transient = np.zeros_like(x_arr)
    
    for n in range(1, n_terms + 1):
        lambda_n = (n * np.pi / L)**2
        
        # Fourier coefficient b_n
        numerator = -2 * c0 * n * np.pi
        denominator = (L**2) * ((k/D) + lambda_n)
        b_n = numerator / denominator
        
        # Time decay factor
        decay = np.exp(-(D * lambda_n + k) * t)
        
        # Spatial shape
        spatial = np.sin(n * np.pi * x_arr / L)
        
        c_transient += b_n * spatial * decay
        
    return c_infinity + c_transient