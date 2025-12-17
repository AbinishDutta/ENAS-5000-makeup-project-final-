import numpy as np

def ftcs_solver(p, Nx, Nt):
    """
    Solves using Explicit Euler (FTCS).
    """
    L, D, k, c0, T_final = p['L'], p['D'], p['k'], p['c0'], p['T_final']
    
    dx = L / Nx
    dt = T_final / Nt
    x = np.linspace(0, L, Nx + 1)
    
    # Check stability
    alpha = (D * dt) / (dx**2)
    stability_limit = 0.5
    if alpha > stability_limit:
        print(f"WARNING: Unstable! alpha ({alpha:.2f}) > 0.5.")
        print(f"Reduce dt or increase dx.")
    
    # Initialize grid
    # c[i] stores current time step
    # c_new[i] stores next time step
    c = np.zeros(Nx + 1)
    c_new = np.zeros(Nx + 1)
    
    # Initial Condition: c(x,0) = 0
    c[:] = 0.0
    
    # Boundary Condition at t=0
    c[0] = c0
    
    # Time Stepping
    for n in range(Nt):
        # Explicit update for interior nodes (1 to Nx-1)
        # Using vectorized numpy operations for speed
        # c[0:-2] is c_{i-1}
        # c[1:-1] is c_{i}
        # c[2:]   is c_{i+1}
        
        c_new[1:-1] = c[1:-1] + alpha * (c[2:] - 2*c[1:-1] + c[:-2]) - k * dt * c[1:-1]
        
        # Apply Boundary Conditions
        c_new[0] = c0   # Source
        c_new[-1] = 0   # Sink
        
        # Update array for next step
        c[:] = c_new[:]
        
    return x, c