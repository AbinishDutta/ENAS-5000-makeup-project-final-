# Parameters for the diffusion-decay model in biological tissue


params = {
    'L': 0.005,          # Length of tissue [m] (5 mm) (2 mm is more typical but lets use 5 mm for better visualization)
    'D': 1e-10,          # Diffusion coefficient [m^2/s]
    'k': 2e-4,           # Decay rate constant [1/s]
    'c0': 1.0,           # Source concentration [arbitrary units]
    'T_final': 3600*2,   # Simulation time [s] (2 hours)
}