import numpy as np
import matplotlib.pyplot as plt
from numerical_solution_EE import ftcs_solver
from analytical_solution import analytical_concentration
from params import params




# Comparison Plot

Nx_sim = 50
Nt_sim = 200
x_num, c_num = ftcs_solver(params, Nx_sim, Nt_sim)
c_ana = analytical_concentration(x_num, params['T_final'], params)

plt.figure(figsize=(10, 5))
plt.plot(x_num*1000, c_ana, 'k-', linewidth=2, label='Analytical')
plt.plot(x_num*1000, c_num, 'r--', marker='o', markevery=2, label='Numerical (CN)')
plt.xlabel('Distance (mm)')
plt.ylabel('Concentration ($c/c_0$)')
plt.title(f'Drug Concentration at t = {params["T_final"]/3600:.1f} hours')
plt.legend()
plt.grid(True)
plt.show()



# Convergence Study

print("Running Convergence Study (Log-Log Plot)...")
grids = [20, 40, 80, 160, 320]
errors = []
dx_values = []

# To see spatial convergence clearly, we need small dt.
# We scale Nt with Nx^2 or just use a very high fixed Nt to isolate spatial error,
# or scale Nt proportional to Nx (constant Courant number).
# Let's scale Nt linearly with Nx to keep it efficient but stable.
for N in grids:
    # Scaling Nt with N^2 to maintain stability (constant alpha) to fix the nan thing

    Nt_conv = 2 * (N**2)  
    
    # Run Solver
    x_grid, c_num_conv = ftcs_solver(params, N, Nt_conv)
    c_ana_conv = analytical_concentration(x_grid, params['T_final'], params)
    
    # RMS Error
    error = np.sqrt(np.mean((c_num_conv - c_ana_conv)**2))
    
    # If error is infinite/nan, don't append (or handle it)
    if not np.isfinite(error):
        print(f"Grid N={N} was unstable!")
        break
        
    errors.append(error)
    dx_values.append(params['L'] / N)

# Slope Calculation
log_dx = np.log(dx_values)
log_err = np.log(errors)
slope, intercept = np.polyfit(log_dx, log_err, 1)

print(f"Calculated Convergence Slope: {slope:.3f}")

plt.figure(figsize=(8, 6))
plt.loglog(dx_values, errors, 'bo-', label=f'Numerical Error (Slope={slope:.2f})')

# Reference Line (Slope 2)
dx_arr = np.array(dx_values)
ref_line = np.exp(intercept) * (dx_arr / dx_arr[0])**2 * (errors[0]/errors[-1])*0.1 # visual adjustment
# Just simpler reference:
ref_line = errors[0] * (dx_arr / dx_arr[0])**2
plt.loglog(dx_arr, ref_line, 'k--', label='Reference Slope = 2')

plt.xlabel('Grid spacing $\Delta x$ (log)')
plt.ylabel('RMS Error (log)')
plt.title('Convergence Study:')
plt.legend()
plt.grid(True, which="both", ls="-")
plt.show()