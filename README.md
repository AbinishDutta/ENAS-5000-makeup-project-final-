# Project B: 1D Drug Delivery Simulation

## 1. Problem Description
I modelled the transport of a drug through a tissue slab of thickness $L$. The drug enters the tissue at $x=0$ (e.g., via a skin patch) and then diffuses towards $x=L$. As the drug molecules spread, they are simultaneously metabolized (eliminated) by biological processes at a rate proportional to their concentration.

The physical system is governed by the 1D **Reaction-Diffusion Equation**:

$$
\frac{\partial c}{\partial t} = D \frac{\partial^2 c}{\partial x^2} - kc
$$

Where:
*   $c(x,t)$ is the drug concentration.
*   $D$ is the diffusion coefficient.
*   $k$ is the decay (metabolic) rate constant.

## 2. Mathematical Derivation

### Boundary and Initial Conditions
To solve the PDE, I applied the following physical constraints:
1.  **Source ($x=0$):** Constant concentration maintained by the patch.
    $$ c(0, t) = c_0 $$
2.  **Sink ($x=L$):** The drug is rapidly cleared by blood flow upon reaching the other side, maintaining zero concentration.
    $$ c(L, t) = 0 $$
3.  **Initial State ($t=0$):** The tissue contains no drug initially.
    $$ c(x, 0) = 0 $$

### Analytical Approach
Because the boundary conditions are non-homogeneous ($c_0 \neq 0$), I decomposed the solution into a steady-state component $c_\infty(x)$ and a transient component $c_T(x,t)$:
$$ c(x,t) = c_\infty(x) + c_T(x,t) $$

1.  **Steady State:** Setting $\frac{\partial c}{\partial t} = 0$, I solved $0 = D c'' - k c$. The solution satisfying BCs is:
    $$ c_\infty(x) = c_0 \frac{\sinh(\sqrt{k/D}(L-x))}{\sinh(\sqrt{k/D}L)} $$
2.  **Transient State:** I solved for the deviation from steady state using Separation of Variables and Fourier Series.

---

## 3. Numerical Method: Explicit Euler (FTCS)

To approximate the solution numerically, I used the **Finite Difference Method** with a Forward-Time Central-Space (FTCS) scheme.

### Discretization
I defined a grid with spatial step $\Delta x$ and time step $\Delta t$, where $c_i^n$ represents the concentration at node $i$ and time step $n$.

1.  **Time Derivative (Forward Difference):**
    $$ \frac{\partial c}{\partial t} \approx \frac{c_i^{n+1} - c_i^n}{\Delta t} $$

2.  **Spatial Derivative (Central Difference):**
    $$ \frac{\partial^2 c}{\partial x^2} \approx \frac{c_{i+1}^n - 2c_i^n + c_{i-1}^n}{\Delta x^2} $$

### The Update Equation
Substituting these approximations into the governing PDE:

$$ \frac{c_i^{n+1} - c_i^n}{\Delta t} = D \left( \frac{c_{i+1}^n - 2c_i^n + c_{i-1}^n}{\Delta x^2} \right) - k c_i^n $$

Rearranging to solve for the concentration at the next time step ($c_i^{n+1}$):

$$ c_i^{n+1} = c_i^n + \frac{D \Delta t}{\Delta x^2} (c_{i+1}^n - 2c_i^n + c_{i-1}^n) - k \Delta t c_i^n $$

Letting $\alpha = \frac{D \Delta t}{\Delta x^2}$, the vectorized update formula is:

$$ c_i^{n+1} = (1 - 2\alpha - k\Delta t)c_i^n + \alpha(c_{i+1}^n + c_{i-1}^n) $$

### Stability Condition
The Explicit Euler method is **conditionally stable**. For the solution not to oscillate and diverge, the coefficient of $c_i^n$ must remain non-negative. This requires:

$$ \Delta t \le \frac{\Delta x^2}{2D} $$

---

## 4. Parameter Justification

### Physical Parameters
I selected parameters representative of small drug molecules diffusing through biological tissue.

| Parameter | Value | Justification |
| :--- | :--- | :--- |
| **Length ($L$)** | $5 \text{ mm}$ ($0.005 \text{ m}$) | Typical depth for transdermal drug delivery or tissue slab models. |
| **Diffusion ($D$)** | $1 \times 10^{-10} \text{ m}^2/\text{s}$ | Realistic for small molecules in water/tissue (approx $1/10^{th}$ of pure water diffusion). |
| **Decay ($k$)** | $2 \times 10^{-4} \text{ s}^{-1}$ | Corresponds to a half-life of $\approx 1$ hour ($\ln(2)/k \approx 3465$s), a typical metabolic clearance rate. |
| **Concentration ($c_0$)** | $1.0$ (normalized) | Represents $100\%$ saturation at the source patch. |

### Numerical Parameters
To ensure accuracy and stability with the FTCS scheme:

1.  **Spatial Grid ($N_x$):**
    *   I chose $N_x = 50$, giving $\Delta x = 0.1 \text{ mm}$. This provides sufficient resolution to capture the concentration gradient.
2.  **Time Step ($N_t$):**
    *   **Stability Check:**
        $$ \Delta t_{max} = \frac{(10^{-4})^2}{2 \cdot 10^{-10}} = \frac{10^{-8}}{2 \cdot 10^{-10}} = 50 \text{ seconds} $$
    *   I simulated for $T_{final} = 7200$ seconds (2 hours).
    *   I chose $N_t = 1000$, resulting in $\Delta t = 7.2 \text{ s}$.
    *   Since $7.2 \text{ s} < 50 \text{ s}$, our simulation is **stable**.

---

## 5. Python Implementation

```python
import numpy as np

def solve_drug_delivery_ftcs(L, D, k, c0, T_final, Nx, Nt):
    """
    Solves 1D Reaction-Diffusion using Explicit Euler (FTCS).
    """
    # 1. Grid setup
    dx = L / Nx
    dt = T_final / Nt
    x = np.linspace(0, L, Nx + 1)
    
    # 2. Stability check
    alpha = (D * dt) / (dx**2)
    if alpha > 0.5:
        raise ValueError(f"Instability detected! alpha={alpha:.2f}. "
                         f"dt must be < {dx**2 / (2*D):.2f}s")

    # 3. Initialization
    c = np.zeros(Nx + 1)      # Current time step (Initial Condition c=0)
    c_new = np.zeros(Nx + 1)  # Next time step
    
    # 4. Time Stepping
    for n in range(Nt):
        # Update interior nodes
        # c[1:-1] are nodes 1 to N-1
        diffusion_term = alpha * (c[2:] - 2*c[1:-1] + c[:-2])
        reaction_term = k * dt * c[1:-1]
        
        c_new[1:-1] = c[1:-1] + diffusion_term - reaction_term
        
        # Apply Boundary Conditions
        c_new[0] = c0   # Source
        c_new[-1] = 0   # Sink
        
        # Update for next iteration
        c[:] = c_new[:]
        
    return x, c

```
