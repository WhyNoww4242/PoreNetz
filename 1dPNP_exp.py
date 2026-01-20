import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp

# Parameters (Dimensionless or Arbitrary Units)
L = 1.0          # Half-distance between electrodes
lambda_D = 0.05  # Debye screening length (small epsilon)
kappa = 1/lambda_D
V = 0.5          # Applied voltage (keep small for linear approx)
lambda_S = 0.02  # Stern layer thickness
x = np.linspace(-L, L, 500)

# 1. Steady-state Potential Profile (from Page 7, Eq. 25)
# Note: This accounts for screening and the Stern layer
denominator = np.sinh(kappa * L) + kappa * lambda_S * np.cosh(kappa * L)
phi_ss = V * np.sinh(kappa * x) / denominator

# 2. Charge Density Profile (from Page 7, Eq. 22)
# Using Poisson's equation: rho = -epsilon * d^2(phi)/dx^2
# In the linear limit, rho is proportional to sinh(kappa * x)
rho_ss = -(kappa**2) * phi_ss # Simplified dimensionless form

def calcPhi(rho_np, points, bc_left, bc_right, Debye):
    
    rho = jnp.array(rho_np)
    n = len(rho)

    matrix = (
    -2.0 * jnp.eye(n)
    + jnp.eye(n, k=1)
    + jnp.eye(n, k=-1)
    )



    return phi_vec, dphi_vec

print(phi_ss[0],phi_ss[-1],'phi')
# Plotting
fig, ax1 = plt.subplots(figsize=(8, 5))

ax1.set_xlabel('Position (x/L)')
ax1.set_ylabel('Potential ($\Phi$)', color='tab:blue')
ax1.plot(x, phi_ss, label='Potential Profile', color='tab:blue', linewidth=2)
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax1.grid(True, linestyle='--', alpha=0.7)

ax2 = ax1.twinx()
ax2.set_ylabel('Charge Density ($\\rho_e$)', color='tab:red')
ax2.plot(x, rho_ss, label='Charge Density', color='tab:red', linestyle=':')
ax2.tick_params(axis='y', labelcolor='tab:red')

plt.title('Potential and Charge Density Profiles (Linearized Solution)')
fig.tight_layout()
plt.show()