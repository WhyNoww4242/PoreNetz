import jax.numpy as jnp
from jax import jit, lax
import matplotlib.pyplot as plt
from modules.graphStyle import set_plot_style, add_subplot_labels
import diffrax

def weights(points, D_i, D_eff):
    dist = jnp.diff(points)  # finds all point distances
    # creates a_vec(diffusion) and b_vec (electromigration) discretions 
    a_vec = jnp.ones_like(dist)
    b_vec = jnp.ones_like(dist)
    a_vec = a_vec.at[:].set(D_i/dist)
    b_vec = b_vec.at[:].set(D_eff/(2*dist))
    return a_vec, b_vec

def phi(c_array, ci_len, n_points, d_eff_array):
    """
    Calculate electric potential gradient from charge density
    c_array: flattened concentration array (ci_len * n_points,)
    """
    # Reshape to (ci_len, n_points)
    c_reshaped = jnp.reshape(c_array, (ci_len, n_points))
    
    # Calculate total charge density at each point
    # d_eff_array: (ci_len,), c_reshaped: (ci_len, n_points)
    rho = jnp.dot(d_eff_array, c_reshaped)  # (n_points,)
    
    return rho

def ODE_dis(t, cp, args):
    """
    ODE for multiple ionic species with electromigration using finite volume method
    cp is organized as: [species1_points, species2_points, species3_points, ...]
    """
    j_left, j_right, dist, a_vecs, b_vecs, d_eff, ci_len = args
    
    n_points = len(cp) // ci_len
    
    # Get potential (charge density) at each spatial point
    rho = phi(cp, ci_len, n_points, d_eff)
    
    # Initialize time derivative
    dcp_dt = jnp.zeros_like(cp)
    
    # Process each species
    for i in range(ci_len):
        # Extract concentration for this species
        start_idx = i * n_points
        end_idx = (i + 1) * n_points
        c_species = cp[start_idx:end_idx]
        
        # Get the weights for this species
        a_vec = a_vecs[i]
        b_vec = b_vecs[i]
        
        # Get boundary fluxes for this species
        j_left_i = j_left[i]
        j_right_i = j_right[i]
        
        # Shifted concentrations (neighbors)
        cw = jnp.roll(c_species, 1)   # west/left neighbor
        ce = jnp.roll(c_species, -1)  # east/right neighbor
        
        # Finite volume method
        dc_dt = jnp.zeros(n_points)
        
        # Left flux contributions (flux from west into cell)
        dc_dt = dc_dt.at[:-1].add((b_vec * rho[:-1] - a_vec) * c_species[:-1] / dist)
        dc_dt = dc_dt.at[1:].add((b_vec * rho[1:] - a_vec) * c_species[1:] / dist)
        
        # Right flux contributions (flux from east into cell)  
        dc_dt = dc_dt.at[:-1].add((b_vec * rho[:-1] + a_vec) * cw[:-1] / dist)
        dc_dt = dc_dt.at[1:].add((b_vec * rho[1:] + a_vec) * ce[1:] / dist)
        
        # Boundary conditions
        dc_dt = dc_dt.at[0].add(j_left_i / dist[0])
        dc_dt = dc_dt.at[-1].add(j_right_i / dist[-1])
        
        # Store in full array
        dcp_dt = dcp_dt.at[start_idx:end_idx].set(dc_dt)
    
    return dcp_dt

def main():
    D_i = jnp.array([1.0, 1.0, 1.0])
    D_eff = jnp.array([1.0, -2.0, 1.0])
    points = jnp.linspace(0, 1, 10)  # Changed back to 10 points for faster execution
    
    # Get spatial discretization info
    dist = jnp.diff(points)
    ci_len = len(D_i)
    n_points = len(points)
    
    # Calculate weights for each species and stack them
    a_vecs = []
    b_vecs = []
    for i in range(ci_len):
        a, b = weights(points, D_i[i], D_eff[i])
        a_vecs.append(a)
        b_vecs.append(b)
    
    # Stack into arrays: (ci_len, n_points-1)
    a_vecs = jnp.array(a_vecs)
    b_vecs = jnp.array(b_vecs)
    
    # Define boundary fluxes for each species
    # Only first species (index 0) has flux of 1.0 at both boundaries
    # IMPORTANT: These must be JAX arrays, not Python lists
    j_left = jnp.array([1.0, 0.0, 0.0])   # [species1, species2, species3]
    j_right = jnp.array([1.0, 0.0, 0.0])  # [species1, species2, species3]
    
    print(f"Boundary fluxes (left): {j_left}")
    print(f"Boundary fluxes (right): {j_right}")
    
    # Pack args tuple
    args = (j_left, j_right, dist, a_vecs, b_vecs, D_eff, ci_len)
    
    term = diffrax.ODETerm(ODE_dis)
    
    # Initial condition: all species start with concentration 1.0
    y0 = jnp.ones(ci_len * n_points)
    
    # Temporal discretisation
    t0 = 0.0
    t_final = 0.1
    δt = 0.0001
    saveat = diffrax.SaveAt(ts=jnp.linspace(t0, t_final, 50))
    
    # Tolerances
    rtol = 1e-6
    atol = 1e-6
    stepsize_controller = diffrax.PIDController(
        pcoeff=0.3, icoeff=0.4, rtol=rtol, atol=atol, dtmax=0.001
    )
    
    solver = diffrax.Tsit5()
    
    # Solve
    print("Starting simulation...")
    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0,
        t_final,
        δt,
        y0,
        args=args,
        saveat=saveat,
        stepsize_controller=stepsize_controller,
        max_steps=None,
    )
    
    print("Solution shape:", sol.ys.shape)
    print("Times:", sol.ts.shape)
    
    # Reshape solution for analysis
    # sol.ys shape is (n_times, ci_len * n_points)
    n_times = len(sol.ts)
    sol_reshaped = sol.ys.reshape(n_times, ci_len, n_points)
    
    print("\nInitial concentrations:")
    for i in range(ci_len):
        print(f"Species {i+1}: {sol_reshaped[0, i, :]}")
    
    print("\nFinal concentrations:")
    for i in range(ci_len):
        print(f"Species {i+1}: {sol_reshaped[-1, i, :]}")
    
    # Create plots at three different times
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Select three time indices: early, middle, late
    time_indices = [0, n_times // 2, n_times - 1]
    time_labels = ['Early', 'Middle', 'Final']
    
    for ax_idx, (t_idx, t_label) in enumerate(zip(time_indices, time_labels)):
        ax = axes[ax_idx]
        
        # Plot each species
        for i in range(ci_len):
            ax.plot(points, sol_reshaped[t_idx, i, :], 
                   label=f'Species {i+1} (z={D_eff[i]:.0f})', 
                   marker='o', markersize=4, linewidth=2)
        
        ax.set_xlabel('Position')
        ax.set_ylabel('Concentration')
        ax.set_title(f'{t_label} (t = {sol.ts[t_idx]:.3f})')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('concentration_profiles.png', dpi=300, bbox_inches='tight')
    print("\nPlot saved as 'concentration_profiles.png'")
    plt.show()
    
    return sol

if __name__ == "__main__":
    set_plot_style()
    solution = main()