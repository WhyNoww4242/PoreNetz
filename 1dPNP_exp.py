import numpy as np
import jax.numpy as jnp
from jax import jit, lax
import matplotlib.pyplot as plt
from modules.graphStyle import set_plot_style, add_subplot_labels
import diffrax

def volumeMask(x0:float = 0, dx_array:np.array = np.full(400, 1e-2))->jnp.array:
    """
    Manually sets the control volumes of discretation and returns centroids of volumes
    
    Parameters:
    -----------
    x0: inital domain point
    dx_array: 1D array of control volume element sizes
        - Default size 100 with all volumes dx = 0.01

    Returns:
    --------
    points_array: jnp.array of points
    vol_array: jnp.array of control volume values
    """
    points = np.zeros_like(dx_array)  # point locations 
    points[0] = x0+dx_array[0]/2  # initial point
    # loops over volume array and calculates centroid location
    for i in range(1, len(dx_array)):
        points[i] = points[i-1]+dx_array[i]/2
    # converts into jax array
    vol_array = jnp.array(dx_array)
    points_array = jnp.array(points)
    return vol_array, points_array

def weights(points:jnp.array, D_i:float, D_eff:float):
    """
    Finds weights vectors for the FVM 
    
    Parameters:
    -----------
    points: jnp.array of volume centroids
    D_i: diffusion coefficient of ion i
    D_eff: effective diffusion of the electromigration term

    Returns:
    --------
    a_vec: jnp.array of diffusive coefficents for species i
    b_vec: jnp.array of electormigrative coefficents for species i
    """
    dist = jnp.diff(points)  # finds all point distances
    # creates a_vec(diffusion) and b_vec (electromigration) discretions 
    a_vec = jnp.ones_like(dist)
    b_vec = jnp.ones_like(dist)
    a_vec = a_vec.at[:].set(D_i/dist)
    b_vec = b_vec.at[:].set(D_eff/(2*dist))
    return a_vec, b_vec

def grad_phi(cp_tensor, d_effs, vol_vec, Debye):
    """
    """
    # this is bad i am sorry future bryce
    qs = 0.025
    n = len(q_encl)+1
    q_encl = jnp.dot(d_effs, cp_tensor)
    gamma_vec = jnp.zeros(n)
    gamma_vec = gamma_vec.at[1:].add(-q_encl/(Debye*vol_vec))
    gamma_vec = gamma_vec.at[0].set(qs)

    # solve system Ax=b by using inverse x = b A^-1
    A = -jnp.eye(n) + jnp.eye(n, k=-1)
    del_phi = jnp.dot(gamma_vec, jnp.linalg.inv(A))

    return del_phi

def ODE_dis(t, cp_vec, args):
    """
    """
    js_left, js_right, vol_vec, as_vec, bs_vec, d_effs, Debye= args

    ions_num = len(js_left)  # number of ions based on flux vector
    cp_tensor = jnp.reshape(cp_vec, (-1, ions_num))
    cw_tensor = jnp.roll(cp_tensor, 1)[1:,:]
    ce_tensor = jnp.roll(cp_tensor, -1)[:-1,:]
    del_phi = grad_phi(cp_tensor, d_effs, vol_vec)

    for i in range(ions_num):
      # left (west) flux to centroid 
      cp_tensor =  cp_tensor.at[:-1,i].multiply((bs_vec[:,i]*del_phi[:-1] - as_vec[:,i]))
      cp_tensor =  cp_tensor.at[:-1,i].add((bs_vec[:,i]*del_phi[:-1] + as_vec[:,i])*cw_tensor[:,i])
      # right (east) flux to centroid 
      cp_tensor =  cp_tensor.at[1:,i].multiply((bs_vec[:,i]*del_phi[1:] - as_vec[:,i]))
      cp_tensor =  cp_tensor.at[1:,i].add((bs_vec[:,i]*del_phi[1:] + as_vec[:,i])*ce_tensor[:,i])
      
      # constant flux boundary conditions
      cp_tensor = cp_tensor.at[0,i].add(js_left[i])  # left
      cp_tensor = cp_tensor.at[-1,i].add(js_right[i])  # right
      
      cp_tensor = cp_tensor.at[:,i].divide(vol_vec)  # divides by volume of each cell
    
    dcp_dt = jnp.reshape(cp_tensor, -1)
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