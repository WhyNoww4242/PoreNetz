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


def ODE_dis_old(t, cp, args):
    """
    """
    j_left, j_right, dist, a_vec, b_vec, d_eff, ci_len = args
    del_phi = phi(cp, ci_len, dist, d_eff)
    
    cw = jnp.roll(cp, 1)[1:]
    ce = jnp.roll(cp, -1)[:-1]
    
    dcp_dt = jnp.zeros_like(cp)
    dcp_dt = dcp_dt.at[:-1].add((b_vec*del_phi[:-1] - a_vec)*cp[:-1]/dist)  # left flux for cp
    dcp_dt = dcp_dt.at[:-1].add((b_vec*del_phi[:-1] + a_vec)*cw/dist)  # left flux for cp
    dcp_dt = dcp_dt.at[1:].add((b_vec*del_phi[1:] - a_vec)*cp[1:]/dist)  # right flux for cp
    dcp_dt = dcp_dt.at[1:].add((b_vec*del_phi[1:] + a_vec)*ce/dist)  # right flux for cp
    
    # BC added and distance assumed to be mirroed
    dcp_dt = dcp_dt.at[0].add(j_left/dist[0])
    dcp_dt = dcp_dt.at[-1].add(j_right/dist[-1])
    
    return dcp_dt

def main():
    D_i = jnp.array([1, 1, 1])
    D_eff = jnp.array([1, -2, 1])
    points = jnp.linspace(0, 1, 10)
    BC_left = jnp.array([1, 0, 0])
    BC_right = jnp.array([1, 0, 0])
    
    # Get spatial discretization info
    dist = jnp.diff(points)
    ci_len = len(D_i)
    
    # Calculate weights for each species
    a0, b0 = weights(points, D_i[0], D_eff[0])
    a1, b1 = weights(points, D_i[1], D_eff[1])
    a2, b2 = weights(points, D_i[2], D_eff[2])
    
    # Stack the weights for all species
    # Note: You'll need to decide how to combine these for multiple species
    # For now, using the first species as an example
    a_vec = a0
    b_vec = b0
    
    # Define boundary fluxes
    j_left = 0.0  # Define your left boundary flux
    j_right = 0.0  # Define your right boundary flux
    
    # Pack args tuple - THIS WAS MISSING!
    args = (j_left, j_right, dist, a_vec, b_vec, D_eff, ci_len)
    
    term = diffrax.ODETerm(ODE_dis)
    y0 = jnp.ones((3*len(points)))
    
    # Temporal discretisation
    t0 = 0
    t_final = 1
    δt = 0.0001
    saveat = diffrax.SaveAt(ts=jnp.linspace(t0, t_final, 50))
    
    # Tolerances
    rtol = 1e-10
    atol = 1e-10
    stepsize_controller = diffrax.PIDController(
        pcoeff=0.3, icoeff=0.4, rtol=rtol, atol=atol, dtmax=0.001
    )
    
    solver = diffrax.Tsit5()
    
    # Pass args to diffeqsolve - THIS WAS MISSING!
    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0,
        t_final,
        δt,
        y0,
        args=args,  # <-- ADD THIS LINE
        saveat=saveat,
        stepsize_controller=stepsize_controller,
        max_steps=None,
    )
    
    print("Solution shape:", sol.ys.shape)
    return sol

if __name__ == "__main__":
    set_plot_style()
    main()