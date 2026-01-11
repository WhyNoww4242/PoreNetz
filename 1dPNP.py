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

def phi(c_array, ci_len, dist, d_eff_array):
    """
    things need to be rewritting in terms of volume
    """
    rho_array = jnp.reshape(c_array, (ci_len, -1))
    print(rho_array.shape, 'b dot')
    rho_array = jnp.dot(rho_array, d_eff_array.T)
    rho = jnp.sum(rho_array, axis=0)
    del_phi = rho.at[1:].multiply(dist)
    del_phi = rho.at[0].multiply(dist[0])
    return del_phi

def ODE_dis(t, cp, args):
    """
    dist probably needs fixing
    """
    j_left, j_right, dist, a_vec, b_vec, d_eff, ci_len = args
    del_phi = phi(cp, ci_len, dist, d_eff)
    
    cw = jnp.roll(cp, 1)[1:]
    ce = jnp.roll(cp, -1)[:-1]
    
    dcp_dt = jnp.zeros_like(cp)
    dcp_dt = dcp_dt.at[:-1].add((b_vec*del_phi[:-1] - a_vec)*cp[:-1]/dist)  # left flux for cp
    dcp_dt = dcp_dt.at[1:].add((b_vec*del_phi[1:] - a_vec)*cp[1:]/dist)  # right flux for cp
    dcp_dt = dcp_dt.at[:-1].add((b_vec*del_phi[:-1] + a_vec)*cw/dist)  # left flux for cp
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