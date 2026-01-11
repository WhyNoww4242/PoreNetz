import jax.numpy as jnp
from jax import jit, lax
import matplotlib.pyplot as plt
from modules.graphStyle import set_plot_style, add_subplot_labels

def weights(points, D_i, D_eff, BC_left, BC_right):
    dist = jnp.diff(points)
    # creates 
    a_vec = jnp.ones((len(points)+1/))
    b_vec = jnp.ones((len(points)+1))
    a_vec = a_vec.at[1:-1].multiply(D_i/dist)
    b_vec = b_vec.at[1:-1].multiply(D_eff/(2*dist))
    return a_vec, b_vec

def phi(c0, c1, c2):
    return 

def term():
    

def main():
    D_i = jnp.array([1, 1, 1])
    D_eff = jnp.array([1, -2, 1])
    points = jnp.linspace(0, 1, 10)
    
    BC_left = jnp.array([1, 0, 0])
    BC_right = jnp.array([1, 0, 0])
    
    a0, b0 = weights(points, D_i[0], D_eff[0])
    a1, b1 = weights(points, D_i[1], D_eff[1])
    a2, b2 = weights(points, D_i[2], D_eff[2])

if __name__ == "__main__":
    set_plot_style()
    main()    
