import numpy as np
import jax.numpy as jnp
from jax import tree_util, jit

# remove later
import matplotlib.pyplot as plt

class Space():
    """
    Docstring for Space
    """
    def __init__(self, Domain, tDomain, initial_vec):
        self.Domain = Domain
        self.tDomain = tDomain
        self.empty = jnp.zeros((Domain.size,tDomain.size))
        self.val = val

        self.cp = jnp.zeros_like(self.empty)
        self.cm = jnp.zeros_like(self.empty)
        self.phi = jnp.zeros_like(self.empty)
        print('print',self.cp.nbytes)

    @jit
    def solver(self):
        return self.Domain[10], self.tDomain[10] 
    
    def _tree_flatten(self):
        children = (self.Domain, self.tDomain) 
        aux_data = {'val': self.val}  # static values
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)
        

if __name__ == "__main__":
    z_Space = jnp.linspace(0, 1, 1000)
    t_Space = jnp.linspace(0, 1, 100_000)

    tree_util.register_pytree_node(Space,
                               Space._tree_flatten,
                               Space._tree_unflatten)

    poreTest = Space(z_Space, t_Space, 0)
    
    print(poreTest.solver())