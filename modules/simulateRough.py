import numpy as np
import jax.numpy as jnp
from jax import tree_util, jit

# remove later
import matplotlib.pyplot as plt

class Space():
    """
    Docstring for Space
    """
    def __init__(self, Domain, tDomain, val):
        self.Domain = Domain
        self.tDomain = tDomain
        self.val = val

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
        
    
"""
  def _tree_flatten(self):
    children = (self.x,)  # arrays / dynamic values
    aux_data = {'mul': self.mul}  # static values
    return (children, aux_data)

  @classmethod
  def _tree_unflatten(cls, aux_data, children):
    return cls(*children, **aux_data)"""

if __name__ == "__main__":
    z_Space = jnp.linspace(0, 1, 100)
    t_Space = jnp.linspace(0, 1, 10_000)

    tree_util.register_pytree_node(Space,
                               Space._tree_flatten,
                               Space._tree_unflatten)

    poreTest = Space(z_Space, t_Space, 0)
    print(poreTest.solver())