import jax.numpy as jnp
from jax import tree_util, jit
import matplotlib.pyplot as plt
# Import plotting utilities from graphStyle.py
from graphStyle import set_plot_style, add_subplot_labels


class Space:
    """
    Space class with domain, time, and named vector fields.
    Each vector field (cp, cm, phi) is a JAX array.
    Fully PyTree-compatible.
    """
    def __init__(self, Domain: jnp.ndarray, tDomain: jnp.ndarray, 
                 cp: jnp.ndarray, cm: jnp.ndarray, phi: jnp.ndarray,
                 diff: float, eff:float):
        self.Domain = Domain
        self.tDomain = tDomain
        self.cp = cp
        self.cm = cm
        self.phi = phi
        self.diff = diff
        self.eff = eff
    
        
    @jit
    def solver(self):
        # Example function: return some sample values
        diffs = jnp.diff(self.Domain)
        delta_vec = jnp.append(diffs, diffs[-1])
        
        delta_t = jnp.diff(self.tDomain)
        diff=self.diff
        eff=self.eff
        a_vec = -diff*1/delta_vec 
        b_vec = eff*1/delta_vec
    
        def delPhi(diagPhi):
            mPhi = jnp.zeros((self.Domain.size, self.Domain.size))
            mPhi += -2*jnp.diag(diagPhi, k=0)
            mPhi += jnp.diag(diagPhi[1:], k=1)
            mPhi += jnp.diag(diagPhi[:-1], k=-1)
            delPhi = jnp.sum(mPhi, axis=1)
            return delPhi*delta_vec
        
        def cMat(cp, cm, alphap, alpham):

            cpMat = jnp.diag(cp, k=0)+jnp.diag(cp[1:], k=1)+jnp.diag(cp[:-1], k=-1)
            cmMat = jnp.diag(cm, k=0)+jnp.diag(cm[1:], k=1)+jnp.diag(cm[:-1], k=-1)
            apMat = jnp.diag(alphap[1:], k=1)+jnp.diag(alphap[:-1], k=-1)
            amMat = jnp.diag(alpham[1:], k=1)+jnp.diag(alpham[:-1], k=-1)

            cp += jnp.sum((cpMat @ apMat), axis=1)
            cm += jnp.sum((cmMat @ amMat), axis=1)
            return cp, cm
        
        def PhiCalc(cp, cm):
            Phi = cp - cm
            return Phi

                # for loop initalization
        cp0 = self.cp
        cm0 = self.cm
        curPhi = PhiCalc(cp0, cm0)
        cp = cp0
        cm = cm0

        cpMat = jnp.diag(cp0, k=0)+jnp.diag(cp0[1:], k=1)+jnp.diag(cp0[:-1], k=-1)
        cmMat = jnp.diag(cm0, k=0)+jnp.diag(cm0[1:], k=1)+jnp.diag(cm0[:-1], k=-1)

        for dt in delta_t:
            nabPhi = delPhi(curPhi)
            alphap = (a_vec)/(2*a_vec+b_vec*nabPhi)*dt
            alpham = (a_vec)/(2*a_vec-b_vec*nabPhi)*dt
            cp, cm = cMat(cp, cm, alphap, alpham)
            curPhi = PhiCalc(cp, cm)
            print(cp)

        return delPhi(self.phi)

    
    def initialPlot(self):
        """Plot the initial concentrations of all vector fields."""
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=300)
        
        vec_names = ['cp', 'cm', 'phi']
        
        for idx, name in enumerate(vec_names):
            vec = getattr(self, name)
            # Plot the initial state (t=0, which is the first column)
            axes[idx].plot(self.Domain, vec[:, 0], linewidth=2)
            axes[idx].set_xlabel(r'Spatial Domain ($z$)')
            axes[idx].set_ylabel(f'{name}', rotation=0, labelpad=20)
            axes[idx].set_title(f'{name}')
        
        # Add subplot labels
        add_subplot_labels(axes)
        
        plt.tight_layout()
        plt.savefig('initial_concentrations.png')
        plt.show()
    
    # PyTree flatten
    def _tree_flatten(self):
        children = (self.Domain, self.tDomain, self.cp, self.cm, self.phi,)
        aux_data = {'diff': self.diff, 'eff': self.eff}
        return children, aux_data
    
    # PyTree unflatten
    @classmethod
    def _tree_unflatten(cls, aux_data, children): 
        return cls(*children, **aux_data)


# Register as a PyTree
tree_util.register_pytree_node(Space, Space._tree_flatten, Space._tree_unflatten)


if __name__ == "__main__":
    # Set plot style
    set_plot_style()
    
    # Spatial and temporal domains
    z_Space = jnp.linspace(0, 1, 100)
    t_Space = jnp.linspace(0, 1, 100_000)
    
    # Initial vectors as JAX arrays
    cm = jnp.arange(1,101)# jnp.ones((z_Space.size))
    cp = jnp.ones((z_Space.size))
    phi = jnp.ones((z_Space.size))

    diff = 1
    print(phi,'phi')
    # Create Space object
    poreTest = Space(z_Space, t_Space, cp, cm, phi, diff, diff)
    
    # Call solver
    print(poreTest.solver())
    
    # Plot initial concentrations
    # poreTest.initialPlot()