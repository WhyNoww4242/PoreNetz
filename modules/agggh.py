import jax.numpy as jnp
from jax import jit, lax
import matplotlib.pyplot as plt
from graphStyle import set_plot_style, add_subplot_labels


class Space:
    def __init__(self, Domain, tDomain, cp, cm, phi, diff, eff, phi_w):
        self.Domain = Domain
        self.tDomain = tDomain
        self.cp = cp
        self.cm = cm
        self.phi = phi
        self.diff = diff
        self.eff = eff
        self.phi_w = phi_w  # Background/wall potential
    
    def solver(self):
        return self._solve_core(self.Domain, self.tDomain, self.cp, self.cm, 
                                self.phi, self.diff, self.eff, self.phi_w)
    
    @staticmethod
    @jit
    def _solve_core(Domain, tDomain, cp_init, cm_init, phi_init, diff, eff, phi_w):
        n = Domain.size
        delta_z = Domain[1] - Domain[0]  # Scalar spacing
        delta_t = jnp.diff(tDomain)
        
        # Build Laplacian matrix for Poisson solve
        def build_laplacian():
            # Using central difference: (φ[i-1] - 2φ[i] + φ[i+1]) / dz²
            diag_main = -2.0 * jnp.ones(n) / (delta_z**2)
            diag_off = jnp.ones(n-1) / (delta_z**2)
            
            # Build tridiagonal matrix
            M = jnp.zeros((n, n))
            M = M.at[jnp.arange(n), jnp.arange(n)].set(diag_main)
            M = M.at[jnp.arange(n-1), jnp.arange(1, n)].set(diag_off)
            M = M.at[jnp.arange(1, n), jnp.arange(n-1)].set(diag_off)
            
            # Boundary conditions: Dirichlet at both ends
            # Left: φ[0] = phi_w
            M = M.at[0, :].set(0.0)
            M = M.at[0, 0].set(1.0)
            
            # Right: φ[-1] = 0 (or your other BC)
            M = M.at[-1, :].set(0.0)
            M = M.at[-1, -1].set(1.0)
            
            return M
        
        M_lap = build_laplacian()
        
        def solve_poisson(cp, cm):
            """Solve ∇²φ = cp - cm for φ"""
            rho = cp - cm
            
            # Right-hand side with boundary conditions
            b = rho
            b = b.at[0].set(phi_w)  # Left BC: φ = phi_w
            b = b.at[-1].set(0.0)   # Right BC: φ = 0
            
            # Solve M φ = b
            phi = jnp.linalg.solve(M_lap, b)
            return phi**2
        
        # Coefficients (scalars)
        a_coeff = diff / delta_z 
        b_coeff = eff / delta_z
        
        def step_fn(carry, dt):
            cp_c, cm_c, phi_c = carry
            
            # 1. SOLVE POISSON: ∇²φ = cp - cm
            phi_new = solve_poisson(cp_c, cm_c)
            
            # 2. Compute ∇φ for migration term (central difference)
            grad_phi = jnp.zeros_like(phi_new)
            grad_phi = grad_phi.at[1:-1].set((phi_new[2:] - phi_new[:-2]) / (2*delta_z))
            # Boundaries use one-sided differences
            grad_phi = grad_phi.at[0].set((phi_new[1] - phi_new[0]) / delta_z)
            grad_phi = grad_phi.at[-1].set((phi_new[-1] - phi_new[-2]) / delta_z)
            
            # Initialize next step
            cp_next = cp_c.copy()
            cm_next = cm_c.copy()
            
            # Interior points: diffusion + migration
            interior_idx = jnp.arange(1, n-1)
            
            # Diffusion flux (central difference for second derivative)
            laplacian_cp = (cp_c[:-2] - 2*cp_c[1:-1] + cp_c[2:]) / (delta_z**2)
            laplacian_cm = (cm_c[:-2] - 2*cm_c[1:-1] + cm_c[2:]) / (delta_z**2)
            
            # Concentration gradient (for migration)
            grad_cp = (cp_c[2:] - cp_c[:-2]) / (2*delta_z)
            grad_cm = (cm_c[2:] - cm_c[:-2]) / (2*delta_z)
            
            # Full flux: diffusion + migration
            # For cations (+): flux = D∇²c + μ c ∇φ
            # For anions (-): flux = D∇²c - μ c ∇φ
            flux_p = a_coeff * laplacian_cp + b_coeff * (grad_cp * grad_phi[1:-1] + cp_c[1:-1] * (phi_new[2:] - 2*phi_new[1:-1] + phi_new[:-2])/(delta_z**2))
            flux_m = 2*a_coeff * laplacian_cm - 2*b_coeff * (grad_cm * grad_phi[1:-1] + cm_c[1:-1] * (phi_new[2:] - 2*phi_new[1:-1] + phi_new[:-2])/(delta_z**2))
            
            cp_next = cp_next.at[interior_idx].set(cp_c[interior_idx] + flux_p * dt)
            cm_next = cm_next.at[interior_idx].set(cm_c[interior_idx] + flux_m * dt)
            
            # LEFT BOUNDARY [0]: dc/dx = eff * C (Robin BC)
            # Forward difference: (c[1] - c[0])/dz = eff * c[0]
            # c_new[0] = c[0] + dt * D * (c[1] - c[0])/dz² + dt * eff * c[0]
            bc_flux_p = a_coeff * (cp_c[1] - cp_c[0]) / delta_z + eff * cp_c[0]
            bc_flux_m = a_coeff * (cm_c[1] - cm_c[0]) / delta_z + eff * cm_c[0]
            
            cp_next = cp_next.at[0].set(cp_c[0] + bc_flux_p * dt)
            cm_next = cm_next.at[0].set(cm_c[0] + bc_flux_m * dt)
            
            # RIGHT BOUNDARY [-1]: No flux (dC/dx = 0)
            # Use one-sided difference to maintain no-flux
            cp_next = cp_next.at[-1].set(cp_next[-2])
            cm_next = cm_next.at[-1].set(cm_next[-2])
            
            return (cp_next, cm_next, phi_new), (cp_next, cm_next, phi_new)
        
        _, history = lax.scan(step_fn, (cp_init, cm_init, phi_init), delta_t)
        return history
    
    def plot_refined(self, history, requested_times):
        set_plot_style()
        
        fig, axes = plt.subplots(1, len(requested_times), figsize=(15, 3.5), 
                                dpi=300, sharey=True)
        
        cp_hist, cm_hist, phi_hist = history
        
        for i, t_target in enumerate(requested_times):
            idx = jnp.argmin(jnp.abs(self.tDomain[1:] - t_target))
            cp_v, cm_v, phi_v = cp_hist[idx], cm_hist[idx], phi_hist[idx]
            
            axes[i].plot(self.Domain, cp_v, label=r'$c_+$', lw=3.5)
            axes[i].plot(self.Domain, cm_v, label=r'$c_-$', lw=3.5, ls='--')
            axes[i].plot(self.Domain, phi_v, label=r'$\phi$', lw=3.5, alpha=0.4)
            
            axes[i].set_title(rf'$\tau = {t_target:.3f}$', fontsize=14, fontweight='bold')
            axes[i].set_xlabel(r'Position ($Z$)')
            axes[i].set_xticks([0, 0.5, 1.0])
            
            if i == 0:
                axes[i].set_ylabel('Concentration')
            
            axes[i].legend(frameon=False, loc='best')
        
        add_subplot_labels(axes)
        plt.tight_layout()
        plt.savefig('concentration_evolution.png')
        plt.show()


if __name__ == "__main__":
    set_plot_style()
    
    z_Space = jnp.linspace(0, 1, 100)
    t_Space = jnp.linspace(0, 0.1, 50_000)
    
    cp_init = jnp.ones(100)
    cm_init = jnp.ones(100)
    phi_init = jnp.zeros(100)
    phi_w = 1.0  # Wall potential at left boundary
    
    pore = Space(z_Space, t_Space, cp_init, cm_init, phi_init, 
                 diff=0.1, eff=0.05, phi_w=phi_w)
    history = pore.solver()
    pore.plot_refined(history, [0.02, 0.05, 0.1])