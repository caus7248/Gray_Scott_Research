"""Model and grid definitions; array layout is (species, y, x)."""
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class Grid:
    nx: int = 128
    ny: int = 128
    lx: float = 128.0
    ly: float = 128.0

    def __post_init__(self):
        if any(isinstance(n, bool) or not isinstance(n, int) or n < 3 for n in (self.nx, self.ny)):
            raise ValueError('nx and ny must be integers >= 3')
        if not all(np.isfinite(v) and v > 0 for v in (self.lx, self.ly)):
            raise ValueError('domain lengths must be finite and positive')

    @property
    def dx(self): return self.lx / self.nx
    @property
    def dy(self): return self.ly / self.ny
    @property
    def shape(self): return (self.ny, self.nx)

@dataclass(frozen=True)
class Parameters:
    du: float = 0.2
    dv: float = 0.1
    feed: float = 0.014
    kill: float = 0.045

    def __post_init__(self):
        if not all(np.isfinite(v) and v >= 0 for v in (self.du, self.dv, self.feed, self.kill)):
            raise ValueError('model parameters must be finite and nonnegative')

class GrayScott:
    """Method-of-lines RHS with an injected spatial operator."""
    def __init__(self, grid, spatial, parameters=None):
        self.grid, self.spatial = grid, spatial
        self.parameters = parameters if parameters is not None else Parameters()

    def reaction(self, t, state):
        """Pointwise reaction; spatial collocation products are not dealiased."""
        u, v = state
        p = self.parameters
        r = u*v*v
        return np.stack((-r+p.feed*(1-u), r-(p.feed+p.kill)*v))

    def diffusion(self, t, state):
        p = self.parameters
        return np.stack((p.du*self.spatial.laplacian(state[0], self.grid),
                         p.dv*self.spatial.laplacian(state[1], self.grid)))

    def __call__(self, t, state):
        return self.diffusion(t, state) + self.reaction(t, state)

def initial_state(grid, seed=0, width=0.15, noise=0.02):
    """Centered patch, with optional reproducible perturbations inside it."""
    if not np.isfinite(width) or not 0 < width <= 1:
        raise ValueError('width must lie in (0, 1]')
    if not np.isfinite(noise) or not 0 <= noise <= 0.25:
        raise ValueError('noise must lie in [0, 0.25]')
    y, x = np.indices(grid.shape)
    patch = ((abs((x+0.5)/grid.nx-0.5) <= width/2) &
             (abs((y+0.5)/grid.ny-0.5) <= width/2))
    state = np.stack((np.ones(grid.shape), np.zeros(grid.shape)))
    state[0, patch], state[1, patch] = 0.5, 0.25
    rng = np.random.default_rng(seed)
    state[:, patch] += noise * rng.uniform(-1, 1, (2, patch.sum()))
    return state
