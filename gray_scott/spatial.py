"""Spatial discretizations, independent of reaction kinetics and time stepping."""
from typing import Protocol
import numpy as np
from .core import Grid

class SpatialOperator(Protocol):
    def laplacian(self, field: np.ndarray, grid: Grid) -> np.ndarray: ...

class PeriodicFiniteDifference:
    """Second-order five-point Laplacian on a uniform rectangular grid."""
    def laplacian(self, field, grid):
        if field.shape != grid.shape:
            raise ValueError('field shape does not match grid')
        return ((np.roll(field, 1, axis=1)-2*field+np.roll(field, -1, axis=1))/grid.dx**2
                + (np.roll(field, 1, axis=0)-2*field+np.roll(field, -1, axis=0))/grid.dy**2)

    def diffusion_dt_limit(self, grid, diffusivity):
        """Forward-Euler pure-diffusion bound; does not bound reaction stability."""
        return (float('inf') if diffusivity == 0 else
                1/(2*diffusivity*(grid.dx**-2+grid.dy**-2)))
