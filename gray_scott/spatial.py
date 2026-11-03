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


class PeriodicFourier:
    """Fourier collocation Laplacian. Nonlinear products remain on the base grid.

    No dealiasing is implied: refine the grid to check nonlinear aliasing errors.
    """
    def __init__(self):
        self._grid = None
        self._symbol = None

    def _multiplier(self, grid):
        if grid != self._grid:
            kx = 2*np.pi*np.fft.fftfreq(grid.nx, d=grid.dx)
            ky = 2*np.pi*np.fft.fftfreq(grid.ny, d=grid.dy)
            self._symbol = -(ky[:, None]**2+kx[None, :]**2)
            self._grid = grid
        return self._symbol

    def laplacian(self, field, grid):
        if field.shape != grid.shape:
            raise ValueError('field shape does not match grid')
        result = np.fft.ifft2(self._multiplier(grid)*np.fft.fft2(field))
        return result.real if np.isrealobj(field) else result

    def diffusion_dt_limit(self, grid, diffusivity):
        return float('inf') if diffusivity == 0 else 2/(diffusivity*np.max(-self._multiplier(grid)))
