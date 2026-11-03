"""Composable Gray–Scott reaction–diffusion simulation."""
from .core import Grid, Parameters, GrayScott, initial_state
from .spatial import SpatialOperator, PeriodicFiniteDifference, PeriodicFourier
from .time import TimeStepper, ForwardEuler, RK4
from .simulation import simulate, simulate_adaptive

__all__ = ['Grid', 'Parameters', 'GrayScott', 'initial_state', 'SpatialOperator',
           'PeriodicFiniteDifference', 'PeriodicFourier', 'simulate_adaptive', 'TimeStepper', 'ForwardEuler', 'RK4', 'simulate']
