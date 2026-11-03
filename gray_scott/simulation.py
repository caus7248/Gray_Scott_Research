"""Streaming integration: retain only snapshots you need."""
from numbers import Integral
import numpy as np
from .time import ForwardEuler

def simulate(model, state, *, dt=0.5, steps=1000, save_every=100, stepper=None):
    """Yield (time, copied_state), including initial and final states.

    Custom steppers must not mutate the supplied state. No clipping is applied.
    Explicit methods require the caller to choose a stable time step.
    """
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError('dt must be finite and positive')
    for name, value, minimum in [('steps', steps, 0), ('save_every', save_every, 1)]:
        if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
            raise ValueError(f'{name} must be an integer >= {minimum}')
    state = np.array(state, dtype=float, copy=True)
    if state.shape != (2, *model.grid.shape) or not np.all(np.isfinite(state)):
        raise ValueError('state must be finite with shape (2, ny, nx)')
    method = stepper if stepper is not None else ForwardEuler()
    yield 0.0, state.copy()
    for n in range(1, steps+1):
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            state = method.step(model, (n-1)*dt, state, dt)
        if state.shape != (2, *model.grid.shape) or not np.all(np.isfinite(state)):
            raise FloatingPointError(f'invalid state at step {n}; try a smaller dt')
        if n % save_every == 0 or n == steps:
            yield n*dt, state.copy()
