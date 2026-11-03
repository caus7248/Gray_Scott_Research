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


def simulate_adaptive(model, state, *, output_times, rtol=1e-6, atol=1e-9,
                      max_step=np.inf):
    """Stream RK45 dense output at specified times, independent of internal steps.

    Times must start at zero and increase strictly. SciPy performs embedded
    Dormand–Prince 5(4) error control and step rejection. This remains explicit.
    """
    from scipy.integrate import RK45
    times = np.asarray(output_times, dtype=float)
    if (times.ndim != 1 or len(times) == 0 or not np.isfinite(times).all()
            or times[0] != 0 or np.any(np.diff(times) <= 0)):
        raise ValueError('output_times must start at zero and increase strictly')
    if not all(np.isfinite(x) and x > 0 for x in (rtol, atol)):
        raise ValueError('rtol and atol must be finite positive scalars')
    if np.isnan(max_step) or max_step <= 0:
        raise ValueError('max_step must be positive')
    state = np.array(state, dtype=float, copy=True)
    shape = (2, *model.grid.shape)
    if state.shape != shape or not np.isfinite(state).all():
        raise ValueError('state must be finite with shape (2, ny, nx)')
    yield 0.0, state.copy()
    if len(times) == 1:
        return
    def rhs(t, flat):
        result = model(t, flat.reshape(shape))
        if not np.isfinite(result).all():
            raise FloatingPointError('nonfinite RHS during adaptive integration')
        return result.ravel()
    solver = RK45(rhs, 0, state.ravel(), times[-1], rtol=rtol,
                  atol=atol, max_step=max_step)
    index = 1
    while solver.status == 'running':
        message = solver.step()
        if solver.status == 'failed':
            raise RuntimeError(f'RK45 failed: {message}')
        interpolant = solver.dense_output()
        while index < len(times) and times[index] <= solver.t:
            result = interpolant(times[index]).reshape(shape)
            if not np.isfinite(result).all():
                raise FloatingPointError('nonfinite adaptive output')
            yield float(times[index]), result.copy()
            index += 1
