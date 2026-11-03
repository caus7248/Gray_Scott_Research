# Gray–Scott Research

Fresh Python implementation of

```text
u_t = Du Δu - uv² + F(1-u)
v_t = Dv Δv + uv² - (F+k)v
```

The previous MATLAB and Python scripts are replaced; they remain in Git history.
This is a new implementation, not a bitwise reproduction of earlier runs.
The initial patch and five-point stencil differ from the old Python defaults.

## Setup and run

Python 3.10 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e '.[plot,test]'
python -m gray_scott --steps 10000 --plot --output outputs/waves
python -m pytest -q
```

Use `python -m gray_scott --help` for grid, physical parameters and output options.
Select RK4 with `--stepper rk4`. Output directories must be new to prevent
accidental overwrite. NPZ snapshots contain scalar `t` and `(ny,nx)` arrays `u,v`.
`metadata.json` records parameters, seed, grid, method, NumPy version and run status.
Failed or interrupted runs retain `status: running`. Snapshots include initial,
final and every requested sampling step. Plotting is optional; no ML dependencies
are required.

## Swappable components

- `core.py`: grid, parameters, initial condition and model RHS.
- `spatial.py`: `SpatialOperator` protocol and periodic finite differences.
- `time.py`: `TimeStepper` protocol, forward Euler and classical RK4.
- `simulation.py`: streaming integration and validation.
- `__main__.py`: CLI, numerical output and optional final plot.

```python
from gray_scott import (
    Grid, GrayScott, PeriodicFiniteDifference, RK4, initial_state, simulate,
)
grid = Grid(nx=128, ny=96, lx=128, ly=96)
model = GrayScott(grid, PeriodicFiniteDifference())
for t, state in simulate(model, initial_state(grid), dt=0.5,
                         steps=1000, save_every=100, stepper=RK4()):
    u, v = state
```

A spatial operator implements `laplacian(field, grid)`, returning a same-shape
array. It owns boundary handling. A time stepper implements
`step(rhs, t, state, dt)`, returning a new state without mutating its input.
The model is callable as `rhs(t, state)` with state layout `(2, ny, nx)` in u,v
order. Future implicit or split methods can use their own configured linear
solvers/model access behind this interface; none are implemented yet.

## Numerical specification

Uniform rectangular periodic grid without duplicated endpoints. The baseline
Laplacian uses the second-order five-point stencil, with explicit dx and dy:

```text
Lh(w)[j,i] = (w[j,i-1] - 2w[j,i] + w[j,i+1])/dx²
          + (w[j-1,i] - 2w[j,i] + w[j+1,i])/dy²
```

Euler is first order and RK4 fourth order in time for smooth solutions.
Forward Euler's pure-diffusion bound is
`dt <= 1 / (2 max(Du,Dv) (1/dx² + 1/dy²))`.
The CLI rejects Euler steps above this bound. Reaction stability can impose
additional restrictions: the diffusion bound alone is not sufficient.
The Python API leaves step size to the caller. RK4 also has a finite stability
region; neither method is adaptive. Nonfinite states fail explicitly and no
clipping is applied. At fixed domain size, explicit diffusion steps scale as h².

Defaults: Du=0.2, Dv=0.1, F=0.014, k=0.045, dt=0.5, 128×128 domain and grid.
Initial state: u=1,v=0 outside a centered patch; u=0.5,v=0.25 inside it,
with reproducible bounded noise in the patch.

Tests cover spatial and temporal convergence, periodic conservation, equilibrium,
component substitution, snapshot ownership, nontrivial evolution and invalid input.
