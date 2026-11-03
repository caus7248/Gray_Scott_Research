"""Run a simulation, saving numerical snapshots and reproducibility metadata."""
import argparse
import json
from pathlib import Path
import numpy as np
import scipy
from dataclasses import asdict
from . import Grid, Parameters, GrayScott, PeriodicFiniteDifference, PeriodicFourier, simulate_adaptive, ForwardEuler, RK4, initial_state, simulate

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nx', type=int, default=128)
    parser.add_argument('--ny', type=int, default=128)
    parser.add_argument('--lx', type=float, default=128)
    parser.add_argument('--ly', type=float, default=128)
    parser.add_argument('--dt', type=float, default=0.5)
    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--save-every', type=int, default=100)
    parser.add_argument('--feed', type=float, default=0.014)
    parser.add_argument('--kill', type=float, default=0.045)
    parser.add_argument('--du', type=float, default=0.2)
    parser.add_argument('--dv', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--stepper', choices=['euler', 'rk4', 'rk45'], default='euler')
    parser.add_argument('--spatial', choices=['fd', 'fourier'], default='fd')
    parser.add_argument('--rtol', type=float, default=1e-6)
    parser.add_argument('--atol', type=float, default=1e-9)
    parser.add_argument('--max-step', type=float, default=np.inf)
    parser.add_argument('--output', type=Path, default=Path('outputs/run'))
    parser.add_argument('--plot', action='store_true', help='save final.png (requires matplotlib)')
    args = parser.parse_args()
    grid = Grid(args.nx, args.ny, args.lx, args.ly)
    parameters = Parameters(args.du, args.dv, args.feed, args.kill)
    spatial = PeriodicFiniteDifference() if args.spatial == 'fd' else PeriodicFourier()
    if args.stepper == 'euler' and args.dt > spatial.diffusion_dt_limit(grid, max(args.du, args.dv)):
        parser.error('dt exceeds the forward-Euler diffusion bound; reduce dt')
    model = GrayScott(grid, spatial, parameters)
    stepper = ForwardEuler() if args.stepper == 'euler' else RK4()
    if args.stepper == 'rk45':
        if (args.steps < 0 or args.save_every < 1 or not np.isfinite(args.dt)
                or args.dt <= 0):
            parser.error('steps >= 0, save-every >= 1 and finite dt > 0 required')
        indices = list(range(0, args.steps+1, args.save_every))
        if indices[-1] != args.steps:
            indices.append(args.steps)
        snapshots = simulate_adaptive(model, initial_state(grid, seed=args.seed),
            output_times=np.asarray(indices)*args.dt, rtol=args.rtol,
            atol=args.atol, max_step=args.max_step)
    else:
        snapshots = simulate(model, initial_state(grid, seed=args.seed), dt=args.dt,
                             steps=args.steps, save_every=args.save_every, stepper=stepper)
    # Validate the integration configuration before creating the output directory.
    first = next(snapshots)
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = dict(grid=asdict(grid), parameters=asdict(parameters), dt=args.dt,
                    steps=args.steps, save_every=args.save_every, seed=args.seed,
                    stepper=args.stepper, spatial=args.spatial, rtol=args.rtol, atol=args.atol,
                    max_step=None if np.isinf(args.max_step) else args.max_step,
                    dealiasing='none',
                    numpy_version=np.__version__, scipy_version=scipy.__version__, initial_condition=dict(width=0.15, noise=0.02),
                    status='running')
    metadata_path = args.output / 'metadata.json'
    metadata_path.write_text(json.dumps(metadata, indent=2)+'\n')
    def save(index, time, state):
        np.savez_compressed(args.output / f'snapshot_{index:06d}.npz', t=time, u=state[0], v=state[1])
    t, state = first
    save(0, t, state)
    for index, (t, state) in enumerate(snapshots, 1):
        save(index, t, state)
    metadata['status'] = 'complete'
    metadata_path.write_text(json.dumps(metadata, indent=2)+'\n')
    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        im = ax.imshow(state[1], origin='lower', extent=(0, grid.lx, 0, grid.ly), cmap='magma')
        ax.set(xlabel='x', ylabel='y', title=f'Gray–Scott v, t={t:g}')
        fig.colorbar(im, ax=ax)
        fig.savefig(args.output / 'final.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
    print(f'Saved simulation to {args.output}')

if __name__ == '__main__':
    main()
