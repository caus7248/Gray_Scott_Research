"""Paired random-IC study. Run from the repository root with PYTHONPATH=. ."""
import json, time, platform
from pathlib import Path
import numpy as np
import scipy
from gray_scott import *

TIMES = [0., 20., 100., 300.]
PARAMETERS = Parameters(feed=.035, kill=.065)

def random_ic(grid, seed):
    # One continuum field per seed: evaluate identical periodic smooth spots
    # on every grid rather than drawing unrelated grid-wise noise.
    rng = np.random.default_rng(seed)
    centers = rng.uniform(0, 64, (8, 2))
    radii = rng.uniform(2.5, 4.5, 8)
    x = np.arange(grid.nx)*grid.dx
    y = np.arange(grid.ny)*grid.dy
    total = np.zeros(grid.shape)
    for (cx, cy), radius in zip(centers, radii):
        # periodic smooth distance matching Euclidean distance near each spot
        d2 = (64/np.pi)**2*(np.sin(np.pi*(x[None,:]-cx)/64)**2
                                      + np.sin(np.pi*(y[:,None]-cy)/64)**2)
        total += np.exp(-d2/(2*radius**2))
    mask = 1-np.exp(-3*total)
    return np.stack([1-.5*mask, .25*mask])

def run(n, seed, space, method, dt=.1, rtol=1e-7):
    grid=Grid(n,n,64,64)
    model=GrayScott(grid,space(),PARAMETERS)
    initial=random_ic(grid,seed)
    start=time.perf_counter()
    if method=='rk45':
        out=list(simulate_adaptive(model,initial,output_times=TIMES,rtol=rtol,atol=rtol*.001))
    else:
        # Output exactly at all common checkpoints, retaining fixed dt.
        out=[(0.,initial.copy())];state=initial.copy()
        for left,right in zip(TIMES,TIMES[1:]):
            steps=round((right-left)/dt)
            state=list(simulate(model,state,dt=dt,steps=steps,save_every=steps,
                 stepper=ForwardEuler() if method=='euler' else RK4()))[-1][1]
            out.append((right,state))
    return np.stack([s for t,s in out]),time.perf_counter()-start

def metrics(a,b):
    v=a[1];w=b[1]
    centered=v-v.mean();other=w-w.mean()
    denom=np.linalg.norm(centered)*np.linalg.norm(other)
    return dict(rms=float(np.sqrt(np.mean((a-b)**2))),
                v_rms=float(np.sqrt(np.mean((v-w)**2))),
                max_error=float(np.max(abs(a-b))),
                v_mean=float(v.mean()),v_std=float(v.std()),
                v_mean_error=float(abs(v.mean()-w.mean())),
                v_coverage=float(np.mean(v>.1)),
                coverage_error=float(abs(np.mean(v>.1)-np.mean(w>.1))),
                correlation=float(np.sum(centered*other)/denom) if denom else None)

def main():
    folder=Path('results/random_ic');folder.mkdir(parents=True,exist_ok=True)
    rows=[];refs=[];coarse_refs=[];images={};checks=[]
    strategies=[('fd-euler',PeriodicFiniteDifference,'euler',.1,1e-7),
                ('fd-rk4',PeriodicFiniteDifference,'rk4',.5,1e-7),
                ('fd-rk45',PeriodicFiniteDifference,'rk45',.1,1e-7),
                ('fourier-euler',PeriodicFourier,'euler',.1,1e-7),
                ('fourier-rk4',PeriodicFourier,'rk4',.5,1e-7),
                ('fourier-rk45-loose',PeriodicFourier,'rk45',.1,1e-4),
                ('fourier-rk45',PeriodicFourier,'rk45',.1,1e-7)]
    for seed in range(5):
        ref,seconds=run(128,seed,PeriodicFourier,'rk45',rtol=1e-10)
        refs.append(ref)
        print('reference',seed,round(seconds,2),flush=True)
        # Reference grid/tolerance check on one representative seed.
        if seed==0:
            finer,secs=run(256,seed,PeriodicFourier,'rk45',rtol=1e-11)
            checks=[dict(t=t,**metrics(ref[i],finer[i,:,::2,::2])) for i,t in enumerate(TIMES)]
            np.savez_compressed(folder/'reference_check.npz',reference=ref,finer=finer,times=TIMES)
            print('reference refinement',checks[-1],round(secs,2),flush=True)
        for n in (32,64):
            exact=ref[:,:,::128//n,::128//n]
            for name,space,method,dt,tol in strategies:
                try:
                    a,secs=run(n,seed,space,method,dt,tol)
                    for i,t in enumerate(TIMES):
                        rows.append(dict(seed=seed,n=n,strategy=name,t=t,seconds=secs,**metrics(a[i],exact[i])))
                    if seed==0: images[f'{n}_{name}']=a
                    print(seed,n,name,round(secs,3),rows[-1]['rms'],flush=True)
                except (FloatingPointError,RuntimeError) as exc:
                    rows.append(dict(seed=seed,n=n,strategy=name,failed=str(exc)))
                    print('FAILED',seed,n,name,str(exc),flush=True)
        (folder/'metrics.json').write_text(json.dumps(rows,indent=2))
    spread=[]
    for i in range(5):
        for j in range(i):
            spread.append(dict(seed_a=i,seed_b=j,**metrics(refs[i][-1],refs[j][-1])))
    (folder/'ic_spread.json').write_text(json.dumps(spread,indent=2))
    (folder/'reference_check.json').write_text(json.dumps(checks,indent=2))
    np.savez_compressed(folder/'representative.npz',**images,reference=refs[0],times=TIMES)
    metadata=dict(seeds=list(range(5)),times=TIMES,domain=[64,64],feed=.035,kill=.065,
       du=.2,dv=.1,reference_grid=128,reference_rtol=1e-10,reference_atol=1e-13,
       numpy=np.__version__,scipy=scipy.__version__,python=platform.python_version(),
       ic='8 random smooth periodic spots; identical continuum field across grids',
       timing='single CPU run per seed/strategy; excludes output and plotting; no warmup',
       dealiasing='none',reference_check='256 grid seed 0 only; not a rigorous bound for other seeds')
    (folder/'metadata.json').write_text(json.dumps(metadata,indent=2))

if __name__=='__main__':main()
