"""Render the saved random-IC ensemble; no simulations are rerun."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

folder=Path('results/random_ic')
rows=json.loads((folder/'metrics.json').read_text())
last=[r for r in rows if r.get('t')==300]
labels=['fd-euler','fd-rk4','fd-rk45','fourier-euler','fourier-rk4','fourier-rk45-loose','fourier-rk45']
summary=[]
for n in [32,64]:
 for method in labels:
  subset=[r for r in last if r['n']==n and r['strategy']==method]
  if not subset:continue
  record=dict(n=n,strategy=method)
  for metric in ['seconds','rms','max_error','v_mean_error','coverage_error','correlation']:
   vals=[r[metric] for r in subset]
   record[metric+'_median']=float(np.median(vals))
   record[metric+'_min']=float(np.min(vals))
   record[metric+'_max']=float(np.max(vals))
  summary.append(record)
(folder/'summary.json').write_text(json.dumps(summary,indent=2))

fig,ax=plt.subplots(figsize=(9,6),layout='constrained')
colors=plt.get_cmap('tab10')
for j,method in enumerate(labels):
 for n,marker in [(32,'o'),(64,'s')]:
  r=next((r for r in summary if r['n']==n and r['strategy']==method),None)
  if not r:continue
  x=r['seconds_median'];y=r['rms_median']
  ax.errorbar(x,y,xerr=[[x-r['seconds_min']],[r['seconds_max']-x]],
       yerr=[[y-r['rms_min']],[r['rms_max']-y]],fmt=marker,color=colors(j),capsize=3,
       label=f'{method}, {n}²')
ax.set(xscale='log',yscale='log',xlabel='Runtime per trajectory (seconds)',
       ylabel='Final RMS difference from 128² Fourier reference',
       title='Random-IC sensitivity at t=300 — five paired seeds')
ax.grid(True,which='both',alpha=.2)
ax.legend(fontsize=8,ncol=2)
fig.savefig(folder/'runtime_accuracy.png',dpi=160);plt.close(fig)

z=np.load(folder/'representative.npz')
ref=z['reference'][-1,1,::2,::2]
methods=['fd-euler','fd-rk4','fd-rk45','fourier-euler','fourier-rk45']
fig,axes=plt.subplots(2,6,figsize=(16,6),layout='constrained')
vmax=max(float(ref.max()),*[float(z['64_'+m][-1,1].max()) for m in methods])
errmax=max(float(np.max(abs(z['64_'+m][-1,1]-ref))) for m in methods)
axes[0,0].imshow(ref,origin='lower',extent=[0,64,0,64],cmap='magma',vmin=0,vmax=vmax)
axes[0,0].set_title('Fine-grid reference')
axes[1,0].imshow(z['64_fd-euler'][0,1],origin='lower',extent=[0,64,0,64],cmap='magma',vmin=0,vmax=vmax)
axes[1,0].set_title('Shared initial v')
for j,m in enumerate(methods,1):
    field=z['64_'+m][-1,1]
    im=axes[0,j].imshow(field,origin='lower',extent=[0,64,0,64],cmap='magma',vmin=0,vmax=vmax)
    axes[0,j].set_title(m,fontsize=10)
    err=axes[1,j].imshow(field-ref,origin='lower',extent=[0,64,0,64],cmap='RdBu_r',vmin=-errmax,vmax=errmax)
    axes[1,j].set_title('Difference',fontsize=10)
for ax in axes.ravel():ax.set_xticks([]);ax.set_yticks([])
fig.colorbar(im,ax=axes[0,:],shrink=.8,label='v')
fig.colorbar(err,ax=axes[1,1:],shrink=.8,label='v difference')
fig.suptitle('Same random IC, different algorithms — seed 0, 64² grid, t=300\nCommon color scales; differences include spatial and temporal error',fontsize=13)
fig.savefig(folder/'patterns.png',dpi=160);plt.close(fig)
print(json.dumps(summary,indent=2))

fig,axes=plt.subplots(2,3,figsize=(10,7),layout='constrained')
for i,n in enumerate([32,64]):
    ref=z['reference'][-1,1,::128//n,::128//n]
    for j,(label,field) in enumerate([('Reference',ref),('FD + RK45',z[f'{n}_fd-rk45'][-1,1]),('Fourier + RK45',z[f'{n}_fourier-rk45'][-1,1])]):
        im=axes[i,j].imshow(field,origin='lower',extent=[0,64,0,64],vmin=0,vmax=.4,cmap='magma')
        axes[i,j].set_title(f'{label} — {n}²')
        axes[i,j].set_xticks([]);axes[i,j].set_yticks([])
fig.colorbar(im,ax=axes.ravel().tolist(),shrink=.7,label='v concentration')
fig.suptitle('Random seed 0: grid choice changes the predicted spots\nSame continuum initial field, same tolerances, t=300')
fig.savefig(folder/'grid_patterns.png',dpi=160);plt.close(fig)
