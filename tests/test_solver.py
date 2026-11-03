import numpy as np
import pytest
from gray_scott import *

def test_spatial_order_and_conservation():
    errors=[]
    for n in (24,48,96):
        g=Grid(2*n,n,5,3)
        y,x=np.meshgrid(np.arange(g.ny)*g.dy,np.arange(g.nx)*g.dx,indexing='ij')
        u=np.sin(2*np.pi*x/5)*np.cos(2*np.pi*y/3)
        lap=PeriodicFiniteDifference().laplacian(u,g)
        errors.append(np.max(abs(lap+((2*np.pi/5)**2+(2*np.pi/3)**2)*u)))
        assert abs(lap.sum()) < 1e-10
        assert np.max(abs(PeriodicFiniteDifference().laplacian(np.ones(g.shape),g))) == 0
    assert all(3.8<a/b<4.2 for a,b in zip(errors,errors[1:]))

@pytest.mark.parametrize('method,order',[(ForwardEuler(),1),(RK4(),4)])
def test_time_order(method,order):
    errors=[]
    for n in (10,20):
        y=np.array([1.])
        for j in range(n): y=method.step(lambda t,y:-y,j/n,y,1/n)
        errors.append(abs(y[0]-np.exp(-1)))
    assert 2**(order-.2)<errors[0]/errors[1]<2**(order+.2)

def test_equilibrium_and_snapshot_ownership():
    g=Grid(9,7)
    y=np.stack([np.ones(g.shape),np.zeros(g.shape)])
    out=list(simulate(GrayScott(g,PeriodicFiniteDifference()),y,steps=5,save_every=2))
    assert [t for t,s in out]==[0,1,2,2.5]
    for t,s in out: np.testing.assert_array_equal(s,y)
    out[0][1][0,0,0]=99
    assert out[1][1][0,0,0]==y[0,0,0]==1

def test_injected_components():
    class ZeroSpatial:
        def laplacian(self,field,grid): return np.zeros_like(field)
    g=Grid(8,6)
    m=GrayScott(g,ZeroSpatial(),Parameters(feed=0,kill=0))
    y=np.stack([np.full(g.shape,.7),np.full(g.shape,.2)])
    r=y[0]*y[1]**2
    np.testing.assert_allclose(m(0,y),np.stack([-r,r]))
    out=list(simulate(m,y,dt=.01,steps=10,stepper=RK4()))
    np.testing.assert_allclose(out[-1][1].sum(axis=0),.9)

def test_nontrivial_run_and_seed():
    g=Grid(32,24,32,24)
    y=initial_state(g)
    out=list(simulate(GrayScott(g,PeriodicFiniteDifference()),y,steps=200))
    assert np.isfinite(out[-1][1]).all()
    assert not np.allclose(y,out[-1][1])
    np.testing.assert_array_equal(y,initial_state(g))

@pytest.mark.parametrize('kw',[dict(dt=0),dict(dt=np.nan),dict(steps=-1),dict(steps=1.5),dict(save_every=0)])
def test_invalid(kw):
    g=Grid()
    with pytest.raises(ValueError): list(simulate(GrayScott(g,PeriodicFiniteDifference()),initial_state(g),**kw))

def test_nonfinite_state():
    g=Grid()
    y=initial_state(g); y[0,0,0]=np.nan
    with pytest.raises(ValueError): list(simulate(GrayScott(g,PeriodicFiniteDifference()),y))

@pytest.mark.parametrize('nx,ny',[(18,15),(17,16)])
def test_fourier_modes(nx,ny):
    g=Grid(nx,ny,5,3)
    y,x=np.meshgrid(np.arange(ny)*g.dy,np.arange(nx)*g.dx,indexing='ij')
    u=np.sin(4*np.pi*x/5)*np.cos(2*np.pi*y/3)
    op=PeriodicFourier()
    np.testing.assert_allclose(op.laplacian(u,g),-((4*np.pi/5)**2+(2*np.pi/3)**2)*u,atol=2e-13)
    assert abs(op.laplacian(u,g).sum())<1e-11
    assert np.max(abs(op.laplacian(np.ones(g.shape),g)))<1e-12

def test_separate_terms():
    g=Grid(12,10)
    m=GrayScott(g,PeriodicFourier())
    y=initial_state(g)
    np.testing.assert_allclose(m(0,y),m.diffusion(0,y)+m.reaction(0,y))

def test_adaptive_heat_accuracy_and_output_independence():
    g=Grid(18,15,2*np.pi,2*np.pi)
    yy,xx=np.meshgrid(np.arange(g.ny)*g.dy,np.arange(g.nx)*g.dx,indexing='ij')
    y=np.stack([np.sin(xx)*np.cos(yy),np.cos(xx)])
    class Heat:
        grid=g
        def __call__(self,t,state):
            return np.stack([PeriodicFourier().laplacian(f,g) for f in state])
    times=[0,.013,.31,1.0]
    loose=list(simulate_adaptive(Heat(),y,output_times=times,rtol=1e-3,atol=1e-6))
    tight=list(simulate_adaptive(Heat(),y,output_times=times,rtol=1e-9,atol=1e-12))
    exact=np.stack([y[0]*np.exp(-2),y[1]*np.exp(-1)])
    assert [t for t,s in tight]==times
    assert np.max(abs(tight[-1][1]-exact))<1e-8
    assert np.max(abs(tight[-1][1]-exact))<np.max(abs(loose[-1][1]-exact))
    subset=list(simulate_adaptive(Heat(),y,output_times=[0,1],rtol=1e-9,atol=1e-12))
    np.testing.assert_array_equal(tight[-1][1],subset[-1][1])
    tight[0][1][:]=99
    np.testing.assert_array_equal(y[0],np.sin(xx)*np.cos(yy))

@pytest.mark.parametrize('kw',[dict(output_times=[]),dict(output_times=[1,2]),
    dict(output_times=[0,0]),dict(output_times=[0,np.nan]),
    dict(output_times=[0,1],rtol=0),dict(output_times=[0,1],max_step=-1)])
def test_adaptive_invalid(kw):
    g=Grid(8,8)
    with pytest.raises(ValueError):
        list(simulate_adaptive(GrayScott(g,PeriodicFourier()),initial_state(g),**kw))

def test_adaptive_reaction_diffusion_against_rk4():
    g=Grid(12,10,12,10)
    m=GrayScott(g,PeriodicFourier())
    y=initial_state(g,width=.4)
    a=list(simulate_adaptive(m,y,output_times=[0,.2],rtol=1e-9,atol=1e-12))[-1][1]
    b=list(simulate(m,y,steps=200,dt=.001,stepper=RK4()))[-1][1]
    np.testing.assert_allclose(a,b,atol=2e-9,rtol=2e-9)
