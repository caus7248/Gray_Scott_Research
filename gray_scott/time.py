"""Time steppers consume RHS(t, state), with no spatial-method assumptions."""
from typing import Callable, Protocol
import numpy as np
RHS = Callable[[float, np.ndarray], np.ndarray]

class TimeStepper(Protocol):
    def step(self, rhs: RHS, t: float, state: np.ndarray, dt: float) -> np.ndarray: ...

class ForwardEuler:
    def step(self, rhs, t, state, dt):
        return state + dt * rhs(t, state)

class RK4:
    """Classical explicit fourth-order Runge-Kutta."""
    def step(self, rhs, t, state, dt):
        k1 = rhs(t, state)
        k2 = rhs(t+dt/2, state+dt*k1/2)
        k3 = rhs(t+dt/2, state+dt*k2/2)
        k4 = rhs(t+dt, state+dt*k3)
        return state + dt*(k1+2*k2+2*k3+k4)/6
