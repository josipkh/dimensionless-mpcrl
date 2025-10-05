"""Defines the parameters of the race car OCP and their default values."""

from dataclasses import dataclass
import numpy as np

@dataclass(kw_only=True)
class CarParams:
    # Chassis parameters
    m: np.ndarray         # [kg] mass of the car
    l: np.ndarray         # [m] length of the car
    lr: np.ndarray        # [m] distance from CG to rear axle

    # Longitudinal force parameters (eq. 4)
    cm1: np.ndarray       # [kgm/s^2] 
    cm2: np.ndarray       # [kg/s]
    cr0: np.ndarray       # [kgm/s^2]
    cr2: np.ndarray       # [kg/m]
    cr3: np.ndarray       # [s/m]

    # Stage cost matrix factorization parameters, W = L @ L.T
    L11: np.ndarray
    L22: np.ndarray
    L33: np.ndarray
    L44: np.ndarray
    L55: np.ndarray
    L66: np.ndarray
    L77: np.ndarray
    L88: np.ndarray
    Lloweroffdiag: np.ndarray

    # Terminal cost matrix factorization parameters, WN = LN @ LN.T
    LN11: np.ndarray
    LN22: np.ndarray
    LN33: np.ndarray
    LN44: np.ndarray
    LN55: np.ndarray
    LN66: np.ndarray
    LNloweroffdiag: np.ndarray

    # Linear cost parameters (for EXTERNAL cost)
    c1: np.ndarray
    c2: np.ndarray
    c3: np.ndarray
    c4: np.ndarray
    c5: np.ndarray
    c6: np.ndarray
    c7: np.ndarray
    c8: np.ndarray

    # Stage cost reference parameters (for NONLINEAR_LS cost)
    xref1: np.ndarray
    xref2: np.ndarray
    xref3: np.ndarray
    xref4: np.ndarray
    xref5: np.ndarray
    xref6: np.ndarray
    uref1: np.ndarray
    uref2: np.ndarray

    # Terminal cost reference parameters (for NONLINEAR_LS cost)
    xNref1: np.ndarray
    xNref2: np.ndarray
    xNref3: np.ndarray
    xNref4: np.ndarray
    xNref5: np.ndarray
    xNref6: np.ndarray

    # Constraints
    a_lat_max: np.ndarray   # maximum lateral acceleration [m/s^2]
    a_lat_min: np.ndarray   # minimum lateral acceleration [m/s^2]
    a_long_max: np.ndarray  # maximum longitudinal acceleration [m/s^2]
    a_long_min: np.ndarray  # minimum longitudinal acceleration [m/s^2]
    n_max: np.ndarray       # maximum lateral deviation from center line [m]
    n_min: np.ndarray       # minimum lateral deviation from center line [m]

    D_max: np.ndarray       # maximum duty cycle [-]
    D_min: np.ndarray       # minimum duty cycle [-]
    delta_max: np.ndarray   # maximum steering angle [rad]
    delta_min: np.ndarray   # minimum steering angle [rad]
    dD_max: np.ndarray      # maximum duty cycle rate [1/s]
    dD_min: np.ndarray      # minimum duty cycle rate [1/s]
    ddelta_max: np.ndarray  # maximum steering angle rate [rad/s]
    ddelta_min: np.ndarray  # minimum steering angle rate [rad/s]

    # Controller parameters
    dt: np.ndarray          # time step [s]
    gamma: np.ndarray       # discount factor for the cost function
    N: np.ndarray           # prediction horizon (number of steps)


def get_default_car_params() -> CarParams:
    """Parameter values in the original example."""
    return CarParams(
        m=np.array([0.043]),
        l=np.array([1/15.5]),
        lr=np.array([1/15.5/2]),

        cm1=np.array([0.28]),
        cm2=np.array([0.05]),
        cr0=np.array([0.011]),  # switched with cr2 in the paper
        cr2=np.array([0.006]),  # values taken from the code
        cr3=np.array([5.0]),

        L11=np.array([np.sqrt(1e-1)]),
        L22=np.array([np.sqrt(1e-8)]),
        L33=np.array([np.sqrt(1e-8)]),
        L44=np.array([np.sqrt(1e-8)]),
        L55=np.array([np.sqrt(1e-3)]),
        L66=np.array([np.sqrt(5e-3)]),
        L77=np.array([np.sqrt(1e-3)]),
        L88=np.array([np.sqrt(5e-3)]),
        Lloweroffdiag=np.array([0.0] * sum(range(8))),

        LN11=np.array([np.sqrt(5e0)]),
        LN22=np.array([np.sqrt(1e1)]),
        LN33=np.array([np.sqrt(1e-8)]),
        LN44=np.array([np.sqrt(1e-8)]),
        LN55=np.array([np.sqrt(1e-3)]),
        LN66=np.array([np.sqrt(5e-3)]),
        LNloweroffdiag=np.array([0.0] * sum(range(6))),

        c1=np.array([0.0]),
        c2=np.array([0.0]),
        c3=np.array([0.0]),
        c4=np.array([0.0]),
        c5=np.array([0.0]),
        c6=np.array([0.0]),
        c7=np.array([0.0]),
        c8=np.array([0.0]),

        xref1=np.array([0.0]),
        xref2=np.array([0.0]),
        xref3=np.array([0.0]),
        xref4=np.array([0.0]),
        xref5=np.array([0.0]),
        xref6=np.array([0.0]),
        uref1=np.array([0.0]),
        uref2=np.array([0.0]),

        xNref1=np.array([0.0]),
        xNref2=np.array([0.0]),
        xNref3=np.array([0.0]),
        xNref4=np.array([0.0]),
        xNref5=np.array([0.0]),
        xNref6=np.array([0.0]),

        a_lat_max=np.array([+4.0]),
        a_lat_min=np.array([-4.0]),
        a_long_max=np.array([+4.0]),
        a_long_min=np.array([-4.0]),
        n_max=np.array([+0.12]),
        n_min=np.array([-0.12]),

        D_max=np.array([+1.0]),
        D_min=np.array([-1.0]),
        delta_max=np.array([+0.4]),
        delta_min=np.array([-0.4]),
        dD_max=np.array([+10.0]),
        dD_min=np.array([-10.0]),
        ddelta_max=np.array([+2.0]),
        ddelta_min=np.array([-2.0]),

        dt=np.array([0.02]),
        gamma=np.array([1.0]),
        N=np.array([50]),
    )


if __name__ == "__main__":
    from pprint import pprint
    car_params = get_default_car_params()
    pprint(car_params)