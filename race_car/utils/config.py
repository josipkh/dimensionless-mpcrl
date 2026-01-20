"""Defines the parameters of the race car OCP and their default values."""

from dataclasses import dataclass, asdict
import numpy as np
import gymnasium as gym

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

    # Cost weight matrices, defined by the square roots of their diagonal entries
    q_diag_sqrt: np.ndarray     # stage cost, state residuals
    r_diag_sqrt: np.ndarray     # stage cost, input residuals
    qe_diag_sqrt: np.ndarray    # terminal cost, state residuals

    # Slack cost weights
    slack_n_linear: np.ndarray          # linear cost on the position constraint [1/m]
    slack_n_quadratic: np.ndarray       # quadratic cost on the position constraint [1/m^2]
    slack_acc_linear: np.ndarray        # linear cost on the acceleration constraint [1/(m/s^2)]
    slack_acc_quadratic: np.ndarray     # quadratic cost on the acceleration constraint [1/(m/s^2)^2]

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
        cr0=np.array([0.011]),  # switched with cr2 in the paper, values taken from the code
        cr2=np.array([0.006]),  # see also Table I here: https://cdn.syscop.de/publications/Verschueren2016b.pdf
        cr3=np.array([5.0]),

        q_diag_sqrt=np.sqrt([
            1e-1,
            1e-8,
            1e-8,
            1e-8,
            1e-3,
            5e-3,
        ]),

        r_diag_sqrt=np.sqrt([
            1e-3,
            5e-3,
        ]),

        qe_diag_sqrt=np.sqrt([
            5e0,
            1e2,
            1e-8,
            1e-8,
            1e-3,
            5e-3,
        ]),

        slack_n_linear=np.array([100.0]),
        slack_n_quadratic=np.array([1.0]),
        slack_acc_linear=np.array([100.0]),
        slack_acc_quadratic=np.array([1.0]),

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