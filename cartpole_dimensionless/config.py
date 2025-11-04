from dataclasses import dataclass
import numpy as np
import gymnasium as gym
from leap_c.ocp.acados.parameters import AcadosParameter


@dataclass(kw_only=True)
class CartPoleParams:
    # Dynamics parameters
    M: np.ndarray         # mass of the cart [kg]
    m: np.ndarray         # mass of the ball [kg]
    g: np.ndarray         # gravity constant [m/s^2]
    l: np.ndarray         # length of the pole [m]
    mu_f: np.ndarray      # friction coefficient [kg/s]

    # Cost matrix factorization parameters, W = L @ L.T
    q_diag_sqrt: np.ndarray     # stage cost, state residuals
    r_diag_sqrt: np.ndarray     # stage cost, input residuals

    # Reference parameters (for NONLINEAR_LS cost)
    xref0: np.ndarray     # reference position
    xref1: np.ndarray     # reference theta
    xref2: np.ndarray     # reference v
    xref3: np.ndarray     # reference thetadot
    uref: np.ndarray      # reference u

    # Controller parameters
    Fmax: np.ndarray      # maximum force applied to the cart [N]
    dt: np.ndarray        # time step [s]
    gamma: np.ndarray     # discount factor for the cost function
    N: np.ndarray         # prediction horizon (number of steps)


def get_default_cartpole_params() -> CartPoleParams:
    """Parameter values in the original leap-c example."""
    return CartPoleParams(
        M=np.array([1.0]),
        m=np.array([0.1]),
        g=np.array([9.81]),
        l=np.array([0.8]),
        mu_f=np.array([1.0]),

        q_diag_sqrt=np.sqrt(np.array([2e3, 2e3, 1e-2, 1e-2])),
        r_diag_sqrt=np.sqrt(np.array([2e-1])),

        xref0=np.array([0.0]),
        xref1=np.array([0.0]),
        xref2=np.array([0.0]),
        xref3=np.array([0.0]),
        uref=np.array([0.0]),

        Fmax=np.array([80.0]),
        dt=np.array([0.05]),
        gamma=np.array([1.0]),
        N=np.array([5]),
    )


def create_acados_params(cartpole_params: CartPoleParams) -> list[AcadosParameter]:
    return [
        # Cost matrix factorization parameters
        AcadosParameter(
            "q_diag_sqrt", default=cartpole_params.q_diag_sqrt
        ),  # cost weights of state residuals
        AcadosParameter(
            "r_diag_sqrt", default=cartpole_params.r_diag_sqrt
        ),  # cost weights of control input residuals
        # Reference parameters
        AcadosParameter(
            "xref0",
            default=cartpole_params.xref0,
            interface="non-learnable",
        ),  # reference position
        AcadosParameter(
            "xref1",
            default=cartpole_params.xref1,
            space=gym.spaces.Box(
                low=np.array([-2.0 * np.pi]),
                high=np.array([2.0 * np.pi]),
                dtype=np.float64,
            ),
            interface="learnable",
        ),  # reference theta
        AcadosParameter(
            "xref2",
            default=cartpole_params.xref2,
            interface="non-learnable",
        ),  # reference v
        AcadosParameter(
            "xref3",
            default=cartpole_params.xref3,
            interface="non-learnable",
        ),  # reference thetadot
        AcadosParameter(
            "uref",
            default=cartpole_params.uref,
            interface="non-learnable",
        ),  # reference u
    ]


if __name__ == "__main__":
    params = get_default_cartpole_params()
    acados_params = create_acados_params(params)
    for p in acados_params:
        print(p.name, p.default)
