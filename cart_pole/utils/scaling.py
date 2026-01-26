import numpy as np
from copy import deepcopy
from cart_pole.utils.config import CartPoleParams


def get_transformation_matrices(
    cartpole_params: CartPoleParams,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns the matrices for transforming the system to a non-dimensional form."""
    l = cartpole_params.l.item()  # length of the pole
    m = cartpole_params.M.item()  # mass of the cart
    g = cartpole_params.g.item()  # gravity constant

    Mx = np.diag([l, 1.0, np.sqrt(g * l), np.sqrt(g / l)])
    Mu = np.diag([m * g])
    Mt = np.diag([np.sqrt(l / g)])

    return Mx, Mu, Mt


def get_similar_cartpole_params(
    reference_params: CartPoleParams, pole_length: float
) -> CartPoleParams:
    """Returns the parameters of a cartpole system (MDP) dynamically similar to the reference one.

    Dynamic matching is based on eq. (28) in this paper: https://ieeexplore.ieee.org/document/10178119
    which contains five parameters: pole length, cart mass, pole mass, cart friction and gravity.

    It is assumed that the friction and gravity cannot be changed.
    """
    Mx, Mu, _ = get_transformation_matrices(reference_params)

    new_params = deepcopy(reference_params)
    new_params.l = np.array([pole_length])

    # match the pi-groups
    new_params.M = np.array(
        [
            reference_params.M.item()
            * np.sqrt(new_params.l.item() / reference_params.l.item())
        ]
    )  # keep relative friction
    new_params.m = reference_params.m * (
        new_params.M / reference_params.M
    )  # mass ratio

    # check the pi-groups
    pi_1_ref, pi_2_ref = get_pi_groups(reference_params)
    pi_1_sim, pi_2_sim = get_pi_groups(new_params)
    assert np.allclose(pi_1_ref, pi_1_sim), "Pi-group 1 mismatch"
    assert np.allclose(pi_2_ref, pi_2_sim), "Pi-group 2 mismatch"    

    # match the cost matrices (just Q and R for now)
    Q, R = get_cost_matrices(reference_params)
    mx, mu, _ = get_transformation_matrices(new_params)
    M = Mx @ np.linalg.inv(mx)
    q_diag = (M.T @ Q @ M).diagonal()
    M = Mu @ np.linalg.inv(mu)
    r_diag = (M.T @ R @ M).diagonal()

    for k in range(5):
        new_params.__setattr__(
            f"L{k + 1}{k + 1}",
            np.array([np.sqrt(q_diag[k] if k < 4 else r_diag[k - 4])]),
        )


    # check the matrices
    q, r = get_cost_matrices(new_params)
    assert np.allclose(Mx @ Q @ Mx, mx @ q @ mx)
    assert np.allclose(Mu @ R @ Mu, mu @ r @ mu)

    # match the input constraint
    new_params.Fmax = reference_params.Fmax * (new_params.M / reference_params.M)

    # match the sampling time
    new_params.dt = reference_params.dt * np.sqrt(new_params.l / reference_params.l)

    # the discount factor (already dimensionless) stays the same

    return new_params


def get_cost_matrices(cartpole_params: CartPoleParams) -> tuple[np.ndarray, np.ndarray]:
    """Returns the cost matrices Q and R for the given cartpole system."""
    Q = np.diag(
        [
            cartpole_params.L11.item() ** 2,
            cartpole_params.L22.item() ** 2,
            cartpole_params.L33.item() ** 2,
            cartpole_params.L44.item() ** 2,
        ]
    )
    R = np.diag([cartpole_params.L55.item() ** 2])
    return Q, R


def get_pi_groups(cartpole_params: CartPoleParams) -> tuple[float, float]:
    """Returns the pi-groups for the given cartpole system."""
    M = cartpole_params.M.item()
    m = cartpole_params.m.item()
    l = cartpole_params.l.item()
    mu_f = cartpole_params.mu_f.item()
    g = cartpole_params.g.item()

    pi_1 = m / M
    pi_2 = mu_f / M * np.sqrt(l / g)

    return pi_1, pi_2


if __name__ == "__main__":
    from cart_pole.utils.config import get_default_cartpole_params

    params_ref = get_default_cartpole_params()
    Mx, Mu, Mt = get_transformation_matrices(params_ref)
    params_sim = get_similar_cartpole_params(
        reference_params=params_ref , pole_length=0.1
    )

    # check pi groups
    pi_1_ref, pi_2_ref = get_pi_groups(params_ref)
    pi_1_sim, pi_2_sim = get_pi_groups(params_sim)
    assert np.allclose(pi_1_ref, pi_1_sim), "Pi-group 1 mismatch"
    assert np.allclose(pi_2_ref, pi_2_sim), "Pi-group 2 mismatch"

    # check the cost matrices
    Q, R = get_cost_matrices(params_ref)
    q, r = get_cost_matrices(params_sim)
    Mx, Mu, Mt = get_transformation_matrices(params_ref)
    mx, mu, mt = get_transformation_matrices(params_sim)
    assert np.allclose(Mx @ Q @ Mx, mx @ q @ mx)
    assert np.allclose(Mu @ R @ Mu, mu @ r @ mu)

    # check the sampling time
    assert np.allclose(params_ref.dt / Mt, params_sim.dt / mt)
    print("ok")
