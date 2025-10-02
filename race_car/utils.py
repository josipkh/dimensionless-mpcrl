import numpy as np
from config import CarParams
import os
from pathlib import Path
from copy import deepcopy
from config import get_default_car_params
import matplotlib.pyplot as plt


def get_cost_matrices(car_params: CarParams) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns the cost matrices Q, R and Q_e for the given race car."""
    Q = np.diag(
        [
            car_params.L11.item() ** 2,
            car_params.L22.item() ** 2,
            car_params.L33.item() ** 2,
            car_params.L44.item() ** 2,
            car_params.L55.item() ** 2,
            car_params.L66.item() ** 2,
        ]
    )

    R = np.diag([car_params.L77.item() ** 2, car_params.L88.item() ** 2])

    Q_e = np.diag(
        [
            car_params.LN11.item() ** 2,
            car_params.LN22.item() ** 2,
            car_params.LN33.item() ** 2,
            car_params.LN44.item() ** 2,
            car_params.LN55.item() ** 2,
            car_params.LN66.item() ** 2,
        ]
    )

    return Q, R, Q_e


def get_transformation_matrices(
    car_params: CarParams,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns the matrices for transforming the system to a non-dimensional form."""
    # s(physical) = Ms * s(dimensionless)
    l = car_params.l.item()  # length of the pole
    cr3 = car_params.cr3.item()  # [s/m] speed coefficient

    Mx = np.diag([l, l, 1.0, 1/cr3, 1.0, 1.0])  # states include the control inputs
    Mu = np.diag([1.0 / (l * cr3), 1.0 / (l * cr3)])  # inputs are rate of change of controls
    Mt = np.diag([cr3 * l])

    return Mx, Mu, Mt


def get_similar_car_params(
        reference_params: CarParams,
        new_length: float,
        new_mass: float,
        new_cr3: float
) -> CarParams:
    """Returns the parameters of a car dynamically similar to the reference one."""

    new_params = deepcopy(reference_params)
    new_params.l = np.array([new_length])
    new_params.m = np.array([new_mass])
    new_params.cr3 = np.array([new_cr3])

    # match the pi-groups
    pi_groups_ref = get_pi_groups(reference_params)
    new_params.lr = np.array([pi_groups_ref[0] * new_length])
    new_params.cm1 = np.array([pi_groups_ref[1] * new_mass / (new_length * new_cr3**2)])
    new_params.cm2 = np.array([pi_groups_ref[2] * new_mass / (new_cr3 * new_length)])
    new_params.cr0 = np.array([pi_groups_ref[3] * new_mass / (new_length * new_cr3**2)])
    new_params.cr2 = np.array([pi_groups_ref[4] * new_mass / new_length])

    # match the MPC cost matrices
    Q, R, Q_e = get_cost_matrices(reference_params)
    Mx, Mu, Mt = get_transformation_matrices(reference_params)
    mx, mu, mt = get_transformation_matrices(new_params)

    M = Mx @ np.linalg.inv(mx)
    q_diag = (M.T @ Q @ M).diagonal()
    qe_diag = (M.T @ Q_e @ M).diagonal()

    M = Mu @ np.linalg.inv(mu)
    r_diag = (M.T @ R @ M).diagonal()

    for k in range(8):
        new_params.__setattr__(
            f"L{k + 1}{k + 1}",
            np.array([np.sqrt(q_diag[k] if k < 6 else r_diag[k - 6])]),
        )
        if k < 6:
            new_params.__setattr__(f"LN{k + 1}{k + 1}", np.array([np.sqrt(qe_diag[k])]))

    # match the constraints
    # D_max, D_min, delta_max, delta_min are not changed (they are dimensionless)
    new_params.dD_max = reference_params.dD_max * Mt / mt
    new_params.dD_min = reference_params.dD_min * Mt / mt
    new_params.ddelta_max = reference_params.ddelta_max * Mt / mt
    new_params.ddelta_min = reference_params.ddelta_min * Mt / mt

    MtCr3 = Mt * reference_params.cr3.item()
    mtcr3 = mt * new_params.cr3.item()
    new_params.a_lat_max = reference_params.a_lat_max * MtCr3 / mtcr3
    new_params.a_lat_min = reference_params.a_lat_min * MtCr3 / mtcr3
    new_params.a_long_max = reference_params.a_long_max * MtCr3 / mtcr3
    new_params.a_long_min = reference_params.a_long_min * MtCr3 / mtcr3
    new_params.n_max = reference_params.n_max * (new_length / reference_params.l.item())
    new_params.n_min = reference_params.n_min * (new_length / reference_params.l.item())

    # match the sampling time
    new_params.dt = reference_params.dt * (mt / Mt).item()

    # the discount factor (already dimensionless) stays the same

    return new_params


def get_pi_groups(car_params: CarParams) -> tuple[float, float]:
    """Returns the pi-groups for the given car."""
    m = car_params.m.item()
    l = car_params.l.item()
    cr3 = car_params.cr3.item()

    pi_1 = car_params.lr.item() / l
    pi_2 = car_params.cm1.item() * l * cr3**2 / m
    pi_3 = car_params.cm2.item() * l * cr3 / m
    pi_4 = car_params.cr0.item() * l * cr3**2 / m
    pi_5 = car_params.cr2.item() * l / m

    return np.array([pi_1, pi_2, pi_3, pi_4, pi_5])


def get_track(car_params: CarParams) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Loads the track data and scales it with the car size."""
    track = "track.txt"  # not likely to change, so hardcoded here
    track_file = os.path.join(str(Path(__file__).parent), track)
    array = np.loadtxt(track_file)
    sref = array[:,0]  # [m]
    xref = array[:,1]  # [m]
    yref = array[:,2]  # [m]
    psiref = array[:,3]  # [rad]
    kapparef = array[:,4]  # [1/m]

    # the track is defined for the small car
    # we need to scale it with the given car size
    default_car_length = get_default_car_params().l.item()
    current_car_length = car_params.l.item()
    scaling_factor = current_car_length / default_car_length
    sref *= scaling_factor
    xref *= scaling_factor
    yref *= scaling_factor
    kapparef /= scaling_factor

    return sref, xref, yref, psiref, kapparef


def plot_track(car_params: CarParams):
    """Plots the track for the given car parameters."""
    import matplotlib.pyplot as plt

    _, xref, yref, psiref, _ = get_track(car_params=car_params)

    plt.figure(figsize=(10, 8))
    plt.plot(xref, yref, label='Track Centerline', color='blue')
    step = 20
    plt.quiver(
        xref[::step],
        yref[::step],
        np.cos(psiref[::step]),
        np.sin(psiref[::step]),
        angles='xy',
        scale_units='xy',
        scale=5*get_default_car_params().l.item()/car_params.l.item(),
        color='red',
        width=0.003,
        label='Heading'
    )
    plt.axis('equal')
    plt.title('Track Centerline with Heading Arrows')
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show(block=False)


def compare_params(car1: CarParams, car2: CarParams):
    # Define parameter groups by field names
    chassis_fields = ['m', 'l', 'lr']
    force_fields = ['cm1', 'cm2', 'cr0', 'cr2', 'cr3']
    controller_fields = ['dt', 'gamma']

    # Helper to print groups side by side
    def print_group(group_name, group_fields):
        print(f"\n--- {group_name} Parameters ---")
        print(f"{'Parameter':<15} | {'Car 1':<20} | {'Car 2':<20}")
        print("-" * 60)
        for name in group_fields:
            val1 = getattr(car1, name)[0]
            val2 = getattr(car2, name)[0]
            # Convert numpy arrays to string, limit length for readability
            str_val1 = np.array2string(val1, precision=4, suppress_small=True)
            str_val2 = np.array2string(val2, precision=4, suppress_small=True)
            print(f"{name:<15} | {str_val1:<20} | {str_val2:<20}")

    print_group("Chassis", chassis_fields)
    print_group("Longitudinal Force", force_fields)
    print_group("Controller", controller_fields)
    print('\n')

    return


def plot_Fxd_surface(car_params: CarParams):
    """
    Plots the Fxd curve over speed (v) and throttle (D).

    Parameters:
        cm1, cm2, cr2, cr0, cr3: car parameters
        v_range: tuple (min_v, max_v)
        D_range: tuple (min_D, max_D)
        resolution: number of points for meshgrid
    """
    default_v_max = 2.5  # [m/s] max speed for the small car
    default_cr3 = get_default_car_params().cr3.item()
    v_max = default_v_max * default_cr3 / car_params.cr3.item()
    v_range = (0, v_max)
    D_range = (0, 1)
    resolution = 100

    # Create meshgrid of v and D
    v = np.linspace(*v_range, resolution)
    D = np.linspace(*D_range, resolution)
    V, D_grid = np.meshgrid(v, D)

    # Compute Fxd over the grid
    cm1 = car_params.cm1.item()
    cm2 = car_params.cm2.item()
    cr2 = car_params.cr2.item()
    cr0 = car_params.cr0.item()
    cr3 = car_params.cr3.item()
    Fxd = (cm1 - cm2 * V) * D_grid - cr2 * V**2 - cr0 * np.tanh(cr3 * V)

    # Plotting
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(V, D_grid, Fxd, cmap='viridis')
    ax.set_xlabel('Speed v [m/s]')
    ax.set_ylabel('Throttle D [0-1]')
    ax.set_zlabel('Fxd [N]')
    ax.set_title('Fxd vs Speed and Throttle (l = {:.2f} m)'.format(car_params.l.item()))
    plt.tight_layout()
    plt.show(block=False)

    return


def plot_Fxd_vs_D_slices(car_params: CarParams, num_slices: int = 10):
    """
    Plots Fxd vs Throttle (D) for different fixed speeds (v).
    """
    default_v_max = 2.5  # [m/s]
    default_cr3 = get_default_car_params().cr3.item()
    v_max = default_v_max * default_cr3 / car_params.cr3.item()

    cm1 = car_params.cm1.item()
    cm2 = car_params.cm2.item()
    cr2 = car_params.cr2.item()
    cr0 = car_params.cr0.item()
    cr3 = car_params.cr3.item()

    D = np.linspace(0, 1, 200)

    # Choose speeds for slicing (linearly spaced from 0 to v_max)
    v_slices = np.linspace(0, v_max, num_slices)

    plt.figure(figsize=(10, 6))

    for v in v_slices:
        Fxd = (cm1 - cm2 * v) * D - cr2 * v**2 - cr0 * np.tanh(cr3 * v)
        label = f"v = {v:.2f} m/s"
        plt.plot(D, Fxd, label=label)

    plt.xlabel("Throttle D [0-1]")
    plt.ylabel("Fxd [N]")
    plt.title("Fxd vs Throttle for Different Speeds (l = {:.2f} m)".format(car_params.l.item()))
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show(block=False)


if __name__ == "__main__":
    from config import get_default_car_params

    car_params = get_default_car_params()
    car_params_sim = get_similar_car_params(
        reference_params=car_params,
        new_length=4.0,
        new_mass=1500.0,
        new_cr3=0.4,
    )
    
    # check dynamic similarity
    assert np.allclose(get_pi_groups(car_params), get_pi_groups(car_params_sim))

    # check the track scaling
    track_data = get_track(car_params=car_params)
    track_data_sim = get_track(car_params=car_params_sim)
    l_ref = car_params.l.item()
    l_sim = car_params_sim.l.item()
    assert np.allclose(track_data[0] / l_ref, track_data_sim[0] / l_sim)  # s
    assert np.allclose(track_data[1] / l_ref, track_data_sim[1] / l_sim)  # x
    assert np.allclose(track_data[2] / l_ref, track_data_sim[2] / l_sim)  # y
    assert np.allclose(track_data[3], track_data_sim[3])  # psi
    assert np.allclose(track_data[4] * l_ref, track_data_sim[4] * l_sim)  # kappa

    # check the cost matrices
    Q, R, Q_e = get_cost_matrices(car_params)
    q, r, q_e = get_cost_matrices(car_params_sim)
    Mx, Mu, Mt = get_transformation_matrices(car_params)
    mx, mu, mt = get_transformation_matrices(car_params_sim)
    assert np.allclose(Mx @ Q @ Mx, mx @ q @ mx)
    assert np.allclose(Mu @ R @ Mu, mu @ r @ mu)
    assert np.allclose(Mx @ Q_e @ Mx, mx @ q_e @ mx)

    # check the sampling time
    assert np.allclose(car_params.dt / Mt, car_params_sim.dt / mt)

    compare_params(car_params, car_params_sim)
    plot_track(car_params=car_params_sim)

    # plot_Fxd_surface(car_params=car_params)
    # plot_Fxd_surface(car_params=car_params_sim)

    # plot_Fxd_vs_D_slices(car_params=car_params)
    # plot_Fxd_vs_D_slices(car_params=car_params_sim)

    print("All checks passed.")
    if plt.get_fignums():
        input("Press Enter to continue...")  # keep the plots open
        plt.close('all')