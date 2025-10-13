"""Tune the parameters of the MPC using Bayesian optimization through Optuna."""
import optuna
from race_car.acados_ocp import test_closed_loop
from race_car.utils.config import CarParams, get_default_car_params
import numpy as np

def objective(trial: optuna.Trial, car_params: CarParams, dimensionless: bool) -> float:

    q_n = trial.suggest_float('q_n', 1e-8, 1, log=True)
    q_alpha = trial.suggest_float('q_alpha', 1e-8, 1, log=True)
    car_params.q_diag_sqrt[1] = np.sqrt(q_n)
    car_params.q_diag_sqrt[2] = np.sqrt(q_alpha)

    qe_s = trial.suggest_float('qe_s', 0.1, 5 + 1)  # offset added to prevent warnings due to rounding errors
    qe_n = trial.suggest_float('qe_n', 1e-2, 1e6, log=True)
    qe_alpha = trial.suggest_float('qe_alpha', 1e-8, 1, log=True)
    car_params.qe_diag_sqrt[0] = np.sqrt(qe_s)
    car_params.qe_diag_sqrt[1] = np.sqrt(qe_n)
    car_params.qe_diag_sqrt[2] = np.sqrt(qe_alpha)
    
    return test_closed_loop(car_params=car_params, dimensionless=dimensionless)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    car_params = get_default_car_params()
    dimensionless = False

    # define the initial guess for the parameters (optional)
    mode = "track"  # "race" or "track"
    if mode == "track":
        car_params.q_diag_sqrt[0] = np.sqrt(1e-1)
        car_params.q_diag_sqrt[1] = np.sqrt(1)
        car_params.q_diag_sqrt[2] = np.sqrt(1)

        car_params.qe_diag_sqrt[0] = np.sqrt(1e-1)
        car_params.qe_diag_sqrt[1] = np.sqrt(1e6)
        car_params.qe_diag_sqrt[2] = np.sqrt(1)

    # run a classic closed-loop test
    # test_closed_loop(car_params=car_params, dimensionless=False)

    sampler = optuna.samplers.TPESampler(seed=0)  # make the sampler behave in a deterministic way
    study = optuna.create_study(sampler=sampler)

    # set the first Optuna trial to use the specified parameters
    study.enqueue_trial({
        "q_n": car_params.q_diag_sqrt[1] ** 2,
        "q_alpha": car_params.q_diag_sqrt[2] ** 2,
        "qe_s": car_params.qe_diag_sqrt[0] ** 2,
        "qe_n": car_params.qe_diag_sqrt[1] ** 2,
        "qe_alpha": car_params.qe_diag_sqrt[2] ** 2,
    })

    # run the HPO
    study.optimize(
        lambda trial: objective(trial, car_params=car_params, dimensionless=dimensionless), 
        n_trials=5)
    
    # print the best parameters
    study.best_params

    if plt.get_fignums():
        input("Press Enter to continue...")  # keep the plots open
        plt.close('all')
    