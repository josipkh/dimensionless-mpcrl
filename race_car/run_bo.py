"""Tune the parameters of the MPC using Bayesian optimization through Optuna."""
import optuna
from race_car.acados_ocp import test_closed_loop
from race_car.utils.config import CarParams
import numpy as np
import csv
import os
from datetime import datetime
from optuna.visualization.matplotlib import plot_optimization_history
import matplotlib.ticker as ticker


def objective(trial: optuna.Trial, car_params: CarParams, mpc_car_params: CarParams, dimensionless: bool, show_plots: bool) -> float:
    """Defines an experiment for Optuna, returns the lap time [s]"""
    # use generic limits for cost weights
    q_min = 1e-6
    q_max = 1e+6

    # parametrize the cost on the first three states (input cost fixed)
    q_s = trial.suggest_float('q_s', q_min, q_max, log=True)
    q_n = trial.suggest_float('q_n', q_min, q_max, log=True)
    q_alpha = trial.suggest_float('q_alpha', q_min, q_max, log=True)
    car_params.q_diag_sqrt[0] = np.sqrt(q_s)
    car_params.q_diag_sqrt[1] = np.sqrt(q_n)
    car_params.q_diag_sqrt[2] = np.sqrt(q_alpha)

    # independent parameters in the terminal cost
    qe_s = trial.suggest_float('qe_s', q_min, q_max, log=True)
    qe_n = trial.suggest_float('qe_n', q_min, q_max, log=True)
    qe_alpha = trial.suggest_float('qe_alpha', q_min, q_max, log=True)
    car_params.qe_diag_sqrt[0] = np.sqrt(qe_s)
    car_params.qe_diag_sqrt[1] = np.sqrt(qe_n)
    car_params.qe_diag_sqrt[2] = np.sqrt(qe_alpha)
    
    return test_closed_loop(car_params, mpc_car_params, dimensionless, show_plots)


def save_results(study: optuna.Study) -> None:
    """Saves the results (parameters and lap time from each trial) as a .csv file"""
    # get the results from each trial
    runs = [
        {'params': trial.params, 'result': trial.values[0]} for trial in study.trials
    ]
    fieldnames = list(runs[0]['params'].keys()) + ['result']

    # create a new file for logging
    output_dir = 'race_car/output'
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filepath = os.path.join(output_dir, f'log_{timestamp}.csv')

    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for run in runs:
            row = {**run['params'], 'result': run['result']}
            writer.writerow(row)

    # save a plot of the results as well
    ax = plot_optimization_history(study)  

    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))  # integer ticks
    ax.yaxis.set_label_text('Lap time [s]')
    ax.set_title("")  # removes the title
    ax.grid(True)
    legend = ax.get_legend()
    if legend:
        legend.remove()

    plt.savefig(filepath[:-3] + "pdf")

    print(f"Results saved in {filepath}")
    return


if __name__ == "__main__":
    from race_car.utils.config import get_default_car_params
    from race_car.utils.scaling import get_large_car_params
    import matplotlib.pyplot as plt
    
    car_params = get_default_car_params()
    mpc_car_params = car_params  # no mismatch
    dimensionless = False
    show_plots = False

    # run a classic closed-loop test
    # test_closed_loop(car_params, mpc_car_params, dimensionless, show_plots)

    # define the initial guess for the parameters (optional)
    init_mode = None  # "race", "track", "best" or None
    skip_init = False
    match init_mode:
        case "track":
            car_params.q_diag_sqrt[0] = np.sqrt(1e-1)
            car_params.q_diag_sqrt[1] = np.sqrt(1)
            car_params.q_diag_sqrt[2] = np.sqrt(1)

            car_params.qe_diag_sqrt[0] = np.sqrt(1e-1)
            car_params.qe_diag_sqrt[1] = np.sqrt(1e6)
            car_params.qe_diag_sqrt[2] = np.sqrt(1)
        case "best":
            car_params.q_diag_sqrt[0] = np.sqrt(1e-1)
            car_params.q_diag_sqrt[1] = np.sqrt(56.3)
            car_params.q_diag_sqrt[2] = np.sqrt(0.18)

            car_params.qe_diag_sqrt[0] = np.sqrt(50266)
            car_params.qe_diag_sqrt[1] = np.sqrt(366398)
            car_params.qe_diag_sqrt[2] = np.sqrt(0.04)
        case None | "race":
            skip_init = True  # optimize without any priors

    # fix the sampler seed for reproducibility
    sampler = optuna.samplers.GPSampler(seed=0)
    study = optuna.create_study(sampler=sampler)

    # set the first Optuna trial to use the specified parameters
    if not skip_init:
        study.enqueue_trial({
            "q_s": car_params.q_diag_sqrt[0] ** 2,
            "q_n": car_params.q_diag_sqrt[1] ** 2,
            "q_alpha": car_params.q_diag_sqrt[2] ** 2,
            "qe_s": car_params.qe_diag_sqrt[0] ** 2,
            "qe_n": car_params.qe_diag_sqrt[1] ** 2,
            "qe_alpha": car_params.qe_diag_sqrt[2] ** 2,
        })

    # run the parameter optimization
    study.optimize(
        lambda trial: objective(trial, car_params, mpc_car_params, dimensionless, show_plots), 
        n_trials=50
    )
    
    # print the best parameters and save the results
    print("Best parameters:")
    print(study.best_params)
    save_results(study)

    # keep the plots open (if there are any)
    if plt.get_fignums():
        plt.show()
        input("Press Enter to continue...")
        plt.close('all')
    