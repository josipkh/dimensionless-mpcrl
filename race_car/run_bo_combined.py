"""This script runs the Bayesian optimization by combining trials from the small and large cars."""
import optuna
from race_car.mpc import test_closed_loop
import numpy as np
import csv
import os
from datetime import datetime
from optuna.visualization.matplotlib import plot_optimization_history
import matplotlib.ticker as ticker
import random
from race_car.utils.config import get_default_car_params
from race_car.utils.scaling import get_large_car_params, get_transformation_matrices


def objective(trial: optuna.Trial, show_plots: bool) -> float:
    """Defines an experiment for Optuna, returns the dimensionless lap time [-]"""
    # randomly select the car size for the experiment
    car_size = random.choice(['small', 'large'])
    car_params = get_default_car_params() if car_size == "small" else get_large_car_params()
    mpc_car_params = car_params  # no mismatch
    dimensionless = True  # this method only works with the dimensionless formulation
    trial.set_user_attr("car_size", car_size)  # keep track of which system was used

    # use generic limits for cost weights
    q_min = 1e-4
    q_max = 1e+4

    # parametrize the cost on the first three states (input cost fixed)
    q_s = trial.suggest_float('q_s', q_min, q_max, log=True)
    q_n = trial.suggest_float('q_n', q_min, q_max, log=True)
    q_alpha = trial.suggest_float('q_alpha', q_min, q_max, log=True)

    # independent parameters in the terminal cost
    qe_s = trial.suggest_float('qe_s', q_min, q_max, log=True)
    qe_n = trial.suggest_float('qe_n', q_min, q_max, log=True)
    qe_alpha = trial.suggest_float('qe_alpha', q_min, q_max, log=True)

    # suggested parameters are already dimensionless, unscale to the dimensional ones
    Mx, _, Mt = get_transformation_matrices(car_params)
    Q_dimensionless = np.diag(np.array([q_s, q_n, q_alpha]))
    Qe_dimensionless = np.diag(np.array([qe_s, qe_n, qe_alpha]))
    Mx_inv = np.linalg.inv(Mx[:3, :3])  # convert just the first three weights
    Q = Mx_inv.T @ Q_dimensionless @ Mx_inv
    Qe = Mx_inv.T @ Qe_dimensionless @ Mx_inv

    # assign the dimensional weights to the car parameters
    car_params.q_diag_sqrt[:3] = np.sqrt(np.diag(Q))
    car_params.qe_diag_sqrt[:3] = np.sqrt(np.diag(Qe))

    # run a closed-loop test
    lap_time = test_closed_loop(car_params, mpc_car_params, dimensionless, show_plots)
    lap_time_dimensionless = lap_time / Mt
    return lap_time_dimensionless


def save_results(study: optuna.Study) -> None:
    """Saves the results (parameters and lap time from each trial) as a .csv file"""
    # get the results from each trial
    runs = [
        {'params': trial.params, 'car_size': trial.user_attrs['car_size'], 'result': trial.values[0]} for trial in study.trials
    ]
    fieldnames = list(runs[0]['params'].keys()) + ['car_size'] + ['result']

    # create a new file for logging
    output_dir = 'race_car/output'
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filepath = os.path.join(output_dir, f'log_{timestamp}.csv')

    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for run in runs:
            row = {**run['params'], 'car_size': run['car_size'], 'result': run['result']}
            writer.writerow(row)

    # get the history of trials
    ax = plot_optimization_history(study)

    # change the line color and label
    ax.get_lines()[0].set_color("black")
    ax.get_lines()[0].set_label('best value')

    # replace the scatter points with colored ones (per vehicle size)
    original_collection = ax.collections[0]
    legend = ax.get_legend()
    if legend:
        legend.remove()

    # https://stackoverflow.com/questions/64369710/what-are-the-hex-codes-of-matplotlib-tab10-palette
    color_map = {
        'small': "#d62728",   # red
        'large': "#1f77b4",   # blue
    }

    # plot one invisible scatter per group for legend handles
    for size, color in color_map.items():
        ax.scatter([], [], c=color, label=size)

    # add the colored points
    points = original_collection.get_offsets()
    x = points[:, 0]
    y = points[:, 1]
    colors = [color_map[run["car_size"]] for run in runs]
    ax.scatter(x, y, c=colors)
    original_collection.remove()
    ax.legend()

    # additional tweaks
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))  # integer ticks
    ax.yaxis.set_label_text('Dimensionless lap time [-]')
    ax.set_title("")  # removes the title
    ax.grid(True)

    plt.savefig(filepath[:-3] + "pdf")

    print(f"Results saved in {filepath}")
    return


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    show_plots = False

    # fix the sampler seed for reproducibility
    sampler = optuna.samplers.TPESampler(seed=0)
    study = optuna.create_study(sampler=sampler)

    # run the parameter optimization
    study.optimize(
        lambda trial: objective(trial, show_plots), 
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
    