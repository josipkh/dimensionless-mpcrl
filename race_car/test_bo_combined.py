"""Test the results of optimizing the dimensionless parameters (on both cars)"""
import pandas as pd
from race_car.utils.config import get_default_car_params
from race_car.utils.scaling import get_transformation_matrices, get_large_car_params
import numpy as np
from race_car.mpc import test_closed_loop
import matplotlib.pyplot as plt
from race_car.utils.plotting import plot_bo_results

# load the results file
filename = "race_car/output/bo_combined.csv"
df = pd.read_csv(filename)

# get the best parameters (minimum dimensionless lap time)
best_row = df.loc[df['result'].idxmin()]
parameter_columns = [col for col in df.columns if col not in ['car_size', 'result']]
best_params = best_row[parameter_columns]

# scale the parameters to the full size vehicle
# (assumed parameter order: ['q_s', 'q_n', 'q_alpha', 'qe_s', 'qe_n', 'qe_alpha'])
for car_size in ["small", "large"]:
    car_params = get_default_car_params() if car_size == "small" else get_large_car_params()
    Mx = get_transformation_matrices(car_params)[0]

    Q_dimensionless = np.diag(np.array(best_params[:3], dtype=float))
    Qe_dimensionless = np.diag(np.array(best_params[3:], dtype=float))
    Mx_inv = np.linalg.inv(Mx[:3, :3])  # convert just the first three weights
    Q = Mx_inv.T @ Q_dimensionless @ Mx_inv
    Qe = Mx_inv.T @ Qe_dimensionless @ Mx_inv

    # assign the dimensional weights to the car parameters
    car_params.q_diag_sqrt[:3] = np.sqrt(np.array(np.diag(Q)))
    car_params.qe_diag_sqrt[:3] = np.sqrt(np.diag(Qe))

    # run a closed-loop experiment to verify the transfer
    test_closed_loop(car_params=car_params, mpc_car_params=car_params, dimensionless=True, show_plots=True, stop_on_fail=False)
    plt.savefig(f"race_car/output/track_bo_combined_{car_size}.pdf")

    # plot also the BO results (in log scale)
    plot_bo_results(input_file=filename, output_file="race_car/output/bo_combined_log.pdf")

    # keep the plots open (if there are any)
    if plt.get_fignums():
        plt.show(block=False)
        input("Press Enter to continue...")
        plt.close('all')

