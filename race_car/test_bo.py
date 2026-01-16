"""Test the transfer of dimensional MPC parameters (small -> large) found with Optuna"""
import pandas as pd
from race_car.utils.config import get_default_car_params
from race_car.utils.scaling import get_transformation_matrices, get_large_car_params, get_cost_matrices
import numpy as np
from race_car.acados_ocp import test_closed_loop
import matplotlib.pyplot as plt

# load the results file
filename = "race_car/output/bo_small.csv"
df = pd.read_csv(filename)

# get the best parameters (minimum lap time)
best_row = df.loc[df['result'].idxmin()]
parameter_columns = [col for col in df.columns if col != 'result']
best_params_small = best_row[parameter_columns]

# scale the parameters to the full size vehicle
# (assumed parameter order: ['q_s', 'q_n', 'q_alpha', 'qe_s', 'qe_n', 'qe_alpha'])
car_params_small = get_default_car_params()
car_params_large = get_large_car_params()

car_params_small.q_diag_sqrt[:3] = np.sqrt(np.array(best_params_small[:3]))
car_params_small.qe_diag_sqrt[:3] = np.sqrt(np.array(best_params_small[3:]))
q_small, _, qe_small = get_cost_matrices(car_params=car_params_small)

mx_small = get_transformation_matrices(car_params=car_params_small)[0]
mx_large = get_transformation_matrices(car_params=car_params_large)[0]

# convert the cost by matching the dimensionless matrices
m = mx_small @ np.linalg.inv(mx_large)
q_large = m.T @ q_small @ m
qe_large = m.T @ qe_small @ m

# set the calculated cost weights in the parameters
car_params_large.q_diag_sqrt = np.sqrt(np.diag(q_large))
car_params_large.qe_diag_sqrt = np.sqrt(np.diag(qe_large))

# run a closed-loop experiment to verify the transfer
test_closed_loop(
    car_params=car_params_large,
    mpc_car_params=car_params_large,
    dimensionless=False,
    show_plots=True,
    stop_on_fail=False
)

# maximize and save the figure (for better resolution)
manager = plt.get_current_fig_manager()
manager.full_screen_toggle()
plt.savefig("race_car/output/track_large_transfer.pdf", bbox_inches='tight')
if plt.get_fignums():
    plt.show(block=False)
    input("Press Enter to continue...")
    plt.close('all')

