This example is based on the pendulum on cart (cart-pole) system, often used as a benchmark.
The full description of the system can be found, e.g., in [this paper](https://ieeexplore.ieee.org/document/10178119).

The goal is to learn how to swing up the pendulum by controlling the force on the cart, despite the short MPC horizon. To reproduce the results in our paper, please do the following:

1) Run the script `run_all_experiments.py` with `dimensionless` set to `True`.
2) Do the same with `dimensionless` set to `False`.
3) Run the script `utils/plotting.py`, with the appropriate paths entered on the bottom of the script.

Note that `run_all_experiments.py` can take a while (cca. 5 hours on a decently powerful workstation). Feel free to optimize the script for parallel execution etc.