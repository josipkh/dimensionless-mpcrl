# Dimensionless MPCRL
### Introduction
This repository contains the implementation of a method for solving Markov decision processes (MDPs) using dimensional analysis, model predictive control (MPC) and reinforcement learning (RL). On top of existing algorithms combining MPC and RL, the focus of the method is on exploiting the similarity between systems of different scales. Similar to scale model experiments often used in naval or aerospace engineering, the method relies on dimensional analysis to identify dynamically similar systems. If the components of the algorithm are then expressed in the dimensionless form, data from systems of different scales can be combined to find an optimal policy which jointly solves a class of similar MDPs. We demonstrate the method on a classic cart pole example. Additionally, we present a race car example where Bayesian optimization (BO) is used to tune the MPC parameters instead. A manuscript describing the method in more detail can be found on [arXiv](https://arxiv.org/abs/2512.08667).

### Usage
The implementation of the cart pole example is based on [`leap-c`](https://leap-c.github.io/leap-c/) (and its dependencies), an open-source framework for implementing, among others, MPC-based RL algorithms. To run the example, please install `leap-c` and its dependencies according to [their instructions](https://leap-c.github.io/leap-c/installation.html).

NOTE: make sure you use the version referenced in `external/leap-c`.

The race car example is implemented using [`acados`](https://docs.acados.org/) and Optuna (v.4.5.0), which can be installed using the instructions [here](https://docs.acados.org/python_interface/index.html#installation), followed by:
```bash
pip install optuna==4.5.0
```

NOTE: the example was tested with `acados` version referenced in `external/leap-c/external/acados`.

There might be some additional minor dependencies required to run the examples, but these should be straightforward.

### Funding
This work was supported in part by the [Croatian Science Foundation](https://hrzz.hr/en/).