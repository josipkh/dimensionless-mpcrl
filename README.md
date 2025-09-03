# Dimensionless MPCRL
### Introduction
This repository contains the implementation of a method for solving Markov decission processes (MDPs) based on model predictive control (MPC) and reinforcement learning (RL). On top of existing algorithms combining MPC and RL, the focus of the method is on exploiting the similarity between systems of different scales. Similar to scale model experiments often used in naval or aerospace engineering, the method relies on dimensional analysis to identify dynamically similar systems. If the components of the algorithm are then expressed in the dimensionless form, data from systems of different scales can be combined to find an optimal policy which solves a class of similar MDPs. More details on the method can be found in [this presentation](https://tinyurl.com/mobdok-presentation-jkh) and will soon be published as a manuscript.

### Usage
The implementation is based on [leap-c](https://github.com/leap-c/leap-c/), an open-source framework for implementing (among others) MPC-based RL algorithms. To run the examples in this repository, please install `leap-c` and its dependencies according to the instructions.

NOTE: make sure you use the version referenced in `external/leap-c`

### Funding
This work was supported in part by the [Croatian Science Foundation](https://hrzz.hr/en/).