#
# Copyright (c) The acados authors.
#
# This file is part of acados.
#
# The 2-Clause BSD License
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.;
#

# author: Daniel Kloeser

import casadi as ca
from utils import get_track
import numpy as np
from config import CarParams, get_default_car_params
from acados_template import AcadosModel, AcadosSim, AcadosSimSolver


def bicycle_model_ocp(car_params: CarParams):
    # define structs
    constraint = ca.types.SimpleNamespace()
    model = ca.types.SimpleNamespace()

    # load track parameters
    [s0, _, _, _, kapparef] = get_track(car_params=car_params)
    length = len(s0)
    pathlength = s0[-1]
    # copy loop to beginning and end
    s0 = np.append(s0, [s0[length - 1] + s0[1:length]])
    kapparef = np.append(kapparef, kapparef[1:length])
    s0 = np.append([-s0[length - 2] + s0[length - 81 : length - 2]], s0)
    kapparef = np.append(kapparef[length - 80 : length - 1], kapparef)

    # compute spline interpolations
    kapparef_s = ca.interpolant("kapparef_s", "bspline", [s0], kapparef)

    ## Race car parameters
    m = car_params.m[0]
    C1 = car_params.lr[0] / car_params.l[0]
    C2 = 1 / car_params.l[0]
    Cm1 = car_params.cm1[0]
    Cm2 = car_params.cm2[0]
    Cr0 = car_params.cr0[0]
    Cr2 = car_params.cr2[0]

    ## CasADi Model
    # set up states & controls
    s = ca.SX.sym("s")
    n = ca.SX.sym("n")
    alpha = ca.SX.sym("alpha")
    v = ca.SX.sym("v")
    D = ca.SX.sym("D")
    delta = ca.SX.sym("delta")
    x = ca.vertcat(s, n, alpha, v, D, delta)

    # controls
    derD = ca.SX.sym("derD")
    derDelta = ca.SX.sym("derDelta")
    u = ca.vertcat(derD, derDelta)

    # xdot
    sdot = ca.SX.sym("sdot")
    ndot = ca.SX.sym("ndot")
    alphadot = ca.SX.sym("alphadot")
    vdot = ca.SX.sym("vdot")
    Ddot = ca.SX.sym("Ddot")
    deltadot = ca.SX.sym("deltadot")
    xdot = ca.vertcat(sdot, ndot, alphadot, vdot, Ddot, deltadot)

    # dynamics
    Fxd = (Cm1 - Cm2 * v) * D - Cr2 * v * v - Cr0 * ca.tanh(5 * v)
    sdota = (v * ca.cos(alpha + C1 * delta)) / (1 - kapparef_s(s) * n)
    f_expl = ca.vertcat(
        sdota,
        v * ca.sin(alpha + C1 * delta),
        v * C2 * delta - kapparef_s(s) * sdota,
        Fxd / m * ca.cos(C1 * delta),
        derD,
        derDelta,
    )

    # constraint on forces
    a_lat = C2 * v * v * delta + Fxd * ca.sin(C1 * delta) / m
    a_long = Fxd / m

    # Model bounds
    model.n_min = car_params.n_min[0]
    model.n_max = car_params.n_max[0]

    # state bounds
    model.throttle_min = car_params.D_min[0]
    model.throttle_max = car_params.D_max[0]

    model.delta_min = car_params.delta_min[0]
    model.delta_max = car_params.delta_max[0]

    # input bounds
    model.ddelta_min = car_params.ddelta_min[0]
    model.ddelta_max = car_params.ddelta_max[0]
    model.dthrottle_min = car_params.dD_min[0]
    model.dthrottle_max = car_params.dD_max[0]

    # nonlinear constraint
    constraint.alat_min = car_params.a_lat_min[0]
    constraint.alat_max = car_params.a_lat_max[0]

    constraint.along_min = car_params.a_long_min[0]
    constraint.along_max = car_params.a_long_max[0]

    # Define initial conditions
    model.x0 = np.array([-2, 0, 0, 0, 0, 0])

    # define constraints struct
    constraint.alat = ca.Function("a_lat", [x, u], [a_lat])
    constraint.pathlength = pathlength
    constraint.expr = ca.vertcat(a_long, a_lat, n, D, delta)

    model_name = "spatial_bicycle_model_ocp"

    # add labels for states and controls
    model.x_labels = [
        "$s$ [m]",
        "$n$ [m]",
        r"$\alpha$ [rad]",
        "$v$ [m/s]",
        "$D$ [-]",
        r"$\delta$ [rad]",
    ]
    model.u_labels = [
        r"$\dot{\D}$ [rad/s]",
        r"$\dot{\delta}$ [rad/s]",
    ]
    model.t_label = "$t$ [s]"

    # Define model struct
    model.f_impl_expr = xdot - f_expl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u
    model.name = model_name
    return model, constraint


def bicycle_model_sim(car_params: CarParams) -> ca.types.SimpleNamespace:
    # get the OCP model
    model_ocp = bicycle_model_ocp(car_params=car_params)[0]

    # remove the control input rate from the model for simulation
    nx = 4
    nu = 2
    model = ca.types.SimpleNamespace()
    model.f_impl_expr = model_ocp.f_impl_expr[0:nx]
    model.f_expl_expr = model_ocp.f_expl_expr[0:nx]
    model.x = model_ocp.x[0:nx]
    model.xdot = model_ocp.xdot[0:nx]
    model.u = model_ocp.x[nx:nx+nu]
    model.name = model_ocp.name[:-4] + "_sim"
    model.x_labels = model_ocp.x_labels[0:nx]
    model.u_labels = model_ocp.x_labels[nx:nx+nu]
    model.t_label = model_ocp.t_label
    return model


def export_acados_integrator(car_params: CarParams) -> AcadosSimSolver:
    """Create and return an acados integrator for the car model."""
    model = bicycle_model_sim(car_params=car_params)
    acados_model = AcadosModel()
    acados_model.x = model.x
    acados_model.u = model.u
    acados_model.f_expl_expr = model.f_expl_expr
    acados_model.f_impl_expr = model.f_impl_expr
    acados_model.xdot = model.xdot
    acados_model.name = "car_model"
    acados_model.x_labels = model.x_labels
    acados_model.u_labels = model.u_labels
    acados_model.t_label = model.t_label

    sim = AcadosSim()
    sim.model = acados_model
    sim.solver_options.T = car_params.dt.item()
    sim.solver_options.integrator_type = "ERK"
    sim.solver_options.num_stages = 4
    sim.solver_options.num_steps = 1
    print("Setting up acados integrator...")
    return AcadosSimSolver(sim, verbose=False)


if __name__ == "__main__":
    from config import get_default_car_params
    car_params = get_default_car_params()
    integrator = export_acados_integrator(car_params=car_params)
    print("Integrator successfully created.")