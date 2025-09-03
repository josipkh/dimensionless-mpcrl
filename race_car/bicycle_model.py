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
from tracks.read_data_fcn import getTrack
import numpy as np
from config import get_default_car_params


def bicycle_model(track="LMS_Track.txt"):
    # define structs
    constraint = ca.types.SimpleNamespace()
    model = ca.types.SimpleNamespace()

    model_name = "Spatialbicycle_model"

    # load track parameters
    [s0, _, _, _, kapparef] = getTrack(track)
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
    # m = 0.043
    # C1 = 0.5  # [-] lr / (lr + lf)
    # C2 = 15.5  # [1/m] 1 / (lr + lf) -> L = 0.06451 m
    # Cm1 = 0.28
    # Cm2 = 0.05
    # Cr0 = 0.011
    # Cr2 = 0.006

    car_params = get_default_car_params()
    m = car_params.m[0]
    C1 = car_params.lr[0] / car_params.l[0]
    C2 = 1 / car_params.l[0]
    Cm1 = car_params.cm1[0]
    Cm2 = car_params.cm2[0]
    Cr0 = car_params.cr0[0]
    Cr2 = car_params.cr2[0]

    ## CasADi Model
    # set up states & controls
    s = ca.MX.sym("s")
    n = ca.MX.sym("n")
    alpha = ca.MX.sym("alpha")
    v = ca.MX.sym("v")
    D = ca.MX.sym("D")
    delta = ca.MX.sym("delta")
    x = ca.vertcat(s, n, alpha, v, D, delta)

    # controls
    derD = ca.MX.sym("derD")
    derDelta = ca.MX.sym("derDelta")
    u = ca.vertcat(derD, derDelta)

    # xdot
    sdot = ca.MX.sym("sdot")
    ndot = ca.MX.sym("ndot")
    alphadot = ca.MX.sym("alphadot")
    vdot = ca.MX.sym("vdot")
    Ddot = ca.MX.sym("Ddot")
    deltadot = ca.MX.sym("deltadot")
    xdot = ca.vertcat(sdot, ndot, alphadot, vdot, Ddot, deltadot)

    # algebraic variables
    z = ca.vertcat([])

    # parameters
    p = ca.vertcat([])

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
    # model.n_min = -0.12  # width of the track [m]
    # model.n_max = 0.12  # width of the track [m]
    model.n_min = car_params.n_min[0]
    model.n_max = car_params.n_max[0]

    # state bounds
    # model.throttle_min = -1.0
    # model.throttle_max = 1.0
    model.throttle_min = car_params.D_min[0]
    model.throttle_max = car_params.D_max[0]

    # model.delta_min = -0.40  # minimum steering angle [rad]
    # model.delta_max = 0.40  # maximum steering angle [rad]
    model.delta_min = car_params.delta_min[0]
    model.delta_max = car_params.delta_max[0]

    # input bounds
    # model.ddelta_min = -2.0  # minimum change rate of stering angle [rad/s]
    # model.ddelta_max = 2.0  # maximum change rate of steering angle [rad/s]
    # model.dthrottle_min = -10  # -10.0  # minimum throttle change rate
    # model.dthrottle_max = 10  # 10.0  # maximum throttle change rate
    model.ddelta_min = car_params.ddelta_min[0]
    model.ddelta_max = car_params.ddelta_max[0]
    model.dthrottle_min = car_params.dD_min[0]
    model.dthrottle_max = car_params.dD_max[0]

    # nonlinear constraint
    # constraint.alat_min = -4  # maximum lateral force [m/s^2]
    # constraint.alat_max = 4  # maximum lateral force [m/s^2]
    constraint.alat_min = car_params.a_lat_min[0]
    constraint.alat_max = car_params.a_lat_max[0]

    # constraint.along_min = -4  # maximum lateral force [m/s^2]
    # constraint.along_max = 4  # maximum lateral force [m/s^2]
    constraint.along_min = car_params.a_long_min[0]
    constraint.along_max = car_params.a_long_max[0]

    # Define initial conditions
    model.x0 = np.array([-2, 0, 0, 0, 0, 0])

    # define constraints struct
    constraint.alat = ca.Function("a_lat", [x, u], [a_lat])
    constraint.pathlength = pathlength
    constraint.expr = ca.vertcat(a_long, a_lat, n, D, delta)

    # Define model struct
    params = ca.types.SimpleNamespace()
    params.C1 = C1
    params.C2 = C2
    params.Cm1 = Cm1
    params.Cm2 = Cm2
    params.Cr0 = Cr0
    params.Cr2 = Cr2
    model.f_impl_expr = xdot - f_expl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u
    model.z = z
    model.p = p
    model.name = model_name
    model.params = params
    return model, constraint
