from race_car.utils.config import CarParams
from acados_template import AcadosOcp
from model import car_model_ocp
import numpy as np
from race_car.utils.scaling import get_cost_matrices
import scipy.linalg
import casadi as ca


def export_ocp(car_params: CarParams, dimensionless: bool) -> AcadosOcp:
    """Exports the OCP for the given race car parameters."""
    ocp = AcadosOcp()

    # horizon and time step
    ocp.solver_options.N_horizon = int(car_params.N.item())
    ocp.solver_options.tf = car_params.dt.item() * int(car_params.N.item())

    # prediction model
    ocp.model = car_model_ocp(car_params=car_params, dimensionless=dimensionless)

    # constraints on states
    ocp.constraints.idxbx = np.array([1, 4, 5])
    ocp.constraints.lbx = np.array([car_params.n_min.item(), car_params.D_min.item(), car_params.delta_min.item()])
    ocp.constraints.ubx = np.array([car_params.n_max.item(), car_params.D_max.item(), car_params.delta_max.item()])

    ocp.constraints.idxbx_e = ocp.constraints.idxbx
    ocp.constraints.lbx_e = ocp.constraints.lbx
    ocp.constraints.ubx_e = ocp.constraints.ubx

    # constraints on inputs
    ocp.constraints.idxbu = np.array([0, 1])
    ocp.constraints.lbu = np.array([car_params.dD_min.item(), car_params.ddelta_min.item()])
    ocp.constraints.ubu = np.array([car_params.dD_max.item(), car_params.ddelta_max.item()])

    # constraints on nonlinear expressions (defined in the model)
    lh = np.array([car_params.a_long_min.item(), car_params.a_lat_min.item()])
    uh = np.array([car_params.a_long_max.item(), car_params.a_lat_max.item()])
    ocp.constraints.lh = lh
    ocp.constraints.uh = uh
    ocp.constraints.lh_e = lh
    ocp.constraints.uh_e = uh

    # keep all constraints hard for now, no slacks

    # cost weights (NONLINEAR_LS), defined for the discrete-time cost
    Q, R, Qe = get_cost_matrices(car_params=car_params)
    ocp.solver_options.cost_scaling = np.ones(ocp.solver_options.N_horizon + 1)

    ocp.cost.cost_type_0 = 'NONLINEAR_LS'
    ocp.model.cost_y_expr_0 = ocp.model.u
    ocp.cost.yref_0 = np.zeros(ocp.model.cost_y_expr_0.shape)  # will not be changed later
    ocp.cost.W_0 = R
    
    ocp.cost.cost_type = 'NONLINEAR_LS'
    ocp.model.cost_y_expr = ca.vertcat(ocp.model.x, ocp.model.u)
    ocp.cost.yref = np.zeros(ocp.model.cost_y_expr.shape)  # will be changed at runtime
    ocp.cost.W = scipy.linalg.block_diag(Q, R)

    ocp.cost.cost_type_e = 'NONLINEAR_LS'
    ocp.model.cost_y_expr_e = ocp.model.x
    ocp.cost.yref_e = np.zeros(ocp.model.cost_y_expr_e.shape)  # will be changed at runtime
    ocp.cost.W_e = Qe

    ocp.constraints.x0 = np.zeros(ocp.model.x.shape)  # initial state, will be set in the MPC loop

    # solver options
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.sim_method_num_stages = 4
    ocp.solver_options.sim_method_num_steps = 3
    ocp.solver_options.nlp_solver_max_iter = 200
    ocp.solver_options.tol = 1e-4

    ocp.code_export_directory = os.path.join("codegen", f"ocp_{car_params.l.item():.3g}".replace(".", "_"))  # prevent overwriting

    if dimensionless:
        ocp = nondimensionalize_ocp(ocp, car_params)

    return ocp


def nondimensionalize_ocp(ocp: AcadosOcp, car_params: CarParams) -> AcadosOcp:
    raise NotImplementedError

    return ocp


if __name__ == "__main__":
    from race_car.utils.config import get_default_car_params
    from race_car.utils.scaling import get_large_car_params
    from race_car.utils.track import get_track
    from race_car.utils.plotting import plot_results_classic, plot_results_track, plot_lat_acc
    from model import export_acados_integrator
    from acados_template import AcadosOcpSolver
    import os
    import time
    import matplotlib.pyplot as plt

    dimensionless = False

    # create the OCP solver
    car_params = get_default_car_params()
    acados_ocp = export_ocp(car_params=car_params, dimensionless=dimensionless)
    acados_ocp_solver = AcadosOcpSolver(
        acados_ocp=acados_ocp, 
        verbose=False,
        json_file=os.path.join("json", f"ocp_{car_params.l.item():.3g}".replace(".", "_") + ".json")  # prevent overwriting
    )
    print("Setting up acados OCP solver...")

    # create the integrator for simulation (always dimensional)
    integrator = export_acados_integrator(car_params=car_params, dimensionless=False)

    # get the track data
    s_max = get_track(car_params)[0][-1]  # total track length
    sref_N = 46.5 * car_params.l.item()  # terminal progress reference ("carrot")

    # preallocate log data
    nx = acados_ocp.model.x.rows()
    nu = acados_ocp.model.u.rows()
    Nsim = 500  # from the original example
    simX = np.zeros((Nsim, nx))
    simU = np.zeros((Nsim, nu))
    s0 = 0.0  # start from standstill
    simX[0, 0] = s0
    tcomp_sum = 0
    tcomp_max = 0
    n_solver_fails = 0

    # closed-loop simulation
    N = acados_ocp.solver_options.N_horizon
    for i in range(Nsim-1):
        # set the initial condition
        acados_ocp_solver.constraints_set(0, "lbx", simX[i, :])
        acados_ocp_solver.constraints_set(0, "ubx", simX[i, :])

        # update reference (progress along centerline)
        s0 = simX[i, 0]  # current track progress
        for j in range(1, N):  # for the intermediate stages
            yref = np.array([s0 + sref_N * j / N, 0, 0, 0, 0, 0, 0, 0])
            acados_ocp_solver.cost_set(j, "yref", yref)
        acados_ocp_solver.cost_set(N, "yref", np.array([s0 + sref_N, 0, 0, 0, 0, 0]))

        # solve the ocp
        t = time.time()

        status = acados_ocp_solver.solve()
        if status != 0:
            print("acados returned status {} in closed loop iteration {}.".format(status, i))
            n_solver_fails += 1

        elapsed = time.time() - t

        # record timings
        tcomp_sum += elapsed
        if elapsed > tcomp_max:
            tcomp_max = elapsed

        # get the optimal control input (part of the state in this formulation)
        u_opt = acados_ocp_solver.get(1, "x")[-nu:]

        # simulate one step with the integrator
        x_current = simX[i, :nx-nu]  # remove the extended states
        x_next = integrator.simulate(x=x_current, u=u_opt)

        # logging (inputs are control rates)
        simX[i+1, :-nu] = x_next
        simX[i+1, -nu:] = u_opt
        # simX[i+1, :] = acados_ocp_solver.get(1, "x")  # take the ocp solution as the simulation
        # simU[i, :] = acados_ocp_solver.get(0, "u")

        # check if one lap is done and break and remove entries beyond
        if x_next[0] > s_max + 0.1:
            # find where vehicle first crosses start line
            N0 = np.where(np.diff(np.sign(simX[:, 0])))[0][0]
            Nsim = i - N0  # correct to final number of simulation steps for plotting
            simX = simX[N0:i, :]
            simU = simU[N0:i, :]
            break
    
    # print some stats
    def format_number(x):
        if abs(x) >= 1:
            return "{:.2f}".format(x)
        else:
            return "{:.3g}".format(x)  # 3 significant digits
    
    print("Average computation time: {} s".format(format_number(tcomp_sum / Nsim)))
    print("Maximum computation time: {} s".format(format_number(tcomp_max)))
    print("Average speed: {} m/s".format(format_number(np.average(simX[:, 3]))))
    print("Lap time: {} s".format(format_number(Nsim * car_params.dt.item())))
    print("Number of solver fails: {} ({} %)".format(n_solver_fails, format_number(100*n_solver_fails/Nsim)))

    # plot the results
    t = np.linspace(0.0, Nsim * car_params.dt.item(), Nsim)
    # plot_results_classic(simX, simU, t)
    # plot_lat_acc(simX, simU, t, car_params)
    plot_results_track(simX, car_params, t[-1])

    input("Press ENTER to continue...")
    plt.close('all')