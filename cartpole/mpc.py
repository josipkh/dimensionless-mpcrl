import casadi as ca
import numpy as np
import scipy.linalg
from acados_template import AcadosOcp
from config import CartPoleParams, create_acados_params
from utils import get_transformation_matrices
from leap_c.ocp.acados.controller import AcadosController
from leap_c.ocp.acados.parameters import AcadosParameterManager
from leap_c.ocp.acados.torch import AcadosDiffMpcTorch
from model import export_cartpole_model, export_dimensionless_cartpole_model
import os
from leap_c.examples.utils.casadi import integrate_erk4


class CartpoleControllerDimensionless(AcadosController):
    """TODO"""

    def __init__(
        self,
        cartpole_params: CartPoleParams,
        dimensionless: bool,
    ):
        """TODO"""
        param_manager = AcadosParameterManager(parameters=create_acados_params(cartpole_params), N_horizon=cartpole_params.N.item())
        ocp = export_parametric_ocp(
            param_manager=param_manager,
            cartpole_params=cartpole_params,
            dimensionless=dimensionless,
        )

        diff_mpc = AcadosDiffMpcTorch(ocp=ocp)
        super().__init__(param_manager=param_manager, diff_mpc=diff_mpc)


def export_parametric_ocp(
    param_manager: AcadosParameterManager,
    cartpole_params: CartPoleParams,
    dimensionless: bool
) -> AcadosOcp:
    ocp = AcadosOcp()

    ocp.solver_options.N_horizon = cartpole_params.N.item()
    ocp.dims.nx = 4
    ocp.dims.nu = 1

    if dimensionless:
        Mx, Mu, Mt = get_transformation_matrices(cartpole_params)  # x(physical) = Mx * x(dimensionless)
        # Mx_inv = np.linalg.inv(Mx)
        Mu_inv = np.linalg.inv(Mu)
        Mt_inv = np.linalg.inv(Mt).item()

    dt = cartpole_params.dt.item() * (Mt_inv if dimensionless else 1.0)
    ocp.solver_options.tf = dt * ocp.solver_options.N_horizon

    ######## Model ########
    p = ca.vertcat(
        param_manager.non_learnable_parameters.cat,
        param_manager.learnable_parameters.cat,
    )  # type:ignore

    if dimensionless:
        ocp.model = export_dimensionless_cartpole_model(cartpole_params=cartpole_params)
    else:
        ocp.model = export_cartpole_model(cartpole_params=cartpole_params)
    ocp.model.disc_dyn_expr = integrate_erk4(
        f_expl=ocp.model.f_expl_expr,
        x=ocp.model.x,
        u=ocp.model.u,
        p=p,
        dt=dt,
    )

    ######## Cost ########
    xref = ca.vertcat(*[param_manager.get(f"xref{i}") for i in range(4)])
    uref = param_manager.get("uref")
    yref = ca.vertcat(xref, uref)  # type:ignore
    yref_e = yref[: ocp.dims.nx]
    y = ca.vertcat(ocp.model.x, ocp.model.u)
    y_e = ocp.model.x

    q_diag_sqrt = param_manager.get("q_diag_sqrt")
    r_diag_sqrt = param_manager.get("r_diag_sqrt")
    W_sqrt = ca.diag(ca.vertcat(q_diag_sqrt, r_diag_sqrt))
    W = W_sqrt @ W_sqrt.T
    if dimensionless:
        MxMu_block = scipy.linalg.block_diag(Mx, Mu)
        W = MxMu_block.T @ W @ MxMu_block
    W_e = W[: ocp.dims.nx, : ocp.dims.nx]
    
    ocp.cost.cost_type = "NONLINEAR_LS"
    ocp.cost.cost_type_e = "NONLINEAR_LS"

    ocp.cost.W = W
    ocp.cost.yref = yref
    ocp.model.cost_y_expr = y

    ocp.cost.W_e = W_e
    ocp.cost.yref_e = yref_e
    ocp.model.cost_y_expr_e = y_e

    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"

    ######## Constraints ########
    # initial state
    ocp.constraints.idxbx_0 = np.array([0, 1, 2, 3])
    ocp.constraints.x0 = np.array([0.0, np.pi, 0.0, 0.0])  # no scaling needed

    # input force
    ocp.constraints.idxbu = np.array([0])
    ocp.constraints.ubu = np.array([cartpole_params.Fmax.item() * (Mu_inv if dimensionless else 1.0)])
    ocp.constraints.lbu = -ocp.constraints.ubu

    # cart position
    x_max = 3 * (1.0 if dimensionless else cartpole_params.l.item())
    ocp.constraints.ubx = np.array([x_max])
    ocp.constraints.lbx = -ocp.constraints.ubx
    ocp.constraints.idxbx = np.array([0])
    ocp.constraints.ubx_e = np.array([x_max])
    ocp.constraints.lbx_e = -ocp.constraints.ubx_e
    ocp.constraints.idxbx_e = np.array([0])

    # scale the slack penalty on the cart position
    # 640.0 corresponds to 1e3 for the default (physical) system
    slack_penalty = 640.0 / (1.0 if dimensionless else cartpole_params.l.item() ** 2)
    ocp.constraints.idxsbx = np.array([0])
    ocp.cost.Zu = ocp.cost.Zl = np.array([slack_penalty])
    ocp.cost.zu = ocp.cost.zl = np.array([0.0])

    ocp.constraints.idxsbx_e = np.array([0])
    ocp.cost.Zu_e = ocp.cost.Zl_e = np.array([slack_penalty])
    ocp.cost.zu_e = ocp.cost.zl_e = np.array([0.0])

    ######## Solver configuration ########
    ocp.solver_options.integrator_type = "DISCRETE"
    ocp.solver_options.nlp_solver_type = "SQP"

    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.qp_solver_ric_alg = 1

    # specific codegen folder to prevent overwriting
    # ocp.code_export_directory = os.path.join("codegen", "cartpole", f"ocp_{cartpole_params.l.item():.3g}".replace(".", "_"))
    # if dimensionless:
    #     ocp.code_export_directory += "_dimensionless"

    # assign param manager to ocp
    param_manager.assign_to_ocp(ocp)

    return ocp


if __name__ == "__main__":
    """Compare the dimensional OCP with the reference implementation from leap-c."""
    from config import get_default_cartpole_params, create_acados_params
    from leap_c.examples.cartpole.acados_ocp import export_parametric_ocp as export_ref
    from leap_c.examples.cartpole.acados_ocp import create_cartpole_params
    from leap_c.ocp.acados.torch import AcadosDiffMpcTorch
    import torch

    cartpole_params = get_default_cartpole_params()
    N_horizon = cartpole_params.N.item()

    acados_params_ref = create_cartpole_params(param_interface="global", N_horizon=N_horizon)
    param_manager_ref = AcadosParameterManager(acados_params_ref, N_horizon)
    ocp_ref = export_ref(
        param_manager=param_manager_ref,
        cost_type="NONLINEAR_LS",
        name="cartpole",
        N_horizon=N_horizon,
        T_horizon=cartpole_params.dt.item() * N_horizon,
        Fmax=cartpole_params.Fmax.item(),
        x_threshold=2.4,
    )
    diff_mpc_ref = AcadosDiffMpcTorch(ocp_ref, export_directory=ocp_ref.code_export_directory)

    acados_params_sim = create_acados_params(cartpole_params)
    param_manager_sim = AcadosParameterManager(acados_params_sim, N_horizon)
    ocp_sim = export_parametric_ocp(
        param_manager=param_manager_sim,
        cartpole_params=cartpole_params,
        dimensionless=False,
    )
    diff_mpc_sim = AcadosDiffMpcTorch(ocp_sim, export_directory=ocp_sim.code_export_directory)

    # check the solution for a random initial state
    nx = 4
    x0 = torch.rand(1, nx)
    sol_ref = diff_mpc_ref.forward(x0)
    sol_sim = diff_mpc_sim.forward(x0)
    print(sol_ref)
    print(sol_sim)

    print("ok")
