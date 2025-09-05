import gymnasium as gym
import numpy as np
from config import CarParams, get_default_car_params
from utils import get_transformation_matrices
from model import export_acados_integrator


class RaceCarEnvDimensionless(gym.Env):
    """TODO: write me."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        render_mode: str | None = None,
        car_params: CarParams | None = None,
        dimensionless: bool = True,
        mpc_car_params: CarParams | None = None,
    ):
        if car_params is None or mpc_car_params is None:
            raise ValueError("Car parameters not provided in the env.")

        if dimensionless:
            # use agent instead of env parameters
            self.Ms, self.Ma, self.Mt = get_transformation_matrices(
                mpc_car_params
            )  # s(physical) = Ms * s(dimensionless)
            self.Ms_inv = np.linalg.inv(self.Ms)
            self.Ma_inv = np.linalg.inv(self.Ma)
            self.Mt_inv = np.linalg.inv(self.Mt)

        self.dt = car_params.dt.item()
        self.dimensionless = dimensionless  # whether to use the dimensionless formulation
        self.nx = 4  # number of states
        self.nu = 2  # number of inputs

        self.integrator = export_acados_integrator(car_params=car_params)

        # state and action bounds
        large_number = 1e3  # states unbounded for now
        obs_ub = large_number * np.ones((3,), dtype=np.float32)
        obs_lb = -obs_ub
        act_ub = np.array([car_params.D_max.item(), car_params.delta_max.item()], dtype=np.float32)
        act_lb = np.array([car_params.D_min.item(), car_params.delta_min.item()], dtype=np.float32)

        if dimensionless:
            obs_ub = self.Ms_inv @ obs_ub
            obs_lb = self.Ms_inv @ obs_lb
            act_ub = self.Ma_inv @ act_ub
            act_lb = self.Ma_inv @ act_lb

        self.action_space = gym.spaces.Box(act_lb, act_ub)
        self.observation_space = gym.spaces.Box(obs_lb, obs_ub)

        self.reset_needed = True
        self.t = 0  # physical time
        self.s = None  # physical state

        # For rendering
        if not (render_mode is None or render_mode in self.metadata["render_modes"]):
            raise ValueError(
                f"render_mode must be one of {self.metadata['render_modes']}"
            )
        self.render_mode = render_mode
        self.screen_width = 600
        self.screen_height = 400
        self.window = None
        self.clock = None

        # helper functions for scaling
        self.dim2nondim_s = lambda s: self.Ms_inv @ s if dimensionless else s
        self.nondim2dim_s = lambda s: self.Ms @ s if dimensionless else s
        self.dim2nondim_a = lambda a: self.Ma_inv @ a if dimensionless else a
        self.nondim2dim_a = lambda a: self.Ma @ a if dimensionless else a

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Execute the dynamics of the pendulum on cart."""
        if self.reset_needed:
            raise RuntimeError("Call reset before using the step method.")

        # scale the MPC output back if dimensionless
        if self.dimensionless:
            action = self.nondim2dim_a(action)

        # simulate one time step
        self.s = self.integrator.simulate(s=self.s, a=action)
        self.s_trajectory.append(self.s)  # type: ignore
        self.t += self.dt

        # TODO: continue here

        # calculate the reward
        # (if dimensionless, should be scaled with the env parameters)
        r = abs(np.pi - (abs(theta))) / (10 * np.pi)  # Reward for swingup; Max: 0.1

        # check for termination
        term = False
        trunc = False
        info = {}
        if np.abs(self.s[0]) > self.x_threshold:
            term = True  # Just terminating should be enough punishment when reward is positive
            info = {"task": {"violation": True, "success": False}}
        if self.t > self.max_episode_length:
            # check if the pole is upright in the last 10 steps
            if len(self.s_trajectory) >= 10:
                success = all(
                    np.abs(self.s_trajectory[i][1]) < 0.1 for i in range(-10, 0)
                )  # TODO: check if 0.1 is a good limit
            else:
                success = False  # Not enough data to determine success

            info = {"task": {"violation": False, "success": success}}
            trunc = True
        self.reset_needed = trunc or term

        # make the observation (x,theta,dx,dtheta) dimensionless
        obs = self.dim2nondim_s(self.s) if self.dimensionless else self.s

        return obs, r, term, trunc, info

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:  # type: ignore
        if seed is not None:
            super().reset(seed=seed)
            self.observation_space.seed(seed)
            self.action_space.seed(seed)

        self.t = 0
        self.s = self.init_state()
        self.reset_needed = False

        self.s_trajectory = []

        obs = self.dim2nondim_s(self.s) if self.dimensionless else self.s
        return obs, {}

    def init_state(self) -> np.ndarray:
        """The pendulum is hanging down at the start."""
        return np.array([0.0, np.pi, 0.0, 0.0])

    def render(self):
        pass

    def close(self):
        if self.window is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()


if __name__ == "__main__":
    from utils import get_similar_car_params
    dimensionless = True

    # create envs for the default and similar parameters
    params_ref = get_default_car_params()
    env_ref = RaceCarEnvDimensionless(
        car_params=params_ref,
        dimensionless=dimensionless,
        mpc_car_params=params_ref
    )

    params_sim = get_similar_car_params(
        reference_params=params_ref,
        new_length=4.0,
        new_mass=1500.0,
        new_cr3=2.0
    )
    env_sim = RaceCarEnvDimensionless(
        car_params=params_sim,
        dimensionless=dimensionless,
        mpc_car_params=params_sim
    )

    assert env_ref.action_space == env_sim.action_space
    assert env_ref.observation_space == env_sim.observation_space

    seed = 0
    obs_ref = env_ref.reset(seed=seed)[0]
    obs_sim = env_sim.reset(seed=seed)[0]
    assert np.allclose(obs_ref, obs_sim)

    diffs = []
    obs_ref_log = []
    obs_sim_log = []
    obs_ref_log.append(obs_ref)
    obs_sim_log.append(obs_sim)
    act_log = []
    for _ in range(1000):
        action = env_ref.action_space.sample()
        obs_ref, reward_ref, done_ref, truncated_ref, info_ref = env_ref.step(action)
        obs_sim, reward_sim, done_sim, truncated_sim, info_sim = env_sim.step(action)

        if done_ref or done_sim or truncated_ref or truncated_sim:
            seed += 1
            env_ref.reset(seed=seed)
            env_sim.reset(seed=seed)

        obs_ref_log.append(obs_ref)
        obs_sim_log.append(obs_sim)
        diffs.append(np.max(np.abs(obs_ref - obs_sim)))
        act_log.append(action)
        # assert np.allclose(obs_ref, obs_sim, atol=1e-04)
    print(f"max diff: {np.max(diffs)}")

    # import matplotlib.pyplot as plt

    # obs_ref_log = np.array(obs_ref_log)
    # obs_sim_log = np.array(obs_sim_log)
    # act_log = np.array(act_log)
    # nx = 4
    # fig, ax = plt.subplots(nx + 1, 1, sharex=True)
    # labels = ["ref", "sim"]  # ["scipy", "acados"]
    # for i in range(nx):
    #     ax[i].plot(obs_ref_log[:, i], color="b", label=labels[0])
    #     ax[i].plot(obs_sim_log[:, i], color="r", linestyle="--", label=labels[1])
    #     ax[i].grid()
    #     ax[i].set_ylabel(f"$x_{i}$")
    # ax[0].legend()
    # ax[-1].step(
    #     list(range(obs_ref_log.shape[0])),
    #     np.append([act_log[0]], act_log),
    #     where="post",
    #     color="b",
    # )
    # ax[-1].set_ylabel("$u$")
    # ax[-1].grid()

    # plt.subplots_adjust(left=None, bottom=None, right=None, top=None, hspace=0.4)
    # fig.align_ylabels(ax)
    # plt.show(block=False)
    # print("ok")
    # print("Press ENTER to close the plot")
    # input()
    # plt.close()

    env_ref.close()
    env_sim.close()

