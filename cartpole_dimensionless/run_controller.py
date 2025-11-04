"""A test script to run the cartpole (dimensionless) controller with default parameters."""
import datetime
from pathlib import Path
from typing import Any, Generator

import gymnasium as gym
import numpy as np

from leap_c.controller import ParameterizedController
from leap_c.run import init_run
from leap_c.torch.rl.buffer import ReplayBuffer
from leap_c.trainer import Trainer, TrainerConfig

from cartpole_dimensionless.env import CartpoleEnvDimensionless
from cartpole_dimensionless.controller import CartpoleControllerDimensionless


class ControllerTrainer(Trainer):
    """A trainer that just runs the controller with default parameters, without any training.

    Attributes:
        controller: The parameterized controller to use.
        collate_fn: The function used to collate observations and actions.
    """

    def __init__(
        self,
        cfg: TrainerConfig,
        val_env: gym.Env,
        output_path: str | Path,
        device: str,
        controller: ParameterizedController,
    ):
        """

        Args:
            cfg: The trainer configuration.
            val_env: The validation environment.
            output_path: The path to save outputs to.
            device: The device to use.
            controller: The parameterized controller to use.
        """
        super().__init__(cfg, val_env, output_path, device)
        self.controller = controller

        buffer = ReplayBuffer(1, device, collate_fn_map=controller.collate_fn_map)
        self.collate_fn = buffer.collate


    def train_loop(self) -> Generator[int, None, None]:
        """No training - just return immediately."""
        while True:
            yield 1


    def act(
        self, obs, deterministic: bool = False, state=None
    ) -> tuple[np.ndarray, Any, dict[str, float]]:
        """Use the controller with default parameters."""
        obs_batched = self.collate_fn([obs])

        default_param = self.controller.default_param(obs)

        param_batched = self.collate_fn([default_param])

        ctx, action = self.controller(obs_batched, param_batched, ctx=state)

        action = action.cpu().numpy()[0]

        return action, ctx, {}


def create_cfg(seed: int) -> TrainerConfig:
    """Return the default configuration for running controller experiments."""
    cfg = TrainerConfig()

    cfg.seed = seed
    cfg.train_steps = 1  # No training
    cfg.train_start = 0
    cfg.val_freq = 1  # Validate immediately
    cfg.val_num_rollouts = 1  # One validation rollout is enough
    cfg.val_deterministic = True
    cfg.val_num_render_rollouts = 1  # Render the validation rollout (create video)
    cfg.val_render_mode = "rgb_array"
    cfg.val_report_score = "cum"
    cfg.ckpt_modus = "none"  # No checkpoints needed

    cfg.log.verbose = True
    cfg.log.interval = 1_000
    cfg.log.window = 10_000
    cfg.log.csv_logger = True
    cfg.log.tensorboard_logger = False
    cfg.log.wandb_logger = False
    cfg.log.wandb_init_kwargs = {}

    return cfg


def run_controller(
    cfg: TrainerConfig,
    env: gym.Env,
    controller: ParameterizedController,
    output_path: str | Path,
    device: str,
) -> float:
    """
    Args:
        cfg: The configuration for running the controller.
        output_path: The path to save outputs to.
        device: The device to use.
    """
    trainer = ControllerTrainer(
        val_env=env,
        controller=controller,
        output_path=output_path,
        device=device,
        cfg=cfg,
    )
    init_run(trainer, cfg, output_path)

    print("Running the closed-loop test...")

    final_score = trainer.run()
    print(f"Final validation score: {final_score}")

    return final_score


if __name__ == "__main__":
    from cartpole_dimensionless.config import get_default_cartpole_params
    # from utils import get_similar_cartpole_params

    env_params = get_default_cartpole_params()
    mpc_params = env_params
    dimensionless = False
    seed = 0
    keep_output = False
    device = "cpu"
    render_mode = "rgb_array"

    env = CartpoleEnvDimensionless(cartpole_params=env_params, mpc_cartpole_params=mpc_params, dimensionless=dimensionless, render_mode=render_mode)
    controller = CartpoleControllerDimensionless(cartpole_params=mpc_params, dimensionless=dimensionless)
    cfg = create_cfg(seed)
    output_root = "output" if keep_output else "/tmp"
    time_str = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    output_path = Path(f"{output_root}/cartpole/test_controller_{seed}_{time_str}")

    run_controller(cfg, env, controller, output_path, device)
