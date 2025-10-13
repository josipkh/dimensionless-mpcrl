"""This modules extends `leap-c` with a new environment and controller."""

__all__ = ["create_env", "create_controller"]

from leap_c.examples import CONTROLLER_REGISTRY, ENV_REGISTRY, create_controller, create_env

from race_car.env import RaceCarEnv
from race_car.controller import RaceCarController, RaceCarControllerConfig

ENV_REGISTRY["race_car"] = RaceCarEnv
CONTROLLER_REGISTRY["race_car"] = (RaceCarController, RaceCarControllerConfig, dict())