# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from ast import arg
import numpy as np
import random

from bidexhands.utils.config import set_np_formatting, set_seed, get_args, parse_sim_params, load_cfg
from bidexhands.utils.parse_task import parse_task
from bidexhands.utils.process_marl import get_AgentIndex
from bidexhands.algorithms.rl.ppo import PPO
from bidexhands.tasks.hand_base.vec_task import VecTaskPython, VecTaskPythonArm
from bidexhands.tasks.ur_by_pick import UrBotyardHandPick

def parse_task(args, cfg, cfg_train, sim_params, agent_index):

    # create native task and pass custom config
    device_id = args.device_id
    rl_device = args.rl_device

    cfg["seed"] = cfg_train.get("seed", -1)
    cfg_task = cfg["env"]
    cfg_task["seed"] = cfg["seed"]

    print("Python")

    try:
        task = UrBotyardHandPick(
            cfg=cfg,
            sim_params=sim_params,
            physics_engine=args.physics_engine,
            device_type=args.device,
            device_id=device_id,
            headless=args.headless,
            is_multi_agent=False)
    except NameError as e:
        print(e)
    
    env = VecTaskPython(task, rl_device)

    return task, env


def process_sarl(args, env, cfg_train, logdir):
    learn_cfg = cfg_train["learn"]
    is_testing = learn_cfg["test"]
    # is_testing = True
    # Override resume and testing flags if they are passed as parameters.
    if args.model_dir != "":
        # is_testing = True
        chkpt_path = args.model_dir

    if args.max_iterations != -1:
        cfg_train["learn"]["max_iterations"] = args.max_iterations

    project_id = "Dof6_ik_0"
    logdir = "logs/UrBotyardHandPick/ppo/" + project_id

    """Set up the algo system for training or inferencing."""
    model = PPO(vec_env=env,
              cfg_train=cfg_train,
              device=env.rl_device,
              sampler=learn_cfg.get("sampler", 'sequential'),
              log_dir=logdir,
              is_testing=is_testing,
              print_log=learn_cfg["print_log"],
              apply_reset=False,
              asymmetric=(env.num_states > 0)
              )

    if is_testing and args.model_dir != "":
        print("Loading model from {}".format(chkpt_path))
        model.test(chkpt_path)
    elif args.model_dir != "":
        print("Loading model from {}".format(chkpt_path))
        model.load(chkpt_path)

    return model


def train():
    print("Algorithm: ", args.algo)
    agent_index = get_AgentIndex(cfg)
    task, env = parse_task(args, cfg, cfg_train, sim_params, agent_index)
    runner = process_sarl(args, env, cfg_train, logdir)
    iterations = cfg_train["learn"]["max_iterations"]
    if args.max_iterations > 0:
        iterations = args.max_iterations
    runner.run(num_learning_iterations=iterations, log_interval=cfg_train["learn"]["save_interval"])

if __name__ == '__main__':
    set_np_formatting()
    args = get_args(task_name="BotyardHandPick", algo="ppo")
    cfg, cfg_train, logdir = load_cfg(args)
    sim_params = parse_sim_params(args, cfg, cfg_train)
    set_seed(cfg_train.get("seed", -1), cfg_train.get("torch_deterministic", False))
    train()
