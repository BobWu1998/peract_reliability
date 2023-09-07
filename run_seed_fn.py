import os
import pickle
import gc
import logging
from typing import List

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from rlbench import CameraConfig, ObservationConfig
from yarr.replay_buffer.wrappers.pytorch_replay_buffer import PyTorchReplayBuffer
from yarr.runners.offline_train_runner import OfflineTrainRunner
from yarr.utils.stat_accumulator import SimpleAccumulator

from helpers.custom_rlbench_env import CustomRLBenchEnv, CustomMultiTaskRLBenchEnv
import torch.distributed as dist

from agents import c2farm_lingunet_bc
from agents import peract_bc
from agents import arm
from agents.baselines import bc_lang, vit_bc_lang

from uncertainty_module.temperature_scaling import TemperatureScaler
from uncertainty_module.action_selection import ActionSelection

def run_seed(rank,
             cfg: DictConfig,
             obs_config: ObservationConfig,
             cams,
             multi_task,
             seed,
             world_size) -> None:
    dist.init_process_group("gloo",
                            rank=rank,
                            world_size=world_size)

    task = cfg.rlbench.tasks[0]
    tasks = cfg.rlbench.tasks
    print('task {}, tasks {}'.format(task, tasks))

    task_folder = task if not multi_task else 'multi'
    replay_path = os.path.join(cfg.replay.path, task_folder, cfg.method.name, 'seed%d' % seed)
    print(replay_path)
    if cfg.method.name == 'ARM':
        raise NotImplementedError("ARM is not supported yet")

    elif cfg.method.name == 'BC_LANG':
        assert cfg.ddp.num_devices == 1, "BC_LANG only supports single GPU training"
        replay_buffer = bc_lang.launch_utils.create_replay(
            cfg.replay.batch_size, cfg.replay.timesteps,
            cfg.replay.prioritisation,
            cfg.replay.task_uniform,
            replay_path if cfg.replay.use_disk else None,
            cams, cfg.rlbench.camera_resolution)

        bc_lang.launch_utils.fill_multi_task_replay(
            cfg, obs_config, rank,
            replay_buffer, tasks, cfg.rlbench.demos,
            cfg.method.demo_augmentation, cfg.method.demo_augmentation_every_n,
            cams)

        agent = bc_lang.launch_utils.create_agent(
            cams[0], cfg.method.activation, cfg.method.lr,
            cfg.method.weight_decay, cfg.rlbench.camera_resolution,
            cfg.method.grad_clip)

    elif cfg.method.name == 'VIT_BC_LANG':
        assert cfg.ddp.num_devices == 1, "VIT_BC_LANG only supports single GPU training"
        replay_buffer = vit_bc_lang.launch_utils.create_replay(
            cfg.replay.batch_size, cfg.replay.timesteps,
            cfg.replay.prioritisation,
            cfg.replay.task_uniform,
            replay_path if cfg.replay.use_disk else None,
            cams, cfg.rlbench.camera_resolution)

        vit_bc_lang.launch_utils.fill_multi_task_replay(
            cfg, obs_config, rank,
            replay_buffer, tasks, cfg.rlbench.demos,
            cfg.method.demo_augmentation, cfg.method.demo_augmentation_every_n,
            cams)

        agent = vit_bc_lang.launch_utils.create_agent(
            cams[0], cfg.method.activation, cfg.method.lr,
            cfg.method.weight_decay, cfg.rlbench.camera_resolution,
            cfg.method.grad_clip)

    elif cfg.method.name == 'C2FARM_LINGUNET_BC':
        replay_buffer = c2farm_lingunet_bc.launch_utils.create_replay(
            cfg.replay.batch_size, cfg.replay.timesteps,
            cfg.replay.prioritisation,
            cfg.replay.task_uniform,
            replay_path if cfg.replay.use_disk else None,
            cams, cfg.method.voxel_sizes,
            cfg.rlbench.camera_resolution)

        c2farm_lingunet_bc.launch_utils.fill_multi_task_replay(
            cfg, obs_config, rank,
            replay_buffer, tasks, cfg.rlbench.demos,
            cfg.method.demo_augmentation, cfg.method.demo_augmentation_every_n,
            cams, cfg.rlbench.scene_bounds,
            cfg.method.voxel_sizes, cfg.method.bounds_offset,
            cfg.method.rotation_resolution, cfg.method.crop_augmentation,
            keypoint_method=cfg.method.keypoint_method)

        agent = c2farm_lingunet_bc.launch_utils.create_agent(cfg)

    elif cfg.method.name == 'PERACT_BC':
        replay_buffer = peract_bc.launch_utils.create_replay(
            cfg.replay.batch_size, cfg.replay.timesteps,
            cfg.replay.prioritisation,
            cfg.replay.task_uniform,
            replay_path if cfg.replay.use_disk else None,
            cams, cfg.method.voxel_sizes,
            cfg.rlbench.camera_resolution)

        peract_bc.launch_utils.fill_multi_task_replay(
            cfg, obs_config, rank,
            replay_buffer, tasks, cfg.rlbench.demos,
            cfg.method.demo_augmentation, cfg.method.demo_augmentation_every_n,
            cams, cfg.rlbench.scene_bounds,
            cfg.method.voxel_sizes, cfg.method.bounds_offset,
            cfg.method.rotation_resolution, cfg.method.crop_augmentation,
            keypoint_method=cfg.method.keypoint_method)

        agent = peract_bc.launch_utils.create_agent(cfg)

    elif cfg.method.name == 'PERACT_RL':
        raise NotImplementedError("PERACT_RL is not supported yet")

    else:
        raise ValueError('Method %s does not exists.' % cfg.method.name)


    wrapped_replay = PyTorchReplayBuffer(replay_buffer, num_workers=cfg.framework.num_workers)
    stat_accum = SimpleAccumulator(eval_video_fps=30)

    cwd = os.getcwd()
    weightsdir = os.path.join(cwd, 'seed%d' % seed, 'weights')
    logdir = os.path.join(cwd, 'seed%d' % seed)

    # weightsdir = '/home/bobwu/UQ/peract_headless/peract_reliability/ckpts/multi/PERACT_BC/seed0/weights'
    temperature_scaler = TemperatureScaler(
        device = rank,
        rotation_resolution = cfg.method.rotation_resolution,
        batch_size = cfg.replay.batch_size,
        num_rotation_classes = int(360. // cfg.method.rotation_resolution),
        voxel_size = cfg.method.voxel_sizes[0],
        trans_loss_weight=cfg.method.trans_loss_weight,
        rot_loss_weight=cfg.method.rot_loss_weight,
        grip_loss_weight=cfg.method.grip_loss_weight,
        collision_loss_weight=cfg.method.collision_loss_weight,
        training=cfg.temperature.temperature_training,
        use_hard_temp = cfg.temperature.temperature_use_hard_temp,
        hard_temp = cfg.temperature.temperature_hard_temp)
    
    action_selection = ActionSelection(
            device = rank, 
            rotation_resolution = cfg.method.rotation_resolution,
            batch_size = cfg.replay.batch_size, 
            num_rotation_classes = int(360. // cfg.method.rotation_resolution), 
            voxel_size = cfg.method.voxel_sizes[0],
            temperature = cfg.temperature.temperature_hard_temp,
            alpha1 = cfg.risk.alpha1,
            alpha2 = cfg.risk.alpha2,
            alpha3 = cfg.risk.alpha3,
            alpha4 = cfg.risk.alpha4,
            tau = cfg.risk.tau,
            conf_thresh = cfg.risk.conf_thresh)

    train_runner = OfflineTrainRunner(
        agent=agent,
        wrapped_replay_buffer=wrapped_replay,
        train_device=rank,
        stat_accumulator=stat_accum,
        iterations=cfg.framework.training_iterations,
        logdir=logdir,
        logging_level=cfg.framework.logging_level,
        log_freq=cfg.framework.log_freq,
        weightsdir=weightsdir,
        num_weights_to_keep=cfg.framework.num_weights_to_keep,
        save_freq=cfg.framework.save_freq,
        tensorboard_logging=cfg.framework.tensorboard_logging,
        csv_logging=cfg.framework.csv_logging,
        load_existing_weights=cfg.framework.load_existing_weights,
        rank=rank,
        world_size=world_size,
        task_name = task,
        temperature_scaler = temperature_scaler,
        action_selection = action_selection)

    train_runner.start()

    del train_runner
    del agent
    gc.collect()
    torch.cuda.empty_cache()