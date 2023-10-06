import gc
import logging
import os
import sys

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
from typing import List

import hydra
import numpy as np
import torch
import pandas as pd
from omegaconf import DictConfig, OmegaConf, ListConfig
from rlbench import CameraConfig, ObservationConfig
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.backend import task as rlbench_task
from rlbench.backend.utils import task_file_to_task_class
from yarr.runners.independent_env_runner import IndependentEnvRunner
from yarr.utils.stat_accumulator import SimpleAccumulator

from agents import c2farm_lingunet_bc
from agents import peract_bc
from agents import arm
from agents.baselines import bc_lang, vit_bc_lang

from helpers.custom_rlbench_env import CustomRLBenchEnv, CustomMultiTaskRLBenchEnv
from helpers import utils
from helpers.utils import create_obs_config
import torch.distributed as dist

from yarr.utils.rollout_generator import RolloutGenerator
from torch.multiprocessing import Process, Manager

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from yarr.replay_buffer.wrappers.pytorch_replay_buffer import \
    PyTorchReplayBuffer

import psutil

# from uncertainty_module.src.base.calib_scaling import CalibScaler
from uncertainty_module.src.temperature_scaling.temperature_scaling import TemperatureScaler
from uncertainty_module.src.vector_scaling.vector_scaling import VectorScaler
from uncertainty_module.action_selection import ActionSelection


def eval_seed(train_cfg,
              eval_cfg,
              logdir,
              cams,
              env_device,
              multi_task,
              seed,
              env_config,
              obs_config) -> None:

    tasks = eval_cfg.rlbench.tasks
    rg = RolloutGenerator()

    if train_cfg.method.name == 'ARM':
        raise NotImplementedError('ARM not yet supported for eval.py')

    elif train_cfg.method.name == 'BC_LANG':
        agent = bc_lang.launch_utils.create_agent(
            cams[0],
            train_cfg.method.activation,
            train_cfg.method.lr,
            train_cfg.method.weight_decay,
            train_cfg.rlbench.camera_resolution,
            train_cfg.method.grad_clip)

    elif train_cfg.method.name == 'VIT_BC_LANG':
        agent = vit_bc_lang.launch_utils.create_agent(
            cams[0],
            train_cfg.method.activation,
            train_cfg.method.lr,
            train_cfg.method.weight_decay,
            train_cfg.rlbench.camera_resolution,
            train_cfg.method.grad_clip)

    elif train_cfg.method.name == 'C2FARM_LINGUNET_BC':
        agent = c2farm_lingunet_bc.launch_utils.create_agent(train_cfg)

    elif train_cfg.method.name == 'PERACT_BC':
        agent = peract_bc.launch_utils.create_agent(train_cfg)
        

    elif train_cfg.method.name == 'PERACT_RL':
        raise NotImplementedError("PERACT_RL not yet supported for eval.py")

    else:
        raise ValueError('Method %s does not exists.' % train_cfg.method.name)

    stat_accum = SimpleAccumulator(eval_video_fps=30)

    cwd = os.getcwd()
    weightsdir = os.path.join(logdir, 'weights')
    

    # ### adding support for expert demo extraction
    # cfg = train_cfg # eval_cfg
    # os.environ['MASTER_ADDR'] = cfg.ddp.master_addr
    # os.environ['MASTER_PORT'] = cfg.ddp.master_port

    # print('cfg.framework.num_workers', cfg.framework.num_workers)
    # cfg.framework.logging_level = 20
    # cfg.framework.num_workers = 0

    # logging.info('adding support for expert demo')
    # logging.info('env_config{}'.format(env_config))
    # world_size = cfg.ddp.num_devices
    # rank = world_size-1

    # task = cfg.rlbench.tasks[0]
    # tasks = cfg.rlbench.tasks

    # task_folder = task if not multi_task else 'multi'
    # logging.info('rank {}, world_size {}'.format(rank, world_size))
    # dist.init_process_group("gloo",
    #                     rank=rank,
    #                     world_size=world_size)
    # logging.info('adding support for expert demo ... 50%')
    # replay_path = os.path.join(cfg.replay.path, task_folder, cfg.method.name, 'seed%d' % seed)
    
    # replay_buffer = peract_bc.launch_utils.create_replay(
    # cfg.replay.batch_size, cfg.replay.timesteps,
    # cfg.replay.prioritisation,
    # cfg.replay.task_uniform,
    # replay_path if cfg.replay.use_disk else None,
    # cams, cfg.method.voxel_sizes,
    # cfg.rlbench.camera_resolution)

    # peract_bc.launch_utils.fill_multi_task_replay(
    #     cfg, obs_config, rank,
    #     replay_buffer, tasks, cfg.rlbench.demos,
    #     cfg.method.demo_augmentation, cfg.method.demo_augmentation_every_n,
    #     cams, cfg.rlbench.scene_bounds,
    #     cfg.method.voxel_sizes, cfg.method.bounds_offset,
    #     cfg.method.rotation_resolution, cfg.method.crop_augmentation,
    #     keypoint_method=cfg.method.keypoint_method)
    # agent = peract_bc.launch_utils.create_agent(train_cfg)
    # logging.info('agents layer {}'.format(agent))
    # wrapped_replay = PyTorchReplayBuffer(replay_buffer, num_workers=cfg.framework.num_workers)
    # ### end adding support for expert demo extraction
    wrapped_replay = None

    cfg = train_cfg
    world_size = cfg.ddp.num_devices
    rank = world_size-1
    # print('cfg.temperature.temperature_hard_temp',eval_cfg.temperature.temperature_hard_temp)
    


    print('cfg.scaler.type', eval_cfg.scaler.type)
    if eval_cfg.scaler.type == 'temperature':
        
        if eval_cfg.temperature.load_indiv_temp:
            eval_cfg.temperature.temperature_use_hard_temp = True
            single_task_name = tasks[0]
            temp_path = eval_cfg.temperature.temp_log_root + single_task_name + '/'
            eval_cfg.temperature.temperature_hard_temp = float(torch.load(temp_path + single_task_name + '_temperature.pth'))
            print(eval_cfg.temperature.temperature_hard_temp)
            
        calib_scaler = TemperatureScaler(
            calib_type = cfg.scaler.type,
            device = rank,
            rotation_resolution = cfg.method.rotation_resolution,
            batch_size = cfg.replay.batch_size,
            num_rotation_classes = int(360. // cfg.method.rotation_resolution),
            voxel_size = cfg.method.voxel_sizes[0],
            trans_loss_weight=cfg.method.trans_loss_weight,
            rot_loss_weight=cfg.method.rot_loss_weight,
            grip_loss_weight=cfg.method.grip_loss_weight,
            collision_loss_weight=cfg.method.collision_loss_weight,
            training=eval_cfg.temperature.temperature_training,
            use_hard_temp = eval_cfg.temperature.temperature_use_hard_temp,
            hard_temp = eval_cfg.temperature.temperature_hard_temp)
    else:
        calib_scaler = VectorScaler(
            calib_type = cfg.scaler.type,
            device = rank,
            rotation_resolution = cfg.method.rotation_resolution,
            batch_size = cfg.replay.batch_size,
            num_rotation_classes = int(360. // cfg.method.rotation_resolution),
            voxel_size = cfg.method.voxel_sizes[0],
            trans_loss_weight=cfg.method.trans_loss_weight,
            rot_loss_weight=cfg.method.rot_loss_weight,
            grip_loss_weight=cfg.method.grip_loss_weight,
            collision_loss_weight=cfg.method.collision_loss_weight,
            training = eval_cfg.vector.vector_training,
            training_iter = eval_cfg.vector.vector_training_iter,
            scaler_log_root = eval_cfg.vector.vector_log_root)    
        
        calib_scaler.load_parameter(task_name=tasks[0])    
    
    tau = eval_cfg.risk.tau,
    trans_conf_thresh = eval_cfg.risk.trans_conf_thresh,
    rot_conf_thresh = eval_cfg.risk.rot_conf_thresh,
    search_size = eval_cfg.risk.search_size,
    search_step = eval_cfg.risk.search_step
    print("tau:", tau)
    print("trans_conf_thresh:", trans_conf_thresh)
    print("rot_conf_thresh:", rot_conf_thresh)
    print("search_size:", search_size)
    print("search_step:", search_step) 
    
    action_selection = ActionSelection(
            device = rank, 
            rotation_resolution = cfg.method.rotation_resolution,
            batch_size = cfg.replay.batch_size, 
            num_rotation_classes = int(360. // cfg.method.rotation_resolution), 
            voxel_size = cfg.method.voxel_sizes[0],
            temperature = eval_cfg.temperature.temperature_hard_temp,
            alpha1 = eval_cfg.risk.alpha1,
            alpha2 = eval_cfg.risk.alpha2,
            alpha3 = eval_cfg.risk.alpha3,
            alpha4 = eval_cfg.risk.alpha4,
            tau = eval_cfg.risk.tau,
            trans_conf_thresh = eval_cfg.risk.trans_conf_thresh,
            rot_conf_thresh = eval_cfg.risk.rot_conf_thresh,
            search_size = eval_cfg.risk.search_size,
            search_step = eval_cfg.risk.search_step,
            log_dir = eval_cfg.risk.log_dir,
            enabled = eval_cfg.risk.enabled)

    # agent.build(training=False, device=rank, temperature_scaler=temperature_scaler, action_selection=action_selection)
    # agent.build(training=False, device=None, temperature_scaler=temperature_scaler, action_selection=action_selection)
    
    env_runner = IndependentEnvRunner(
        train_env=None,
        agent=agent,
        train_replay_buffer=None,
        num_train_envs=0,
        num_eval_envs=eval_cfg.framework.eval_envs,
        rollout_episodes=99999,
        eval_episodes=eval_cfg.framework.eval_episodes,
        training_iterations=train_cfg.framework.training_iterations,
        eval_from_eps_number=eval_cfg.framework.eval_from_eps_number,
        episode_length=eval_cfg.rlbench.episode_length,
        stat_accumulator=stat_accum,
        weightsdir=weightsdir,
        logdir=logdir,
        env_device=env_device,
        rollout_generator=rg,
        num_eval_runs=len(tasks),
        multi_task=multi_task,
        wrapped_replay=wrapped_replay,
        calib_scaler=calib_scaler,
        action_selection=action_selection)

    manager = Manager()
    save_load_lock = manager.Lock()
    writer_lock = manager.Lock()

    # evaluate all checkpoints (0, 1000, ...) which don't have results, i.e. validation phase
    if eval_cfg.framework.eval_type == 'missing':
        weight_folders = os.listdir(weightsdir)
        weight_folders = sorted(map(int, weight_folders))

        env_data_csv_file = os.path.join(logdir, 'eval_data.csv')
        if os.path.exists(env_data_csv_file):
            env_dict = pd.read_csv(env_data_csv_file).to_dict()
            evaluated_weights = sorted(map(int, list(env_dict['step'].values())))
            weight_folders = [w for w in weight_folders if w not in evaluated_weights]

        print('Missing weights: ', weight_folders)

    # pick the best checkpoint from validation and evaluate, i.e. test phase
    elif eval_cfg.framework.eval_type == 'best':
        env_data_csv_file = os.path.join(logdir, 'eval_data.csv')
        if os.path.exists(env_data_csv_file):
            env_dict = pd.read_csv(env_data_csv_file).to_dict()
            existing_weights = list(map(int, sorted(os.listdir(os.path.join(logdir, 'weights')))))
            task_weights = {}
            for task in tasks:
                weights = list(env_dict['step'].values())

                if len(tasks) > 1:
                    task_score = list(env_dict['eval_envs/return/%s' % task].values())
                else:
                    task_score = list(env_dict['eval_envs/return'].values())

                avail_weights, avail_task_scores = [], []
                for step_idx, step in enumerate(weights):
                    if step in existing_weights:
                        avail_weights.append(step)
                        avail_task_scores.append(task_score[step_idx])

                assert(len(avail_weights) == len(avail_task_scores))
                best_weight = avail_weights[np.argwhere(avail_task_scores == np.amax(avail_task_scores)).flatten().tolist()[-1]]
                task_weights[task] = best_weight

            weight_folders = [task_weights]
            print("Best weights:", weight_folders)
        else:
            raise Exception('No existing eval_data.csv file found in %s' % logdir)

    # evaluate only the last checkpoint
    elif eval_cfg.framework.eval_type == 'last':
        weight_folders = os.listdir(weightsdir)
        weight_folders = sorted(map(int, weight_folders))
        weight_folders = [weight_folders[-1]]
        print("Last weight:", weight_folders)

    # evaluate a specific checkpoint
    elif type(eval_cfg.framework.eval_type) == int:
        weight_folders = [int(eval_cfg.framework.eval_type)]
        print("Weight:", weight_folders)

    else:
        raise Exception('Unknown eval type')

    num_weights_to_eval = np.arange(len(weight_folders))
    if len(num_weights_to_eval) == 0:
        logging.info("No weights to evaluate. Results are already available in eval_data.csv")
        sys.exit(0)

    # evaluate several checkpoints in parallel
    # NOTE: in multi-task settings, each task is evaluated serially, which makes everything slow!
    split_n = utils.split_list(num_weights_to_eval, eval_cfg.framework.eval_envs)
    for split in split_n:
        processes = []
        for e_idx, weight_idx in enumerate(split):
            weight = weight_folders[weight_idx]
            env_runner.start(weight,
                              save_load_lock,
                              writer_lock,
                              env_config,
                              e_idx % torch.cuda.device_count(),
                              eval_cfg.framework.eval_save_metrics,
                              eval_cfg.cinematic_recorder)
        #     p = Process(target=env_runner.start,
        #                 args=(weight,
        #                       save_load_lock,
        #                       writer_lock,
        #                       env_config,
        #                       e_idx % torch.cuda.device_count(),
        #                       eval_cfg.framework.eval_save_metrics,
        #                       eval_cfg.cinematic_recorder))
        #     p.start()
        #     processes.append(p)
        # for p in processes:
        #     p.join()

    del env_runner
    del agent
    gc.collect()
    torch.cuda.empty_cache()

    def count_child_processes():
        current_process = psutil.Process(os.getpid())
        return len(current_process.children(recursive=True))

    print('num_processes', count_child_processes())

    def kill_child_processes():
        current_process = psutil.Process(os.getpid())
        children = current_process.children(recursive=True)  # Get all child processes
        for child in children:
            child.kill()
            try:
                child.wait(timeout=3)  # Wait up to 3 seconds for process to terminate
            except psutil.TimeoutExpired:
                print(f"Child process {child.pid} did not terminate in time.")
            
            if child.is_running():
                print(f"Child process {child.pid} is still running.")
            else:
                print(f"Child process {child.pid} terminated successfully.")


    kill_child_processes() ## !!! DIRTY FIX
    print('num_processes', count_child_processes())

@hydra.main(config_name='eval', config_path='conf')
def main(eval_cfg: DictConfig) -> None:
    
    logging.info('\n' + OmegaConf.to_yaml(eval_cfg))

    start_seed = eval_cfg.framework.start_seed
    logdir = os.path.join(eval_cfg.framework.logdir,
                                eval_cfg.rlbench.task_name,
                                eval_cfg.method.name,
                                'seed%d' % start_seed)

    train_config_path = os.path.join(logdir, 'config.yaml')
    print('train_config_path', train_config_path)
    if os.path.exists(train_config_path):
        with open(train_config_path, 'r') as f:
            train_cfg = OmegaConf.load(f)
    else:
        raise Exception("Missing seed%d/config.yaml" % start_seed)

    env_device = utils.get_device(eval_cfg.framework.gpu)
    logging.info('Using env device %s.' % str(env_device))

    gripper_mode = Discrete()
    arm_action_mode = EndEffectorPoseViaPlanning()
    action_mode = MoveArmThenGripper(arm_action_mode, gripper_mode)

    task_files = [t.replace('.py', '') for t in os.listdir(rlbench_task.TASKS_PATH)
                  if t != '__init__.py' and t.endswith('.py')]
    eval_cfg.rlbench.cameras = eval_cfg.rlbench.cameras if isinstance(
        eval_cfg.rlbench.cameras, ListConfig) else [eval_cfg.rlbench.cameras]
    obs_config = utils.create_obs_config(eval_cfg.rlbench.cameras,
                                         eval_cfg.rlbench.camera_resolution,
                                         train_cfg.method.name)

    if eval_cfg.cinematic_recorder.enabled:
        obs_config.record_gripper_closing = True

    # single-task or multi-task
    if len(eval_cfg.rlbench.tasks) > 1:
        tasks = eval_cfg.rlbench.tasks
        multi_task = True

        task_classes = []
        for task in tasks:
            if task not in task_files:
                raise ValueError('Task %s not recognised!.' % task)
            task_classes.append(task_file_to_task_class(task))

        env_config = (task_classes,
                      obs_config,
                      action_mode,
                      eval_cfg.rlbench.demo_path,
                      eval_cfg.rlbench.episode_length,
                      eval_cfg.rlbench.headless,
                      eval_cfg.framework.eval_episodes,
                      train_cfg.rlbench.include_lang_goal_in_obs,
                      eval_cfg.rlbench.time_in_state,
                      eval_cfg.framework.record_every_n)
    else:
        task = eval_cfg.rlbench.tasks[0]
        multi_task = False

        if task not in task_files:
            raise ValueError('Task %s not recognised!.' % task)
        task_class = task_file_to_task_class(task)

        env_config = (task_class,
                      obs_config,
                      action_mode,
                      eval_cfg.rlbench.demo_path,
                      eval_cfg.rlbench.episode_length,
                      eval_cfg.rlbench.headless,
                      train_cfg.rlbench.include_lang_goal_in_obs,
                      eval_cfg.rlbench.time_in_state,
                      eval_cfg.framework.record_every_n)

    logging.info('Evaluating seed %d.' % start_seed)
    eval_seed(train_cfg,
              eval_cfg,
              logdir,
              eval_cfg.rlbench.cameras,
              env_device,
              multi_task, start_seed,
              env_config,
              obs_config)

if __name__ == '__main__':
    
    main()
