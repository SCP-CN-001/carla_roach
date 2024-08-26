#!/usr/bin/env python

# Copyright (c) 2018-2020 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides the ScenarioManager implementations.
It must not be modified and is for reference only!
"""

from __future__ import print_function

import os
from collections import deque

import gym
import torch
import torch as th
import pdb
import signal
import sys
import time
from torch.nn import functional as F
import numpy as np
import py_trees
import carla
import threading
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.timer import GameTime
from srunner.scenariomanager.watchdog import Watchdog

from leaderboard.autoagents.agent_wrapper import AgentWrapperFactory, AgentError
from leaderboard.envs.sensor_interface import SensorReceivedNoData
from leaderboard.utils.result_writer import ResultOutputProvider
from leaderboard.obs_manager.obs_manager_handler import ObsManagerHandler

from .gym_utils.traffic_light import TrafficLightHandler
from .gym_utils import config_utils
from .gym_utils import transforms as trans_utils
from ..rl_birdview.models.ppo_buffer import PpoBuffer
from .gym_utils.config_utils import load_entry_point
# add Carla 0.9.14 path
Carla_root="/home/vci-1/XT/leaderboard_roach0818/ramble/roach_v2/CARLA_Leaderboard_2.0/CARLA_Leaderboard_20/"
sys.path.append(Carla_root + "PythonAPI/carla/agents/navigation/")
# global planner of Carla0.9.14
from global_route_planner import GlobalRoutePlanner
from route_manipulation import location_route_to_gps,downsample_route
from leaderboard.utils.route_parser import RouteParser
from .criteria import blocked, collision, outside_route_lane, route_deviation, run_stop_sign
from .criteria import encounter_light, run_red_light
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
path = "/home/vci-1/XT/leaderV2/leaderboard/leaderboard/output/" + TIMESTAMP
writer = SummaryWriter(path)
class ScenarioManager(object):

    """
    Basic scenario manager class. This class holds all functionality
    required to start, run and stop a scenario.

    The user must not modify this class.

    To use the ScenarioManager:
    1. Create an object via manager = ScenarioManager()
    2. Load a scenario via manager.load_scenario()
    3. Trigger the execution of the scenario manager.run_scenario()
       This function is designed to explicitly control start and end of
       the scenario execution
    4. If needed, cleanup with manager.stop_scenario()
    """

    def __init__(self, timeout, statistics_manager, debug_mode=0):
        """
        Setups up the parameters, which will be filled at load_scenario()
        """
        self.route_index = None
        self.scenario = None
        self.scenario_tree = None
        self.ego_vehicles = None
        self.other_actors = None
        self._about_to_stop = False
        self._debug_mode = debug_mode
        self._agent_wrapper = None
        self._running = False
        self._timestamp_last_run = 0.0
        self._timeout = float(timeout)
        self._last_dones = None
        self.scenario_duration_system = 0.0
        self.scenario_duration_game = 0.0
        self.start_system_time = 0.0
        self.start_game_time = 0.0
        self.end_system_time = 0.0
        self.end_game_time = 0.0
        self.n_steps_total = 8192
        self.n_steps = 8192
        self._watchdog = None
        self._agent_watchdog = None
        self._scenario_thread = None
        self._policy = None
        self._statistics_manager = statistics_manager
        self._last_obs = None
        self.gamma = 0.99
        self.gae_lambda = 0.9
        self.num_envs = 1
        # gym state_space
        birdview_space = gym.spaces.Box(low=0, high=255, shape=(15, 192, 192), dtype=np.uint8)
        state_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        self.observation_space = {'birdview': birdview_space, 'state': state_space}
        # gym action_space
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,))
        self.buffer = PpoBuffer(self.n_steps, self.observation_space, self.action_space,
                                gamma=self.gamma, gae_lambda=self.gae_lambda, n_envs=self.num_envs)
        self.n_epochs = 20
        self.learning_rate = 1e-5
        self.batch_size = 256
        self.clip_range = 0.2
        self.episodes = 0
        self.vf_coef = 0.5
        self.explore_coef = 0.05
        self.ent_coef = 0.05
        self.vf_coef = 0.5
        self.max_grad_norm = 0.5
        self.target_kl = 0.01
        self.update_adv = False
        self.lr_schedule_step = None
        self.clip_range_vf = None
        # train config
        self._train_cfg = {'entry_point': 'leaderboard.rl_birdview.models.ppo:PPO',
                             'kwargs': {'learning_rate': 1e-05,
                              'n_steps_total': 12288,
                              'batch_size': 256,
                              'n_epochs': 20,
                              'gamma': 0.99,
                              'gae_lambda': 0.9,
                              'clip_range': 0.2,
                              'clip_range_vf': None,
                              'ent_coef': 0.01,
                              'explore_coef': 0.05,
                              'vf_coef': 0.5,
                              'max_grad_norm': 0.5,
                              'target_kl': 0.01,
                              'update_adv': False,
                              'lr_schedule_step': 8}}
        self._policy_kwargs = {'policy_head_arch': [256, 256],
                                 'value_head_arch': [256, 256],
                                 'features_extractor_entry_point': 'leaderboard.rl_birdview.models.torch_layers:XtMaCNN',
                                 'features_extractor_kwargs': {'states_neurons': [256, 256]},
                                 'distribution_entry_point': 'leaderboard.rl_birdview.models.distributions:BetaDistribution',
                                 'distribution_kwargs': {'dist_init': None}}
        # reward setting

        self._maxium_speed = 6.0
        self._tl_offset = -1.6
        self._last_steer = 0.0
        self._global_route = []
        self._global_plan_gps = []
        self._global_plan_world_coord = []
        self._list_stop_signs = []
        ppo_agent = self.learn_ppo_init(2021, 2021)
        self._proximity_threshold = 50
        self._waypoint_step = 1.0
        self.info_route_completion = {}
        self._info_criteria = {}
        # use the callback_id inside the signal handler to allow external interrupts
        signal.signal(signal.SIGINT, self.signal_handler)
        self._route_completed = 0.0
        self._route_length = 0.0
        ROUTES = os.getenv('ROUTES')
        ROUTES_SUBSET = os.getenv('ROUTES_SUBSET')
        self.start_time_store = time.strftime('%Y_%m_%d_%H:_%M_%S', time.localtime(time.time()))
        # V2 route
        self.route_configuration = RouteParser.parse_routes_file(ROUTES, ROUTES_SUBSET)
        self.route_configurations = self.route_configuration[0].keypoints_roach

        # roach routes
        # self.route_descriptions_dict = config_utils.parse_routes_file(
        #     os.getenv('Roach_ROUTES_SUBSET'))

        # V2 waypoints
        target_transforms = self.route_configurations[1:] #not contain start_point
        self._vehicle_stuck_step = 100 #100
        self._vehicle_stuck_counter = 0
        self._speed_queue = deque(maxlen=10)
        # roach waypoints
        # target_transforms1 = self.route_descriptions_dict[0]['ego_vehicles']['hero'][1:]

        self._target_transforms = target_transforms

        self.criteria_route_deviation = route_deviation.RouteDeviation()
        self.criteria_blocked = blocked.Blocked()
       # self.repetitions = 0



    def signal_handler(self, signum, frame):
        """
        Terminate scenario ticking when receiving a signal interrupt
        """
        if self._agent_watchdog and not self._agent_watchdog.get_status():
            raise RuntimeError("Agent took longer than {}s to send its command".format(self._timeout))
        elif self._watchdog and not self._watchdog.get_status():
            raise RuntimeError("The simulation took longer than {}s to update".format(self._timeout))
        self._running = False

    def cleanup(self,reload=True):
        """
        Reset all parameters
        """
        self._timestamp_last_run = 0.0
        self.scenario_duration_system = 0.0
        self.scenario_duration_game = 0.0
        self.start_system_time = 0.0
        self.start_game_time = 0.0
        self.end_system_time = 0.0
        self.end_game_time = 0.0
        if reload:
            self._spectator = None
            self._watchdog = None
            self._agent_watchdog = None
        else:
            self._running = False
            self._scenario_thread.join()
            self.scenario.build_scenarios(self.ego_vehicles[0])
            CarlaDataProvider.set_runtime_init_mode(True)
        self.actor_List = None
        self.vehicle_list = None
        self.walker_list = None
        self._map = None
        self.criteria_collision = None
        self.criteria_light = None
        self.criteria_encounter_light = None
        self.criteria_stop = None
        self.criteria_outside_route_lane = None
        self._speed_queue.clear()
        self._vehicle_stuck_counter = 0
    def get_init_obs(self,timestep):
        # obtain raw data (Roach format)
        self.obs_instance = ObsManagerHandler(self.ego_vehicles[0])
        self.obs_instance.reset(self.ego_vehicles[0])
        obs_dict = self.obs_instance.get_observation(timestep)

        # process the observation into standard shape (Roach)
        obs = self.process_obs(obs_dict['hero'], ['control', 'vel_xy'], train=False)
        return obs
    def load_scenario(self, scenario, agent, route_index, rep_number):
        """
        Load a new scenario
        """

        GameTime.restart()
        self._agent_wrapper = AgentWrapperFactory.get_wrapper(agent)
        self.route_index = route_index
        self.scenario = scenario
        self.scenario_tree = scenario.scenario_tree
        self.ego_vehicles = scenario.ego_vehicles
        self.other_actors = scenario.other_actors
        self.repetition_number = rep_number

        self._spectator = CarlaDataProvider.get_world().get_spectator()
        self._agent_wrapper.setup_sensors(self.ego_vehicles[0])

        self.init_roach()
        timestep = 0
        #ego_vehicles0 = CarlaDataProvider.get_ego()

        obs = self.get_init_obs(timestep)

        # return the initial obs_dict
        return obs
    def init_roach(self):
        # New init
        self.actor_List = CarlaDataProvider.get_world().get_actors()
        self.vehicle_list = self.actor_List.filter("*vehicle*")
        self.walker_list = self.actor_List.filter("*walker*")
        TrafficLightHandler.reset(CarlaDataProvider.get_world())
        self._map = CarlaDataProvider.get_world().get_map()
        self.criteria_collision = collision.Collision(self.ego_vehicles[0], CarlaDataProvider.get_world())
        self.criteria_light = run_red_light.RunRedLight(self._map)
        self.criteria_encounter_light = encounter_light.EncounterLight()
        self.criteria_stop = run_stop_sign.RunStopSign(CarlaDataProvider.get_world())
        self.criteria_outside_route_lane = outside_route_lane.OutsideRouteLane(self._map, self.ego_vehicles[0].get_location())
        self._last_route_location = self.ego_vehicles[0].get_location()

        # global planner of Carla0.9.14
        self._planner = GlobalRoutePlanner(self._map,1.0)
        # To print the scenario tree uncomment the next line
        # py_trees.display.render_dot_tree(self.scenario_tree)
        # start time
        snap_shot = CarlaDataProvider.get_world().get_snapshot()

        # time log (Roach)
        self._timestamp = {
            'step': 0,
            'frame': snap_shot.timestamp.frame,
            'relative_wall_time': 0.0,
            'wall_time': snap_shot.timestamp.platform_timestamp,
            'relative_simulation_time': 0.0,
            'simulation_time': snap_shot.timestamp.elapsed_seconds,
            'start_frame': snap_shot.timestamp.frame,
            'start_wall_time': snap_shot.timestamp.platform_timestamp,
            'start_simulation_time': snap_shot.timestamp.elapsed_seconds
        }

    def build_scenarios_loop(self, debug):
        """
        Keep periodically trying to start the scenarios that are close to the ego vehicle
        Additionally, do the same for the spawned vehicles
        """
        while self._running:
            self.scenario.build_scenarios(self.ego_vehicles[0], debug=debug)
            self.scenario.spawn_parked_vehicles(self.ego_vehicles[0])
            time.sleep(1)
    def learn_ppo_init(self, total_timesteps, seed):
        self._policy_class = load_entry_point('leaderboard.rl_birdview.models.ppo_policy:PpoPolicy')
        if self._policy is None:
            self._policy = self._policy_class(self.observation_space, self.action_space, **self._policy_kwargs)
        model_class = load_entry_point(self._train_cfg['entry_point'])
        ppo_agent = model_class(self._policy, **self._train_cfg['kwargs'])
        return ppo_agent
    def run_scenario(self,obs_dict,episodes):
        """
        Trigger the start of the scenario and wait for it to finish/fail
        """

        self.episodes = episodes
        self._trace_route_to_global_target()
        self._last_obs = obs_dict
        self.start_system_time = time.time()
        self.start_game_time = GameTime.get_time()

        # detects if the simulation is down
        self._watchdog = Watchdog(self._timeout)
        self._watchdog.start()

        # stop the agent from freezing the simulation
        self._agent_watchdog = Watchdog(self._timeout)
        self._agent_watchdog.start()

        self._running = True

        # ppoBufferSamples, Thread for build_scenarios
        self._scenario_thread = threading.Thread(target=self.build_scenarios_loop, args=(self._debug_mode > 0, ))
        self._scenario_thread.start()
       # self.buffer.reset()

        # init ppo

        self._policy = self._policy.train()
        self._last_dones = np.zeros((1,), dtype=np.bool)

        # collect data and learn PPO
        continue_training = self.collect_rollouts(self.buffer, self.n_steps)
        self.train()


        if self.episodes > 10:
            model_folder = os.getenv('LEADERBOARD_ROOT')+'/leaderboard/model_pth/' + self.start_time_store + '/'
            if not os.path.exists(model_folder):
                os.makedirs(model_folder)
            save_path = os.path.join(model_folder, 'roach_'+str(self.episodes)+'.pth')
            torch.save(self._policy,save_path)


    def _tick_scenario(self,ppo_actions,start_timestamp):
        """
        Run next tick of scenario and the agent and tick the world.
        """
        # time log (Roach)
        self.distance_traveled = self._truncate_global_route_till_local_target()
        route_completed = self._is_route_completed()
        if self._running and self.get_running_status():
            CarlaDataProvider.get_world().tick(self._timeout)
        snap_shot = CarlaDataProvider.get_world().get_snapshot()
        self._timestamp['step'] = snap_shot.timestamp.frame-self._timestamp['start_frame']
        self._timestamp['frame'] = snap_shot.timestamp.frame
        self._timestamp['wall_time'] = snap_shot.timestamp.platform_timestamp
        self._timestamp['relative_wall_time'] = self._timestamp['wall_time'] - self._timestamp['start_wall_time']
        self._timestamp['simulation_time'] = snap_shot.timestamp.elapsed_seconds
        self._timestamp['relative_simulation_time'] = self._timestamp['simulation_time'] - self._timestamp['start_simulation_time']

        timestamp = snap_shot.timestamp
        if self._timestamp_last_run < timestamp.elapsed_seconds and self._running:
            self._timestamp_last_run = timestamp.elapsed_seconds

            self._watchdog.update()
            # update game time and actor information
            GameTime.on_carla_tick(timestamp)
            CarlaDataProvider.on_carla_tick()
            self._watchdog.pause()

            try:
                self._agent_watchdog.resume()
                self._agent_watchdog.update()
                ego_action_init_ppo = self._agent_wrapper()
               # pdb.set_trace()
                self._agent_watchdog.pause()

            # special exception inside the agent that isn't caused by the agent
            except SensorReceivedNoData as e:
                raise RuntimeError(e)

            except Exception as e:
                raise AgentError(e)

            self._watchdog.resume()
            #throttle and brake
            if ppo_actions[0][0] >= 0:
                ego_action_init_ppo.throttle = np.float(abs(ppo_actions[0][0]))
                ego_action_init_ppo.brake = np.float(0)
            else:
                ego_action_init_ppo.brake = 0.3 * np.float(abs(ppo_actions[0][0]))
                ego_action_init_ppo.throttle = np.float(0)
            #steer
            ego_action_init_ppo.steer = np.float(ppo_actions[0][1])
            print(f"ego_vehicle_action: throttle:{ego_action_init_ppo.throttle},steer:{ego_action_init_ppo.steer},brake:{ego_action_init_ppo.brake}")

            # ppo controlled ego vehicle
            self.ego_vehicles[0].apply_control(ego_action_init_ppo)

            # Tick scenario. Add the ego control to the blackboard in case some behaviors want to change it
            py_trees.blackboard.Blackboard().set("AV_control", ego_action_init_ppo, overwrite=True)
            self.scenario_tree.tick_once()

            if self._debug_mode > 1:
                self.compute_duration_time()

                # Update live statistics
                self._statistics_manager.compute_route_statistics(
                    self.route_index,
                    self.scenario_duration_system,
                    self.scenario_duration_game,
                    failure_message=""
                )
                self._statistics_manager.write_live_results(
                    self.route_index,
                    self.ego_vehicles[0].get_velocity().length(),
                    ego_action_init_ppo,
                    self.ego_vehicles[0].get_location()
                )

            if self._debug_mode > 2:
                print("\n")
                print("\n")
                py_trees.display.print_ascii_tree(self.scenario_tree, show_status=True)
                sys.stdout.flush()

            if self.scenario_tree.status != py_trees.common.Status.RUNNING:
                self._running = False

            ego_trans = self.ego_vehicles[0].get_transform()
            self._spectator.set_transform(carla.Transform(ego_trans.location + carla.Location(z=70),
                                                          carla.Rotation(pitch=-90)))

        # info tick log (Roach)
        info_blocked = self.criteria_blocked.tick(self.ego_vehicles[0], self._timestamp)
        info_collision = self.criteria_collision.tick(self.ego_vehicles[0], self._timestamp)
        info_light = self.criteria_light.tick(self.ego_vehicles[0], self._timestamp)
        # info_encounter_light = self.criteria_encounter_light.tick(self.ego_vehicles[0], self._timestamp)
        info_stop = self.criteria_stop.tick(self.ego_vehicles[0], self._timestamp)
        info_outside_route_lane = self.criteria_outside_route_lane.tick(self.ego_vehicles[0], self._timestamp, self.distance_traveled)
        self.info_route_deviation = self.criteria_route_deviation.tick(
            self.ego_vehicles[0], self._timestamp, self._global_route[0][0], self.distance_traveled, self._route_length)
        self.info_route_completion = {
            'step': self._timestamp['step'],
            'simulation_time': self._timestamp['relative_simulation_time'],
            'route_completed_in_m': self._route_completed,
            'route_length_in_m': self._route_length,
            'is_route_completed': route_completed
        }
        self._info_criteria = {
            'route_completion': self.info_route_completion,
            'outside_route_lane': info_outside_route_lane,
            'route_deviation': self.info_route_deviation,
            'blocked': info_blocked,
            'collision': info_collision,
            'run_red_light': info_light,
            'encounter_light':0,
            'run_stop_sign': info_stop
        }
        
        done, timeout, terminal_reward, terminal_debug = self.define_terminal(start_timestamp)
        reward, reward_debug = self.defeine_reward(terminal_reward)

	
        #reward = 0
        timestep = 0


        #ego_vehicles0 = self.ego_vehicles[0]
        #self.obs_instance = ObsManagerHandler(ego_vehicles0)
        next_obs_dict = self.obs_instance.get_observation(timestep)
        # roach format observation
        next_obs = self.process_obs(next_obs_dict['hero'], ['control', 'vel_xy'],train=False)
        return next_obs,reward,done
    def _tick_scenario_record(self):
        """
        Run next tick of scenario and the agent and tick the world.
        """
        if self._running and self.get_running_status():
            CarlaDataProvider.get_world().tick(self._timeout)

        timestamp = CarlaDataProvider.get_world().get_snapshot().timestamp

        if self._timestamp_last_run < timestamp.elapsed_seconds and self._running:
            self._timestamp_last_run = timestamp.elapsed_seconds

            self._watchdog.update()
            # Update game time and actor information
            GameTime.on_carla_tick(timestamp)
            CarlaDataProvider.on_carla_tick()
            self._watchdog.pause()

            try:
                self._agent_watchdog.resume()
                self._agent_watchdog.update()
                ego_action = self._agent_wrapper()
                self._agent_watchdog.pause()

            # Special exception inside the agent that isn't caused by the agent
            except SensorReceivedNoData as e:
                raise RuntimeError(e)

            except Exception as e:
                raise AgentError(e)

            self._watchdog.resume()
            self.ego_vehicles[0].apply_control(ego_action)

            # Tick scenario. Add the ego control to the blackboard in case some behaviors want to change it
            if not self._about_to_stop:
                py_trees.blackboard.Blackboard().set("AV_control", ego_action, overwrite=True)
                self.scenario_tree.tick_once()

            if self._debug_mode > 1:
                self.compute_duration_time()

                # Update live statistics
                self._statistics_manager.compute_route_statistics(
                    self.route_index,
                    self.scenario_duration_system,
                    self.scenario_duration_game,
                    failure_message="",
                )
                self._statistics_manager.write_live_results(
                    self.route_index,
                    self.ego_vehicles[0].get_velocity().length(),
                    ego_action,
                    self.ego_vehicles[0].get_location(),
                )

            if self._debug_mode > 2:
                print("\n")
                py_trees.display.print_ascii_tree(self.scenario_tree, show_status=True)
                sys.stdout.flush()

            # when behaviour trees get failure state, tick once before the stop to pass terminal state to the agent
            if self._about_to_stop:
                self._running = False
                self._about_to_stop = False

            if self.scenario_tree.status != py_trees.common.Status.RUNNING and self._running:
                self._about_to_stop = True

            ego_trans = self.ego_vehicles[0].get_transform()
            self._spectator.set_transform(
                carla.Transform(
                    ego_trans.location + carla.Location(z=30), carla.Rotation(pitch=-90)
                )
            )
    def collect_rollouts(self,rollout_buffer,n_rollout_steps):
        n_steps = 0
        rollout_buffer.reset()
        done = False
        total_reward = 0

        # collect rollouts   run_step
        while  n_steps < n_rollout_steps:
            ppo_actions, values, log_probs, mu, sigma, _ = self._policy.forward(self._last_obs)
            #values = log_probs = mu = sigma = 0.5
           # ppo_actions[0][0] = 0.2 #[[0.5,0.0]]
         #   ppo_actions = [[0.2,-0.15]]
            #ppo_actions = [[0.4, 0.0]]
            start_timestamp = CarlaDataProvider.get_world().get_snapshot().timestamp

            # ppo interacts with the environment
            next_obs_dict,rewards, done = self._tick_scenario(ppo_actions,start_timestamp)

            self.buffer.add(next_obs_dict, ppo_actions, rewards, done, values, log_probs, mu, sigma)
            self._last_obs = next_obs_dict
            self._last_dones = done
            # terminate this step
            if self._last_dones:
                self.obs_instance.clean()
                self.criteria_collision.clean()
               # self.criteria_blocked.clean()

              #  self.criteria_light.clean()
          #      self.criteria_stop.clean()
              #  self.criteria_outside_route_lane.clean()
             #   self.criteria_route_deviation.clean()
                break
            total_reward += rewards

            n_steps += 1
        writer.add_scalar('reward/reward_episode',total_reward,self.episodes)
        print(f"{self.episodes}total_reward:{total_reward}")
        # update ppo
        last_values = self._policy.forward_value(self._last_obs)
        self.buffer.compute_returns_and_advantage(last_values, dones=self._last_dones)

        return True
    def process_obs(self,obs, input_states, train=True):

        # if 'speed' in input_states:
        #     state_list.append(obs['speed']['speed_xy'])
        # if 'speed_limit' in input_states:
        #     state_list.append(obs['control']['speed_limit'])
        # if 'control' in input_states:
        #     state_list.append(obs['control']['throttle'])
        #     state_list.append(obs['control']['steer'])
        #     state_list.append(obs['control']['brake'])
        #     state_list.append(obs['control']['gear'] / 5.0)
        # if 'acc_xy' in input_states:
        #     state_list.append(obs['velocity']['acc_xy'])
        # if 'vel_xy' in input_states:
        #     state_list.append(obs['velocity']['vel_xy'])
        # if 'vel_ang_z' in input_states:
        #     state_list.append(obs['velocity']['vel_ang_z'])
        state_list = []
        control = self.ego_vehicles[0].get_control()
        throttle = np.array([control.throttle], dtype=np.float32)
        steer = np.array([control.steer], dtype=np.float32)
        brake = np.array([control.brake], dtype=np.float32)
        gear = np.array([control.gear], dtype=np.float32)
        ev_transform = self.ego_vehicles[0].get_transform()
        vel_w = self.ego_vehicles[0].get_velocity()
        vel_ev = trans_utils.vec_global_to_ref(vel_w, ev_transform.rotation)
        vel_xy = np.array([vel_ev.x, vel_ev.y], dtype=np.float32)
        state_list.append(throttle)
        state_list.append(steer)
        state_list.append(brake)
        state_list.append(gear)
        state_list.append(vel_xy)
        state = np.concatenate(state_list)
       # pdb.set_trace()
        birdview = obs['birdview']['masks']

        if not train:
            birdview = np.expand_dims(birdview, 0)
            state = np.expand_dims(state, 0)

        obs_dict = {
            'state': state.astype(np.float32),
            'birdview': birdview
        }
        return obs_dict
    def train(self):
        for param_group in self._policy.optimizer.param_groups:
            param_group["lr"] = self.learning_rate

        entropy_losses, exploration_losses, pg_losses, value_losses, losses = [], [], [], [], []
        clip_fractions = []
        approx_kl_divs = []
        # train for gradient_steps epochs
        epoch = 0
        data_len = int(self.buffer.buffer_size * self.buffer.n_envs / self.batch_size)
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            self.buffer.start_caching(self.batch_size)
            # while self.buffer.sample_queue.qsize() < 3:
                # time.sleep(0.01)
            for i in range(data_len):

                if self.buffer.sample_queue.empty():
                    while self.buffer.sample_queue.empty():
                        # print(f'buffer_empty: {self.buffer.sample_queue.qsize()}')
                        time.sleep(0.01)
                rollout_data = self.buffer.sample_queue.get()

                values, log_prob, entropy_loss, exploration_loss, distribution = self._policy.evaluate_actions(
                    rollout_data.observations, rollout_data.actions, rollout_data.exploration_suggests,
                    detach_values=False)
                # Normalize advantage
                advantages = rollout_data.advantages
                # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                clip_fraction = th.mean((th.abs(ratio - 1) > self.clip_range).float()).item()
               # clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the different between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(values - rollout_data.old_values,
                                                                     -self.clip_range_vf, self.clip_range_vf)
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)

                loss = policy_loss + self.vf_coef * value_loss \
                    + self.ent_coef * entropy_loss + self.explore_coef * exploration_loss
                writer.add_scalar('loss_all//episodes', loss.item(), self.episodes)
             #   losses.append(loss.item())
                # Detection policy update
                if i % 400 == 0:
                    print(f"policy_loss:{loss}")
                writer.add_scalar('policy_loss/episodes', policy_loss.item(),self.episodes)
                writer.add_scalar('value_loss/episodes', value_loss.item(), self.episodes)
                writer.add_scalar('entropy_loss/episodes', entropy_loss.item(), self.episodes)
                writer.add_scalar('exploration_loss/episodes', exploration_loss.item(), self.episodes)
                # pg_losses.append(policy_loss.item())
                # value_losses.append(value_loss.item())
                # entropy_losses.append(entropy_loss.item())
                # exploration_losses.append(exploration_loss.item())
                # Optimization step
                self._policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self._policy.parameters(), self.max_grad_norm)
                self._policy.optimizer.step()

                with th.no_grad():
                    # old_distribution = self._policy.action_dist.proba_distribution(
                    #     rollout_data.old_mu, rollout_data.old_sigma)
                    old_distribution = self._policy.action_dist.proba_distribution(
                        1, 1)
                    kl_div = th.distributions.kl_divergence(old_distribution.distribution, distribution)
                writer.add_scalar('approx_kl_divs/episodes', kl_div.mean().item(),self.episodes)
                approx_kl_divs.append(kl_div.mean().item())

            if self.target_kl is not None and np.mean(approx_kl_divs) > 1.5 * self.target_kl:
                if self.lr_schedule_step is not None:
                    self.kl_early_stop += 1
                    if self.kl_early_stop >= self.lr_schedule_step:
                        self.learning_rate *= 0.5
                        self.kl_early_stop = 0
                break

            # update advantages
            if self.update_adv:
                self.buffer.update_values(self._policy)
                last_values = self._policy.forward_value(self._last_obs)
                self.buffer.compute_returns_and_advantage(last_values, dones=self._last_dones)


        # logs (roach)
        # self.train_debug = {
        #     "train/entropy_loss": np.mean(entropy_losses),
        #     "train/exploration_loss": np.mean(exploration_losses),
        #     "train/policy_gradient_loss": np.mean(pg_losses),
        #     "train/value_loss": np.mean(value_losses),
        #     "train/last_epoch_kl": np.mean(approx_kl_divs),
        #     "train/clip_fraction": np.mean(clip_fractions),
        #     "train/loss": np.mean(losses),
        #     "train/clip_range": self.clip_range,
        #     "train/train_epoch": epoch,
        #     "train/learning_rate": self.learning_rate
        # }        print(f"self._speed_queue:{self._speed_queue}")
        #         print(f"self._vehicle_stuck_counter:{self._vehicle_stuck_counter}")
    def define_terminal(self,start_timestamp):

        terminal_debug = 0
        ev_vel = self.ego_vehicles[0].get_velocity()
        ev_speed = np.linalg.norm(np.array([ev_vel.x, ev_vel.y]))
        self._speed_queue.append(ev_speed)

        hazard_vehicle_loc = self._is_vehicle_hazard(self.vehicle_list)
        hazard_ped_loc = self._is_walker_hazard(self.walker_list)
        light_state, light_loc, _ = TrafficLightHandler.get_light_state(self.ego_vehicles[0],
                                                         offset=self._tl_offset, dist_threshold=18.0)
        is_free_road = (hazard_vehicle_loc is None) and (hazard_ped_loc is None) \
            and (light_state is None or light_state == carla.TrafficLightState.Green)
        c_vehicle_stuck = self._vehicle_stuck_counter >= self._vehicle_stuck_step

        if is_free_road and np.mean(self._speed_queue) < 0.002: #0.04:#0.002: #0.04:#1.0:
            self._vehicle_stuck_counter += 1
        if np.mean(self._speed_queue) >= 1.0:
            self._vehicle_stuck_counter = 0
        # Done condition 1: route completed
        c_route = self._info_criteria['route_completion']['is_route_completed']
        # Done condition 2: blocked
        # Done condition 2: blocked
        c_blocked = self._info_criteria['blocked'] is not None
        # Done condition 3: route_deviation
        c_route_deviation = self._info_criteria['route_deviation'] is not None
        # Done condition 4: timeout
        self._max_time = 300 #80 #50
        if self._max_time is not None:
            timeout = self._timestamp['relative_simulation_time'] > self._max_time
        else:
            timeout = False
        # Done condition 5: collisionc
        c_collision = self._info_criteria['collision'] is not None
        if c_route:
            print(f"\033[1m> DONE Fail_c_route:{c_route}\033[0m")
        if c_blocked:
            print(f"\033[1m> DONE Fail_c_blocked:{c_blocked}\033[0m")
        if c_route_deviation:
            print(f"\033[1m> DONE Fail_c_route_deviation:{c_route_deviation}\033[0m")
        if timeout:
            print(f"\033[1m> DONE Fail_timeout:{timeout}\033[0m")
        if c_vehicle_stuck:
            print(f"\033[1m> DONE Fail_c_vehicle_stuck:{c_vehicle_stuck}\033[0m")

        done = c_route or c_blocked or c_route_deviation or timeout or c_collision or c_vehicle_stuck
        terminal_reward = 0.0
        if done:
            terminal_reward = -1.0
        if c_collision:
            print(f"\033[1m> DONE Fail_c_collision:{c_collision}\033[0m")
            terminal_reward -= ev_speed
        return done, 0, terminal_reward, terminal_debug

    def defeine_reward(self,terminal_reward):
        ##
        start_time = time.time()
        # function of Carla0.9.14
        ev_transform = self.ego_vehicles[0].get_transform()
        ev_control = self.ego_vehicles[0].get_control()
        ev_vel = self.ego_vehicles[0].get_velocity()
        ev_speed = np.linalg.norm(np.array([ev_vel.x, ev_vel.y]))
        ##
     #   print(f"Time for getting transform: {time.time() - start_time} seconds") #ok
        # action reward
        if abs(ev_control.steer - self._last_steer) > 0.01:
            r_action = -0.1
        else:
            r_action = 0.0
        self._last_steer = ev_control.steer
###
       # print(f"Time for action reward: {time.time() - start_ti me} seconds") #ok
        # desired_speed reward  setting up the
        # actor_List = CarlaDataProvider.get_world().get_actors()
        # vehicle_list = actor_List.filter("*vehicle*")
        # walker_list = actor_List.filter("*walker*")
       # lights_list = actor_List.filter("*traffic_light*")
       # print(f"Time for actor_List: {time.time() - start_time} seconds")
       #  for _actor in actor_List:
       #      if 'traffic.stop' in _actor.type_id:
       #          self._list_stop_signs.append(_actor)

        # refactoring surrounding vehicle and pedestrian recognition
        nearest_vehicle = self._is_vehicle_hazard(self.vehicle_list)
        nearest_walker = self._is_walker_hazard(self.walker_list)

      #  print(f"Time for nearest_vehicle: {time.time() - start_time} seconds")
       # nearest_vehicle, min_distance_v = self.get_nearest_vehicle(self.ego_vehicles[0],vehicle_list,max_distance=9.5)
       # nearest_walker, min_distance_w = self.get_nearest_vehicle(self.ego_vehicles[0],walker_list,max_distance=9.5)


        light_state, light_loc, _ = TrafficLightHandler.get_light_state(self.ego_vehicles[0],
                                                         offset=self._tl_offset, dist_threshold=18.0)
        desired_spd_veh = desired_spd_ped = desired_spd_rl = desired_spd_stop = self._maxium_speed
      #  print(f"Time for TrafficLightHandler: {time.time() - start_time} seconds")
        # find the nearest vehicle in lower 9.5
        if nearest_vehicle is not None:  #if have risk
            nearest_vehicle_location = nearest_vehicle.get_location()
            distance = self.ego_vehicles[0].get_location().distance(nearest_vehicle_location)
            dist_veh = max(0.0, abs(distance-8.0))
            desired_spd_veh = self._maxium_speed * np.clip(dist_veh, 0.0, 5.0)/5.0
            # print(f"nearest:{nearest_vehicle_location}")
            # print(f"distancedistance:{distance}")
            # print(f"dist_veh:{dist_veh}")
            # print(f"desired_spd_veh:{desired_spd_veh}")

        if nearest_walker is not None:
            dist_ped = max(0.0, np.linalg.norm([nearest_walker.get_transform().location.x,nearest_walker.get_transform().location.y])-6.0)
            desired_spd_ped = self._maxium_speed * np.clip(dist_ped, 0.0, 5.0)/5.0

      #  print(f"Time for if nearest_walker is not None: {time.time() - start_time} seconds")
        # traffic light
        if (light_state == carla.TrafficLightState.Red or light_state == carla.TrafficLightState.Yellow):
            dist_rl = max(0.0, np.linalg.norm(light_loc[0:2])-5.0)
            desired_spd_rl = self._maxium_speed * np.clip(dist_rl, 0.0, 5.0)/5.0
      #  print(f"Time for traffic light: {time.time() - start_time} seconds")
        # stop sign
        # stop_sign = self._scan_for_stop_sign(ev_transform)
        # print(f"Time for stop sign: {time.time() - start_time} seconds")
        # stop_loc = None
        # # reconstructingstop sign to intervene in ego
        # if (stop_sign is not None):
        #     trans = stop_sign.get_transform()
        #     tv_loc = stop_sign.trigger_volume.location
        #     loc_in_world = trans.transform(tv_loc)
        #     loc_in_ev = trans_utils.loc_global_to_ref(loc_in_world, ev_transform)
        #     stop_loc = np.array([loc_in_ev.x, loc_in_ev.y, loc_in_ev.z], dtype=np.float32)
        #     dist_stop = max(0.0, np.linalg.norm(stop_loc[0:2])-5.0)
        #     desired_spd_stop = self._maxium_speed * np.clip(dist_stop, 0.0, 5.0)/5.0
       # print(f"Time for if (stop_sign is not None): {time.time() - start_time} seconds")
        desired_speed = min(self._maxium_speed, desired_spd_veh, desired_spd_ped, desired_spd_rl, desired_spd_stop)
###
      #  print(f"Time for desired_speed reward: {time.time() - start_time} seconds")
        # desired_speed reward

        # print(f"desired_spd_veh:{desired_spd_veh}")
        # print(f"desired_spd_ped:{desired_spd_ped}")
        # print(f"desired_speed:{desired_speed}")

        if ev_speed > self._maxium_speed:
            r_speed = 1.0 - np.abs(ev_speed-desired_speed) / self._maxium_speed
        else:
            r_speed = 1.0 - np.abs(ev_speed-desired_speed) / self._maxium_speed

        # r_position reward
        wp_transform = self.get_route_transform()

        d_vec = ev_transform.location - wp_transform.location
        np_d_vec = np.array([d_vec.x, d_vec.y], dtype=np.float32)
        wp_unit_forward = wp_transform.rotation.get_forward_vector()
        np_wp_unit_right = np.array([-wp_unit_forward.y, wp_unit_forward.x], dtype=np.float32)

        lateral_distance = np.abs(np.dot(np_wp_unit_right, np_d_vec))

        r_position = -1.0 * (lateral_distance / 2.0)
###


        # r_rotation reward
        angle_difference = np.deg2rad(np.abs(trans_utils.cast_angle(
            ev_transform.rotation.yaw - wp_transform.rotation.yaw)))
       # print(f"ev_transform.rotation.yaw:{ev_transform.rotation.yaw}")
       # print(f"wp_transform.rotation.yaw:{wp_transform.rotation.yaw}")
        r_rotation = -1.0 * (angle_difference / np.pi)
        r_rotation = -1.0 * angle_difference
        reward = r_speed + r_position + r_rotation + terminal_reward + r_action
        print(f"reward:{reward}//r_speed:{r_speed}//r_position:{r_position}//r_rotation:{r_rotation}//terminal_reward:{terminal_reward}//r_action:{r_action}")
        reward_debug = 0
##
        return reward, reward_debug
    def get_running_status(self):
        """
        returns:
           bool: False if watchdog exception occured, True otherwise
        """
        if self._watchdog:
            return self._watchdog.get_status()
        return True

    def get_nearest_vehicle(self,ego_vehicle, vehicles_list, max_distance=9.5):
        """
        get the nearest surrounding vehicle。
        """
        # get the nearest surrounding vehicle.
        vehicles = vehicles_list

        nearest_vehicle = None
        min_distance = float('inf')
        ego_location = ego_vehicle.get_location()

        for vehicle in vehicles:

            if vehicle.id != ego_vehicle.id:
                # obtain the location of surrounding vehicle
                vehicle_location = vehicle.get_location()
                # calculate relative distance
                distance = ego_location.distance(vehicle_location)
                if distance > 9.5:
                    continue
                if distance < min_distance and distance <= max_distance:
                    min_distance = distance
                    nearest_vehicle = vehicle

        return nearest_vehicle, min_distance

    def _numpy(self,carla_vector, normalize=False):
        result = np.float32([carla_vector.x, carla_vector.y])
        if normalize:
            return result / (np.linalg.norm(result) + 1e-4)
        return result
    def _orientation(self,yaw):
        return np.float32([np.cos(np.radians(yaw)), np.sin(np.radians(yaw))])

    def get_collision(self,p1, v1, p2, v2):
        A = np.stack([v1, -v2], 1)
        b = p2 - p1
        if abs(np.linalg.det(A)) < 1e-3:
            return False, None
        x = np.linalg.solve(A, b)
        collides = all(x >= 0) and all(x <= 1)  # how many seconds until collision

        return collides, p1 + x[0] * v1
    def _is_walker_hazard(self, walkers_list):
        z = self.ego_vehicles[0].get_location().z
        p1 = self._numpy(self.ego_vehicles[0].get_location())
        v1 = 10.0 * self._orientation(self.ego_vehicles[0].get_transform().rotation.yaw)

        for walker in walkers_list:
            v2_hat = self._orientation(walker.get_transform().rotation.yaw)
            s2 = np.linalg.norm(self._numpy(walker.get_velocity()))
            if s2 < 0.05:
                v2_hat *= s2
            p2 = -3.0 * v2_hat + self._numpy(walker.get_location())
            v2 = 8.0 * v2_hat
            collides, collision_point = self.get_collision(p1, v1, p2, v2)
            if collides:
                return walker
        return None
    def _is_vehicle_hazard(self, vehicle_list):
        z = self.ego_vehicles[0].get_location().z
        o1 = self._orientation(self.ego_vehicles[0].get_transform().rotation.yaw)
        p1 = self._numpy(self.ego_vehicles[0].get_location())
        s1 = 9.5 #roach is 9.5, max(10, 3.0 * np.linalg.norm(self._numpy(self.ego_vehicles[0].get_velocity()))) # increases the threshold distance
        v1_hat = o1
        v1 = s1 * v1_hat
        for target_vehicle in vehicle_list:
            if target_vehicle.id == self.ego_vehicles[0].id:
                continue
            o2 = self._orientation(target_vehicle.get_transform().rotation.yaw)
            p2 = self._numpy(target_vehicle.get_location())
            s2 = max(5.0, 2.0 * np.linalg.norm(self._numpy(target_vehicle.get_velocity())))
            v2_hat = o2
            v2 = s2 * v2_hat
            p2_p1 = p2 - p1
            distance = np.linalg.norm(p2_p1)
            p2_p1_hat = p2_p1 / (distance + 1e-4)

            angle_to_car = np.degrees(np.arccos(v1_hat.dot(p2_p1_hat)))
            angle_between_heading = np.degrees(np.arccos(o1.dot(o2)))

            # to consider -ve angles too
            angle_to_car = min(angle_to_car, 360.0 - angle_to_car)
            angle_between_heading = min(angle_between_heading, 360.0 - angle_between_heading)

            if angle_between_heading > 60.0 and not (angle_to_car < 15 and distance < s1):
                continue
            elif angle_to_car > 30.0:
                continue
            elif distance > s1:
                continue
            return target_vehicle
        return None
    def stop_scenario(self,reload):
        """
        This function triggers a proper termination of a scenario
        """
        if self._watchdog:
            self._watchdog.stop()

        if self._agent_watchdog:
            self._agent_watchdog.stop()

        self.compute_duration_time()

        if self.get_running_status():
            if self.scenario is not None:
                self.scenario.terminate()
            if reload:
                if self._agent_wrapper is not None:
                    self._agent_wrapper.cleanup()
                    self._agent_wrapper = None

                self.analyze_scenario()

        # Make sure the scenario thread finishes to avoid blocks
        # self._running = False
        # self._scenario_thread.join()
        # self._scenario_thread = None

    def compute_duration_time(self):
        """
        Computes system and game duration times
        """
        self.end_system_time = time.time()
        self.end_game_time = GameTime.get_time()

        self.scenario_duration_system = self.end_system_time - self.start_system_time
        self.scenario_duration_game = self.end_game_time - self.start_game_time

    def analyze_scenario(self):
        """
        Analyzes and prints the results of the route
        """
        ResultOutputProvider(self)
    def _scan_for_stop_sign(self, vehicle_transform):
        # obtain stop sign (Roach)
        target_stop_sign = None

        ve_dir = vehicle_transform.get_forward_vector()

        wp = self._map.get_waypoint(vehicle_transform.location)
        wp_dir = wp.transform.get_forward_vector()

        dot_ve_wp = ve_dir.x * wp_dir.x + ve_dir.y * wp_dir.y + ve_dir.z * wp_dir.z
        # Ignore all when going in a wrong lane
        if dot_ve_wp > 0:
            for stop_sign in self._list_stop_signs:
                if self.is_affected_by_stop(vehicle_transform.location, stop_sign):
                    # this stop sign is affecting the vehicle
                    target_stop_sign = stop_sign
                    break

        return target_stop_sign
    def is_affected_by_stop(self, vehicle_loc, stop, multi_step=20):
        """
        Check if the given actor is affected by the stop
        """
        # check if the given actor is affected by the stop (Roach)
        affected = False
        # first we run a fast coarse test
        stop_t = stop.get_transform()
        stop_location = stop_t.location
        if stop_location.distance(vehicle_loc) > self._proximity_threshold:
            return affected

        transformed_tv = stop_t.transform(stop.trigger_volume.location)

        # slower and accurate test based on waypoint's horizon and geometric test
        list_locations = [vehicle_loc]
        waypoint = self._map.get_waypoint(vehicle_loc)
        for _ in range(multi_step):
            if waypoint:
                next_wps = waypoint.next(self._waypoint_step)
                if not next_wps:
                    break
                waypoint = next_wps[0]
                if not waypoint:
                    break
                list_locations.append(waypoint.transform.location)

        for actor_location in list_locations:
            if self.point_inside_boundingbox(actor_location, transformed_tv, stop.trigger_volume.extent):
                affected = True

        return affected
    def point_inside_boundingbox(self,point, bb_center, bb_extent):
        """
        X
        :param point:
        :param bb_center:
        :param bb_extent:
        :return:
        """
        # bugfix slim bbox
        bb_extent.x = max(bb_extent.x, bb_extent.y)
        bb_extent.y = max(bb_extent.x, bb_extent.y)

        # pylint: disable=invalid-name
        A = carla.Vector2D(bb_center.x - bb_extent.x, bb_center.y - bb_extent.y)
        B = carla.Vector2D(bb_center.x + bb_extent.x, bb_center.y - bb_extent.y)
        D = carla.Vector2D(bb_center.x - bb_extent.x, bb_center.y + bb_extent.y)
        M = carla.Vector2D(point.x, point.y)

        AB = B - A
        AD = D - A
        AM = M - A
        am_ab = AM.x * AB.x + AM.y * AB.y
        ab_ab = AB.x * AB.x + AB.y * AB.y
        am_ad = AM.x * AD.x + AM.y * AD.y
        ad_ad = AD.x * AD.x + AD.y * AD.y

        return am_ab > 0 and am_ab < ab_ab and am_ad > 0 and am_ad < ad_ad
    def get_route_transform(self):

        loc0 = self._last_route_location
        loc1 = self._global_route[0][0].transform.location
       # print(f"loc1:{loc1}")
        if loc1.distance(loc0) < 0.1:
            yaw = self._global_route[0][0].transform.rotation.yaw
        else:
            f_vec = loc1 - loc0
            yaw = np.rad2deg(np.arctan2(f_vec.y, f_vec.x))
        rot = carla.Rotation(yaw=yaw)
        return carla.Transform(location=loc0, rotation=rot)
    def _trace_route_to_global_target(self):
        # Maintain global waypoints (Roach)
        current_location = self.ego_vehicles[0].get_location()
        for tt in self._target_transforms:
            next_target_location = tt.location
            route_trace = self._planner.trace_route(current_location, next_target_location)
            self._global_route += route_trace
            self._route_length += self._compute_route_length(route_trace)
            current_location = next_target_location

        self._update_leaderboard_plan(self._global_route)
    def _update_leaderboard_plan(self, route_trace):
        # function of Carla0.9.14
        plan_gps = location_route_to_gps(route_trace)
        ds_ids = downsample_route(route_trace, 50)

        self._global_plan_gps += [plan_gps[x] for x in ds_ids]
        self._global_plan_world_coord += [(route_trace[x][0].transform.location, route_trace[x][1]) for x in ds_ids]
    def _truncate_global_route_till_local_target(self, windows_size=5):
        # truncate globalroute till local target (Roach)
        ev_location = self.ego_vehicles[0].get_location()
        closest_idx = 0

        for i in range(len(self._global_route)-1):
            if i > windows_size:
                break

            loc0 = self._global_route[i][0].transform.location
            loc1 = self._global_route[i+1][0].transform.location

            wp_dir = loc1 - loc0
            wp_veh = ev_location - loc0
            dot_ve_wp = wp_veh.x * wp_dir.x + wp_veh.y * wp_dir.y + wp_veh.z * wp_dir.z

            if dot_ve_wp > 0:
                closest_idx = i+1

        distance_traveled = self._compute_route_length(self._global_route[:closest_idx+1])
        self._route_completed += distance_traveled

        if closest_idx > 0:
            self._last_route_location = carla.Location(self._global_route[0][0].transform.location)

        self._global_route = self._global_route[closest_idx:]
        return distance_traveled
    def _compute_route_length(self,route):
        length_in_m = 0.0
        for i in range(len(route)-1):
            d = route[i][0].transform.location.distance(route[i+1][0].transform.location)
            length_in_m += d
        return length_in_m
    def global_plan_gps(self):
        return self._global_plan_gps
    def _is_route_completed(self, percentage_threshold=0.99, distance_threshold=10.0):
        # distance_threshold=10.0
        ev_loc = self.ego_vehicles[0].get_location()

        percentage_route_completed = self._route_completed / self._route_length
        is_completed = percentage_route_completed > percentage_threshold
        is_within_dist = ev_loc.distance(self._target_transforms[-1].location) < distance_threshold

        return is_completed and is_within_dist
