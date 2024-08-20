from importlib import import_module
from gym import spaces
import sys
sys.path.append("/home/vci-1/XT/leaderV2/leaderboard/leaderboard/obs_manager/birdview/")
sys.path.append("/home/vci-1/XT/leaderV2/leaderboard/leaderboard/obs_manager/actor_state/")
from chauffeurnet import ObsManager_bird
from control import ObsManager_control
from speed import ObsManager_speed
from velocity import ObsManager_velocity
import pdb
class ObsManagerHandler(object):

    def __init__(self, ego_vehicles):
        self._obs_managers = {}
        self._obs_configs = {'hero': {'birdview': {'module': 'birdview.chauffeurnet',
                       'width_in_pixels': 192,
                       'pixels_ev_to_bottom': 40,
                       'pixels_per_meter': 2.0,#5.0,
                       'history_idx': [-16, -11, -6, -1],
                       'scale_bbox': True,
                       'scale_mask_col': 1.0},
                      'speed': {'module': 'actor_state.speed'},
                      'control': {'module': 'actor_state.control'},
                      'velocity': {'module': 'actor_state.velocity'}}}
        #self._obs_configs = obs_configs
        self.ego_vehicles = ego_vehicles
        self._init_obs_managers()

    def get_observation(self, timestep):
        obs_dict = {}
        ev_id = 'hero'
   #      om_dict = {}
   #      #for ev_id, om_dict in self._obs_managers.items():
   # # 数组，再以元祖的形式包装ev_id和字典形式的om_dict，self._obs_managers.items() dict_items([('hero', {'birdview': <carla_gym.core.obs_manager.birdview.chauffeurnet.ObsManager object at 0x7f363c5f7f10>, 'speed': <carla_gym.core.obs_manager.actor_state.speed.ObsManager object at 0x7f363c61aa50>, 'control': <carla_gym.core.obs_manager.actor_state.control.ObsManager object at 0x7f363c61a850>,
   # # 'velocity': <carla_gym.core.obs_manager.actor_state.velocity.ObsManager object at 0x7f363c61abd0>})])
   #      om_dict['birdview'] = ObsManager_bird(self._obs_configs,self.ego_vehicles)
   #      om_dict['speed'] = ObsManager_speed(self._obs_configs,self.ego_vehicles)
   #      om_dict['control'] = ObsManager_control(self._obs_configs,self.ego_vehicles)
   #      om_dict['velocity'] = ObsManager_velocity(self._obs_configs,self.ego_vehicles)
        obs_dict[ev_id] = {}
        for obs_id, om in self._obs_managers.items():
            obs_dict[ev_id][obs_id] = om.get_observation()
            '''
            对应这四个类里面的状态获取函数
            {'birdview': <carla_gym.core.obs_manager.birdview.chauffeurnet.ObsManager at 0x7f363c5f7f10>,
             'speed': <carla_gym.core.obs_manager.actor_state.speed.ObsManager at 0x7f363c61aa50>,
             'control': <carla_gym.core.obs_manager.actor_state.control.ObsManager at 0x7f363c61a850>,
            #control这里拿的好像是自车的状态向量？
             'velocity': <carla_gym.core.obs_manager.actor_state.velocity.ObsManager at 0x7f363c61abd0>}
            
                '''
        return obs_dict

    @property
    def observation_space(self):
        obs_spaces_dict = {}
        for ev_id, om_dict in self._obs_managers.items():
            ev_obs_spaces_dict = {}
            for obs_id, om in om_dict.items():
                ev_obs_spaces_dict[obs_id] = om.obs_space
            obs_spaces_dict[ev_id] = spaces.Dict(ev_obs_spaces_dict)
        return spaces.Dict(obs_spaces_dict)

    def reset(self, ego_vehicles):
        self._init_obs_managers()

        #for ev_id, ev_actor in ego_vehicles.items():
        for obs_id, om in self._obs_managers.items():
                om.attach_ego_vehicle(ego_vehicles)

    def clean(self):
       # for ev_id, om_dict in self._obs_managers.items():
        for obs_id, om in self._obs_managers.items():
                om.clean()

        #pdb.set_trace()
        self._obs_managers = {}

    def _init_obs_managers(self):
        self._obs_managers = {}
        # for ev_id, om_dict in self._obs_managers.items():
        # 数组，再以元祖的形式包装ev_id和字典形式的om_dict，self._obs_managers.items() dict_items([('hero', {'birdview': <carla_gym.core.obs_manager.birdview.chauffeurnet.ObsManager object at 0x7f363c5f7f10>, 'speed': <carla_gym.core.obs_manager.actor_state.speed.ObsManager object at 0x7f363c61aa50>, 'control': <carla_gym.core.obs_manager.actor_state.control.ObsManager object at 0x7f363c61a850>,
        # 'velocity': <carla_gym.core.obs_manager.actor_state.velocity.ObsManager object at 0x7f363c61abd0>})])
        self._obs_managers['birdview'] = ObsManager_bird(self._obs_configs, self.ego_vehicles)
        # self._obs_managers['speed'] = ObsManager_speed(self._obs_configs, self.ego_vehicles)
        # self._obs_managers['control'] = ObsManager_control(self._obs_configs, self.ego_vehicles)
        # self._obs_managers['velocity'] = ObsManager_velocity(self._obs_configs, self.ego_vehicles)
      #  print("already init obs manager")

        # for ev_id, ev_obs_configs in self._obs_configs.items():
        #     self._obs_managers[ev_id] = {}
        #     for obs_id, obs_config in ev_obs_configs.items():
        #         ObsManager = getattr(import_module('carla_gym.core.obs_manager.'+obs_config["module"]), 'ObsManager')
        #         self._obs_managers[ev_id][obs_id] = ObsManager(obs_config)

        # obs_dict = {}
#         ev_id = 'hero'
        # om_dict = {}
        # # for ev_id, om_dict in self._obs_managers.items():
        # # 数组，再以元祖的形式包装ev_id和字典形式的om_dict，self._obs_managers.items() dict_items([('hero', {'birdview': <carla_gym.core.obs_manager.birdview.chauffeurnet.ObsManager object at 0x7f363c5f7f10>, 'speed': <carla_gym.core.obs_manager.actor_state.speed.ObsManager object at 0x7f363c61aa50>, 'control': <carla_gym.core.obs_manager.actor_state.control.ObsManager object at 0x7f363c61a850>,
        # # 'velocity': <carla_gym.core.obs_manager.actor_state.velocity.ObsManager object at 0x7f363c61abd0>})])
        # om_dict['birdview'] = ObsManager_bird(self._obs_configs)
        # om_dict['speed'] = ObsManager_speed(self._obs_configs)
        # om_dict['control'] = ObsManager_control(self._obs_configs)
        # om_dict['velocity'] = ObsManager_velocity(self._obs_configs)
        # obs_dict[ev_id] = {}
