import yaml
import torch
import numpy as np

from pearl.utils.instantiations.spaces.discrete_action import DiscreteActionSpace
from pearl.utils.instantiations.spaces.box_action import BoxActionSpace
from pearl.action_representation_modules.identity_action_representation_module import IdentityActionRepresentationModule

from models.pearl import MultiStateCritic, MultiStateActor, MultiStateGaussianActor

class AgentConfig:
    route_window_size = None
    route_resample = None
    last_action_resample = None
    red_light_resample = None
    stop_sign_resample = None
    speed_resample = None
    state_horizen = None
    measurement_dim = None

    control_interval = None
    action_mask_basis = None
    speed_limit = None

    buffer_size = None
    start_steps = None
    throttle_set = None
    steer_set = None
    agent_name = None
    scenario_name = None
    agent_config_dict = None
    exp_config_dict = None

    @classmethod
    def parse_config(cls, path_to_conf_file):
        with open(path_to_conf_file, 'r') as file:
            data = yaml.safe_load(file)

        ## load basic info config
        cls.exp_config_dict = data['Basic']

        ## load state/env config
        for key, value in data['ExtraConfig'].items():
            setattr(cls, key, value)
        cls.measurement_dim = (
            cls.route_window_size * 3 * cls.route_resample +
            cls.last_action_resample * 2 +
            cls.red_light_resample * 1 +
            cls.stop_sign_resample * 1 +
            cls.speed_resample * 1 + 2
        ) * cls.state_horizen

        ## load agent config
        DA_list = ['DSAC', 'PPO']
        CA_list = ['SAC', ]
        algorithm = data['Basic']['algorithm']
        if algorithm in DA_list:
            # Discret algorithm settings
            # if data['Basic']['action_type']['steer'] == 'direct':
            cls.agent_name = 'MaskAgent'
            action_space = (
                np.array(
                    list(map(
                        np.ravel,
                        np.meshgrid(cls.steer_set,
                                    cls.throttle_set)))
                    ).T
                ).astype(np.float32)

            # elif data['Basic']['action_type']['steer'] == 'direct':
            #     cls.agent_name = 'DirectAgent'
            #     action_space = np.vstack([
            #         np.array([0,-1]),
            #         np.array([[-0.5, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.5],
            #                 [0.7]*9]).T,
            #         np.array([[-0.7, -0.5, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.5, 0.7],
            #                 [0.3]*11]).T,
            #         np.array([[-1, -0.6, -0.3, -0.1, 0, 0.1, 0.3, 0.6, 1],
            #                 [0]*9]).T
            #     ]).astype(np.float32)
                
            # else:
            #      raise NotImplementedError('Invalid action type in config.yaml!')

            cls.agent_config_dict = {
                **{key: value for key, value in data['PearlPara'].items() if key not in DA_list},
                **data['PearlPara'][algorithm],
                'action_space': DiscreteActionSpace(torch.from_numpy(action_space)),
                'actor_network_type': MultiStateActor,
                'critic_network_type': MultiStateCritic,
                'action_representation_module': 
                IdentityActionRepresentationModule(*action_space.shape)
            }
            
        elif algorithm in CA_list:
            # Continuous algorithm setting
            # action_space = None # TODO: implement
            # del cls.exp_config_dict['action_set']
            # del cls.exp_config_dict['action_encode'] 
            cls.exp_config_dict['action_type'] = 'direct'
            cls.exp_config_dict['action_mask'] = False        

            cls.agent_config_dict = {
                **data['PearlPara'],
                'action_space': BoxActionSpace([-1, -1],[1, 1]),
                'actor_network_type': MultiStateGaussianActor,
                'critic_network_type': MultiStateCritic
            }                
        else:
            raise NotImplementedError(f'Algorithm {algorithm} not implement yet!')            

        # action_dim = action_space.shape[0]
        # action_mask_basis = np.array([
        #     np.sum(np.eye(action_dim)[:10, :], axis = 0),
        #     np.sum(np.eye(action_dim)[-10:, :], axis = 0),
        #     np.sum(np.eye(action_dim)[[0,1,5,6,10,11,15,16,20,21], :], axis = 0),
        #     np.sum(np.eye(action_dim)[[3,4,8,9,13,14,18,19,23,24], :], axis = 0)
        # ])
        # setattr(cls, 'action_mask_basis', action_mask_basis)


        

