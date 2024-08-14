import logging
import sys

import numpy as np
from omegaconf import OmegaConf
import wandb
import copy
#import ipdb
from carla_gym.utils.config_utils import load_entry_point


class RlBirdviewAgent():
    def __init__(self, path_to_conf_file='config_agent.yaml'):
        self._logger = logging.getLogger(__name__)
        self._render_dict = None
        self.supervision_dict = None
        self.setup(path_to_conf_file)

    def setup(self, path_to_conf_file):
        cfg = OmegaConf.load(path_to_conf_file)
        print(f"cfg:{cfg}")
        '''
        cfg:{'entry_point': 'agents.rl_birdview.rl_birdview_agent:RlBirdviewAgent', 'wb_run_path': 'zhang5g/train_rl_experts/1j1qnv0d', 
        'wb_ckpt_step': None, 'env_wrapper': {'entry_point': 'agents.rl_birdview.utils.rl_birdview_wrapper:RlBirdviewWrapper', 
        'kwargs': {'input_states': ['control', 'vel_xy'], 'acc_as_action': True}}, 'policy': {'entry_point': 'agents.rl_birdview.models.ppo_policy:PpoPolicy', 
        'kwargs': {'policy_head_arch': [256, 256], 'value_head_arch': [256, 256], 'features_extractor_entry_point': 'agents.rl_birdview.models.torch_layers:XtMaCNN', 
        'features_extractor_kwargs': {'states_neurons': [256, 256]}, 'distribution_entry_point': 'agents.rl_birdview.models.distributions:BetaDistribution', 
        'distribution_kwargs': {'dist_init': None}}}, 'training': {'entry_point': 'agents.rl_birdview.models.ppo:PPO', 
        'kwargs': {'learning_rate': 1e-05, 'n_steps_total': 12288, 'batch_size': 256, 'n_epochs': 20, 'gamma': 0.99, 
        'gae_lambda': 0.9, 'clip_range': 0.2, 'clip_range_vf': None, 'ent_coef': 0.01, 'explore_coef': 0.05, 'vf_coef': 0.5, 
        'max_grad_norm': 0.5, 'target_kl': 0.01, 'update_adv': False, 'lr_schedule_step': 8}},
         'obs_configs': {'birdview': {'module': 'birdview.chauffeurnet', 'width_in_pixels': 192,
          'pixels_ev_to_bottom': 40, 'pixels_per_meter': 5.0, 'history_idx': [-16, -11, -6, -1],
           'scale_bbox': True, 'scale_mask_col': 1.0}, 'speed': {'module': 'actor_state.speed'}, 
           'control': {'module': 'actor_state.control'}, 'velocity': {'module': 'actor_state.velocity'}}}
        '''
        # load checkpoint from wandb
        cfg.wb_run_path = None
        if cfg.wb_run_path is not None:
            api = wandb.Api()
            run = api.run(cfg.wb_run_path)
            all_ckpts = [f for f in run.files() if 'ckpt' in f.name]

            if cfg.wb_ckpt_step is None:
                f = max(all_ckpts, key=lambda x: int(x.name.split('_')[1].split('.')[0]))
                self._logger.info(f'Resume checkpoint latest {f.name}')
                sys.exit() #ipdb.set_trace() #bash脚本打断点
            else:
                wb_ckpt_step = int(cfg.wb_ckpt_step)
                f = min(all_ckpts, key=lambda x: abs(int(x.name.split('_')[1].split('.')[0]) - wb_ckpt_step))
                self._logger.info(f'Resume checkpoint closest to step {wb_ckpt_step}: {f.name}')

            f.download(replace=True)
            run.file('config_agent.yaml').download(replace=True)
            cfg = OmegaConf.load('config_agent.yaml')
            self._ckpt = f.name
        else:
            self._ckpt = None

        cfg = OmegaConf.to_container(cfg)

        self._obs_configs = cfg['obs_configs']
        self._train_cfg = cfg['training']

        # prepare policy
        self._policy_class = load_entry_point(cfg['policy']['entry_point'])
        self._policy_kwargs = cfg['policy']['kwargs']
        if self._ckpt is None:
            self._policy = None
        else:
            self._logger.info(f'Loading wandb checkpoint: {self._ckpt}')
            self._policy, self._train_cfg['kwargs'] = self._policy_class.load(self._ckpt)
            self._policy = self._policy.eval()

        self._wrapper_class = load_entry_point(cfg['env_wrapper']['entry_point'])
        self._wrapper_kwargs = cfg['env_wrapper']['kwargs']

    def run_step(self, input_data, timestamp):
        input_data = copy.deepcopy(input_data)

        policy_input = self._wrapper_class.process_obs(input_data, self._wrapper_kwargs['input_states'], train=False)

        actions, values, log_probs, mu, sigma, features = self._policy.forward(
            policy_input, deterministic=True, clip_action=True)
        control = self._wrapper_class.process_act(actions, self._wrapper_kwargs['acc_as_action'], train=False)
        self.supervision_dict = {
            'action': np.array([control.throttle, control.steer, control.brake], dtype=np.float32),
            'value': values[0],
            'action_mu': mu[0],
            'action_sigma': sigma[0],
            'features': features[0],
            'speed': input_data['speed']['forward_speed']
        }
        self.supervision_dict = copy.deepcopy(self.supervision_dict)

        self._render_dict = {
            'timestamp': timestamp,
            'obs': policy_input,
            'im_render': input_data['birdview']['rendered'],
            'action': actions,
            'action_value': values[0],
            'action_log_probs': log_probs[0],
            'action_mu': mu[0],
            'action_sigma': sigma[0]
        }
        self._render_dict = copy.deepcopy(self._render_dict)

        return control

    def reset(self, log_file_path):
        # logger
        self._logger.handlers = []
        self._logger.propagate = False
        self._logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler(log_file_path, mode='w')
        fh.setLevel(logging.DEBUG)
        self._logger.addHandler(fh)

    def learn(self, env, total_timesteps, callback, seed):
        if self._policy is None:
            self._policy = self._policy_class(env.observation_space, env.action_space, **self._policy_kwargs)

        # init ppo model
        model_class = load_entry_point(self._train_cfg['entry_point'])
        model = model_class(self._policy, env, **self._train_cfg['kwargs'])
        model.learn(total_timesteps, callback=callback, seed=seed)

    def render(self, reward_debug, terminal_debug):
        '''
        test render, used in benchmark.py
        '''
        self._render_dict['reward_debug'] = reward_debug
        self._render_dict['terminal_debug'] = terminal_debug

        return self._wrapper_class.im_render(self._render_dict)

    @property
    def obs_configs(self):
        return self._obs_configs
