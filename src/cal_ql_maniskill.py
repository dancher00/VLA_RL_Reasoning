from collections import defaultdict
from dataclasses import dataclass
import os
import random
import time
from typing import Optional
import tqdm
from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper, FlattenRGBDObservationWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import tyro
import mani_skill.envs
import glob


import os
import random
import uuid
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import pyrallis
import wandb
from torch.distributions import Normal, TanhTransform, TransformedDistribution


TensorBatch = List[torch.Tensor]
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"), flush=True)

@dataclass
class TrainConfig:
    # Common things
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    checkpoint: Optional[str] = None
    """path to a pretrained checkpoint file to start evaluation/training from"""
    is_env_with_goal: bool = True

    # Environment specific arguments
    env_id: str = "PickCube-v1"
    """the id of the environment"""
    obs_mode: str = "rgb"
    """the observation mode to use"""
    include_state: bool = True
    """whether to include the state in the observation"""
    env_vectorization: str = "gpu"
    """the type of environment vectorization to use"""
    num_envs: int = 100
    """the number of parallel environments"""
    num_eval_envs: int = 1
    """the number of parallel evaluation environments"""
    num_eval_episodes: int = 10
    partial_reset: bool = False
    """whether to let parallel environments reset upon termination instead of truncation"""
    eval_partial_reset: bool = False
    """whether to let parallel evaluation environments reset upon termination instead of truncation"""
    num_steps: int = 50
    """the number of steps to run in each environment per policy rollout"""
    num_eval_steps: int = 50
    """the number of steps to run in each evaluation environment during evaluation"""
    reconfiguration_freq: Optional[int] = None
    """how often to reconfigure the environment during training"""
    eval_reconfiguration_freq: Optional[int] = 1
    """for benchmarking purposes we want to reconfigure the eval environment each reset to ensure objects are randomized in some tasks"""
    eval_freq: int = 25
    """evaluation frequency in terms of iterations"""
    control_mode: Optional[str] = "pd_ee_delta_pose"
    """the control mode to use for the environment"""
    render_mode: str = "all"
    """the environment rendering mode"""
    ref_min_score: int = 0
    offline_iterations: int = int(100)  # Number of offline updates
    online_iterations: int = int(1_000_000)  # Number of online updates

    # Algorithm specific arguments
    buffer_size: int = 10_000  # Replay buffer size
    buffer_device: str = "cpu"
    batch_size: int = 16  # Batch size for all networks
    discount: float = 0.99  # Discount factor
    alpha_multiplier: float = 1.0  # Multiplier for alpha in loss
    use_automatic_entropy_tuning: bool = True  # Tune entropy
    backup_entropy: bool = False  # Use backup entropy
    policy_lr: float = 3e-5  # Policy learning rate
    qf_lr: float = 3e-4  # Critics learning rate
    soft_target_update_rate: float = 5e-3  # Target network update rate
    bc_steps: int = int(0)  # Number of BC steps at start
    target_update_period: int = 1  # Frequency of target nets updates
    cql_alpha: float = 10.0  # CQL offline regularization parameter
    cql_alpha_online: float = 10.0  # CQL online regularization parameter
    cql_n_actions: int = 10  # Number of sampled actions
    cql_importance_sample: bool = True  # Use importance sampling
    cql_lagrange: bool = False  # Use Lagrange version of CQL
    cql_target_action_gap: float = -1.0  # Action gap
    cql_temp: float = 1.0  # CQL temperature
    cql_max_target_backup: bool = False  # Use max target backup
    cql_clip_diff_min: float = -np.inf  # Q-function lower loss clipping
    cql_clip_diff_max: float = np.inf  # Q-function upper loss clipping
    orthogonal_init: bool = True  # Orthogonal initialization
    normalize: bool = True  # Normalize states
    q_n_hidden_layers: int = 2  # Number of hidden layers in Q networks
    reward_scale: float = 1.0  # Reward scale for normalization
    reward_bias: float = 0.0  # Reward bias for normalization
    # Cal-QL
    mixing_ratio: float = 0.5  # Data mixing ratio for online tuning
    is_sparse_reward: bool = False  # Use sparse reward
    # Wandb logging
    entity: str = "r-rodionvahitoff"
    project: str = "CORL"
    group: str = "Cal-QL"
    name: str = "Cal-QL"


class DictArray(object):
    def __init__(self, buffer_shape, element_space, data_dict=None, device=None):
        self.buffer_shape = buffer_shape
        if data_dict:
            self.data = data_dict
        else:
            assert isinstance(element_space, gym.spaces.dict.Dict)
            self.data = {}
            for k, v in element_space.items():
                if isinstance(v, gym.spaces.dict.Dict):
                    self.data[k] = DictArray(buffer_shape, v, device=device)
                else:
                    dtype = (torch.float32 if v.dtype in (np.float32, np.float128) else
                            torch.uint8 if v.dtype == np.uint8 else
                            torch.int16 if v.dtype == np.int16 else
                            torch.int32 if v.dtype == np.int32 else
                            v.dtype)
                    self.data[k] = torch.zeros(buffer_shape + v.shape, dtype=dtype, device=device)

    def keys(self):
        return self.data.keys()

    def __getitem__(self, index):
        if isinstance(index, str):
            return self.data[index]
        return {
            k: v[index] for k, v in self.data.items()
        }

    def __setitem__(self, index, value):
        if isinstance(index, str):
            self.data[index] = value
        for k, v in value.items():
            self.data[k][index] = v

    @property
    def shape(self):
        return self.buffer_shape

    def reshape(self, shape):
        t = len(self.buffer_shape)
        new_dict = {}
        for k,v in self.data.items():
            if isinstance(v, DictArray):
                new_dict[k] = v.reshape(shape)
            else:
                new_dict[k] = v.reshape(shape + v.shape[t:])
        new_buffer_shape = next(iter(new_dict.values())).shape[:len(shape)]
        return DictArray(new_buffer_shape, None, data_dict=new_dict)


def load_npz_folder_to_tensors(folder_path: str) -> dict:

    file_paths = sorted(glob.glob(os.path.join(folder_path, '*.npz')))
    if not file_paths:
        raise RuntimeError(f'Не найдено ни одного .npz файла в {folder_path}')
    
    data_dict = {}
    
    for fp in file_paths:
        episode = np.load(fp)
        for key in episode.files:
            arr = episode[key]  
            data_dict.setdefault(key, []).append(arr)
    
    tensor_dict = {}
    for key, arr_list in data_dict.items():
      
        stacked = np.stack(arr_list, axis=1)             # shape (E, T, …)
        tensor = torch.from_numpy(stacked)               
        tensor_dict[key] = tensor
    
    return tensor_dict

def get_return_to_go(dataset: Dict, config: TrainConfig) -> np.ndarray:
    rewards = dataset["reward"].cpu().numpy()
    dones = dataset["done"].cpu().numpy()
    T, num_envs = rewards.shape
    returns = [[] for _ in range(num_envs)]
    ep_ret = [0.0 for _ in range(num_envs)]
    ep_len = [0 for _ in range(num_envs)]
    cur_rewards = [[] for _ in range(num_envs)]
    terminals = [[] for _ in range(num_envs)]

    for t in range(T):
        for env_idx in range(num_envs):
            r = float(rewards[t, env_idx].squeeze())
            d = float(dones[t, env_idx].squeeze())
            ep_ret[env_idx] += r
            cur_rewards[env_idx].append(r)
            terminals[env_idx].append(d)
            ep_len[env_idx] += 1
            is_last_step = (t == T - 1) or (ep_len[env_idx] == config.num_steps)
            if d or is_last_step:
                discounted_returns = [0] * ep_len[env_idx]
                prev_return = 0
                if (
                    config.is_sparse_reward
                    and r == config.ref_min_score * config.reward_scale + config.reward_bias
                ):
                    discounted_returns = [r / (1 - config.discount)] * ep_len[env_idx]
                else:
                    for i in reversed(range(ep_len[env_idx])):
                        discounted_returns[i] = cur_rewards[env_idx][i] + config.discount * prev_return * (1 - terminals[env_idx][i])
                        prev_return = discounted_returns[i]
                returns[env_idx] += discounted_returns
                ep_ret[env_idx] = 0.0
                ep_len[env_idx] = 0
                cur_rewards[env_idx] = []
                terminals[env_idx] = []
    return np.array(returns).T




class ReplayBuffer:
    def __init__(self, env, num_envs: int, buffer_size: int, storage_device: torch.device, sample_device: torch.device):
        self.buffer_size = buffer_size
        self.pos = 0
        self.full = False
        self.num_envs = num_envs
        self.storage_device = storage_device
        self.sample_device = sample_device
        self.per_env_buffer_size = buffer_size // num_envs
        # note 128x128x3 RGB data with replay buffer size 100_000 takes up around 4.7GB of GPU memory
        # 32 parallel envs with rendering uses up around 2.2GB of GPU memory.
        self.obs = DictArray((self.per_env_buffer_size, num_envs), env.single_observation_space, device=storage_device)
        # TODO (stao): optimize final observation storage
        self.next_obs = DictArray((self.per_env_buffer_size, num_envs), env.single_observation_space, device=storage_device)
        self.actions = torch.zeros((self.per_env_buffer_size, num_envs) + env.single_action_space.shape, device=storage_device)
        self.logprobs = torch.zeros((self.per_env_buffer_size, num_envs), device=storage_device)
        self.rewards = torch.zeros((self.per_env_buffer_size, num_envs), device=storage_device)
        self.dones = torch.zeros((self.per_env_buffer_size, num_envs), device=storage_device)
        self.values = torch.zeros((self.per_env_buffer_size, num_envs), device=storage_device)
        self.mc_returns = torch.zeros((self.per_env_buffer_size, num_envs), device=storage_device)

    def add(self, obs: torch.Tensor, 
            next_obs: torch.Tensor, 
            action: torch.Tensor, 
            reward: torch.Tensor, 
            done: torch.Tensor):
        if self.storage_device == torch.device("cpu"):
            obs = {k: v.cpu() for k, v in obs.items()}
            next_obs = {k: v.cpu() for k, v in next_obs.items()}
            action = action.cpu()
            reward = reward.cpu()
            done = done.cpu()

        self.obs[self.pos] ={k: v.detach() for k, v in obs.items()}
        self.next_obs[self.pos] = {k: v.detach() for k, v in next_obs.items()}

        self.actions[self.pos] = action.detach()
        self.rewards[self.pos] = reward.detach()
        self.dones[self.pos] = done.detach()

        self.pos += 1
        if self.pos == self.per_env_buffer_size:
            self.full = True
            self.pos = 0
    def sample(self, batch_size: int):
        if self.full:
            batch_inds = torch.randint(0, self.per_env_buffer_size, size=(batch_size, ))
        else:
            batch_inds = torch.randint(0, self.pos, size=(batch_size, ))
        env_inds = torch.randint(0, self.num_envs, size=(batch_size, ))
        obs_sample = self.obs[batch_inds, env_inds]
        next_obs_sample = self.next_obs[batch_inds, env_inds]
        obs_sample = {k: v.to(self.sample_device) for k, v in obs_sample.items()}
        next_obs_sample = {k: v.to(self.sample_device) for k, v in next_obs_sample.items()}
        return [
            obs_sample,
            next_obs_sample,
            self.actions[batch_inds, env_inds].to(self.sample_device),
            self.rewards[batch_inds, env_inds].to(self.sample_device),
            self.dones[batch_inds, env_inds].to(self.sample_device),
            self.mc_returns[batch_inds, env_inds].to(self.sample_device)
        ]

    def load_from_dataset(self, data_folder, config):

        tensors = load_npz_folder_to_tensors(data_folder)
        rtg_np = get_return_to_go(tensors, config)
        mc_returns = torch.as_tensor(rtg_np, device=tensors['reward'].device)

        n = tensors['reward'].shape[0] - 1
        self.obs[:n] = {'rgb': tensors['rgb'][:-1], 'state': tensors['joints'][:-1]}
        self.next_obs[:n] = {'rgb': tensors['rgb'][1:], 'state': tensors['joints'][1:]}
        self.actions[:n] = tensors['action'][:-1]
        self.rewards[:n] = tensors['reward'][:-1]
        self.dones[:n] = tensors['done'][:-1]
        self.mc_returns[:n] = mc_returns[:-1]

        self.pos = n % self.per_env_buffer_size
        self.full = n >= self.per_env_buffer_size



def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)


def compute_mean_std(states: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    mean = states.mean(0)
    std = states.std(0) + eps
    return mean, std


def normalize_states(states: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (states - mean) / std


def extend_and_repeat(tensor: torch.Tensor, dim: int, repeat: int) -> torch.Tensor:
    return tensor.unsqueeze(dim).repeat_interleave(repeat, dim=dim)


def make_mlp(in_channels, mlp_channels, act_builder=nn.ReLU, last_act=True):
    c_in = in_channels
    module_list = []
    for idx, c_out in enumerate(mlp_channels):
        module_list.append(nn.Linear(c_in, c_out))
        if last_act or idx < len(mlp_channels) - 1:
            module_list.append(act_builder())
        c_in = c_out
    return nn.Sequential(*module_list)


class PlainConv(nn.Module):
    def __init__(self,
                 in_channels=3,
                 out_dim=256,
                 pool_feature_map=False,
                 last_act=True, # True for ConvBody, False for CNN
                 image_size=[128, 128]
                 ):
        super().__init__()
        # assume input image size is 128x128 or 128x128

        self.out_dim = out_dim
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1, bias=True), nn.ReLU(),
            nn.MaxPool2d(4, 4) if image_size[0] == 128 and image_size[1] == 128 else nn.MaxPool2d(2, 2),  # [32, 32]
            nn.Conv2d(16, 32, 3, padding=1, bias=True), nn.ReLU(),
            nn.MaxPool2d(2, 2),  # [16, 16]
            nn.Conv2d(32, 128, 3, padding=1, bias=True), nn.ReLU(),
            nn.MaxPool2d(2, 2),  # [8, 8]
            nn.Conv2d(128, 128, 3, padding=1, bias=True), nn.ReLU(),
            nn.MaxPool2d(2, 2),  # [4, 4]
            nn.Conv2d(128, 128, 1, padding=0, bias=True), nn.ReLU(),
        )

        if pool_feature_map:
            self.pool = nn.AdaptiveMaxPool2d((1, 1))
            self.fc = make_mlp(128, [out_dim], last_act=last_act)
        else:
            self.pool = None
            self.fc = make_mlp(128 * 4 * 4, [out_dim], last_act=last_act)

        self.reset_parameters()

    def reset_parameters(self):
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, image):
        x = self.cnn(image)
        if self.pool is not None:
            x = self.pool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x


class EncoderObsWrapper(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
    def forward(self, obs):
        rgb = obs.float() / 255.0
        img = rgb.permute(0,3,1,2)
        return self.encoder(img)


def wandb_init(config: dict) -> None:
    wandb.init(
        config=config,
        project=config["project"],
        group=config["group"],
        name=config["name"],
        entity=config["entity"],
        id=str(uuid.uuid4()),
    )


@torch.no_grad()
def eval_actor(
    env: gym.Env, actor: nn.Module, device: str, n_episodes: int, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    actor.eval()
    episode_rewards = []
    successes = []
    for ep in range(n_episodes):
        state, _ = env.reset(seed = seed + ep)
        done = False
        episode_reward = 0.0
        while not done:
            action = actor.act(state, device)
            state, reward, terminated, truncated, env_infos = env.step(action)
            done = terminated or truncated
            episode_reward += reward.cpu().numpy()
                # Valid only for environments with goal
        episode_rewards.append(episode_reward)
        success = env_infos.get("is_success", False)
        successes.append(float(success))

    actor.train()
    return np.asarray(episode_rewards), np.asarray(success)


def init_module_weights(module: torch.nn.Module, orthogonal_init: bool = False):
    if isinstance(module, nn.Linear):
        if orthogonal_init:
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.constant_(module.bias, 0.0)
        else:
            nn.init.xavier_uniform_(module.weight, gain=1e-2)


class ReparameterizedTanhGaussian(nn.Module):
    def __init__(
        self,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
        no_tanh: bool = False,
    ):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.no_tanh = no_tanh

    def log_prob(
        self, mean: torch.Tensor, log_std: torch.Tensor, sample: torch.Tensor
    ) -> torch.Tensor:
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        if self.no_tanh:
            action_distribution = Normal(mean, std)
        else:
            action_distribution = TransformedDistribution(
                Normal(mean, std), TanhTransform(cache_size=1)
            )
        return torch.sum(action_distribution.log_prob(sample), dim=-1)

    def forward(
        self,
        mean: torch.Tensor,
        log_std: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)

        if self.no_tanh:
            action_distribution = Normal(mean, std)
        else:
            action_distribution = TransformedDistribution(
                Normal(mean, std), TanhTransform(cache_size=1)
            )

        if deterministic:
            action_sample = torch.tanh(mean)
        else:
            action_sample = action_distribution.rsample()

        log_prob = torch.sum(action_distribution.log_prob(action_sample), dim=-1)

        return action_sample, log_prob


class Scalar(nn.Module):
    def __init__(self, init_value: float):
        super().__init__()
        self.constant = nn.Parameter(torch.tensor(init_value, dtype=torch.float32))

    def forward(self) -> nn.Parameter:
        return self.constant


class TanhGaussianPolicy(nn.Module):
    def __init__(
        self,
        envs,
        sample_obs,
        max_action: float,
        log_std_multiplier: float = 1.0,
        log_std_offset: float = -1.0,
        orthogonal_init: bool = False,
        no_tanh: bool = False,
        device: str = 'cuda'
    ):
        super().__init__()
        in_channels = sample_obs["rgb"].shape[-1]
        image_size = sample_obs["rgb"].shape[1:3]
        state_size = envs.single_observation_space['state'].shape[0]
        self.encoder = EncoderObsWrapper(
            PlainConv(in_channels=in_channels, out_dim=256, image_size=image_size) # assume image is 128x128
        )
        self.action_dim = np.prod(envs.single_action_space.shape)
        self.max_action = torch.as_tensor(max_action, dtype=torch.float32, device = device)
        self.orthogonal_init = orthogonal_init
        self.no_tanh = no_tanh

        self.base_network = nn.Sequential(
            nn.Linear(self.encoder.encoder.out_dim + state_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 2 * self.action_dim),
        )

        if orthogonal_init:
            self.base_network.apply(lambda m: init_module_weights(m, True))
        else:
            init_module_weights(self.base_network[-1], False)

        self.log_std_multiplier = Scalar(log_std_multiplier)
        self.log_std_offset = Scalar(log_std_offset)
        self.tanh_gaussian = ReparameterizedTanhGaussian(no_tanh=no_tanh)

    def log_prob(
        self, observations: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        if actions.ndim == 3:
            observations_rgb = extend_and_repeat(observations['rgb'], 1, actions.shape[1])
            observations_joint = extend_and_repeat(observations['state'], 1, actions.shape[1])
            b, r, h, w, c = observations_rgb.shape
            observations_rgb = observations_rgb.reshape(b * r, h, w, c)
            encoder_network_output = self.encoder(observations_rgb)
            b, r, s = observations_joint.shape
            observations_joint = observations_joint.reshape(b * r, s)
            input = torch.cat([encoder_network_output, observations_joint], dim=1)
        else:
            encoder_network_output = self.encoder(observations['rgb'])
            input = torch.cat([encoder_network_output, observations['state']], dim=1)
        base_network_output = self.base_network(input)
        mean, log_std = torch.split(base_network_output, self.action_dim, dim=-1)
        log_std = self.log_std_multiplier() * log_std + self.log_std_offset()
        _, log_probs = self.tanh_gaussian(mean, log_std, False)
        return log_probs

    def forward(
        self,
        observations: torch.Tensor,
        deterministic: bool = False,
        repeat: int | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if repeat is not None:
            observations_rgb = extend_and_repeat(observations['rgb'], 1, repeat)
            observations_joint = extend_and_repeat(observations['state'], 1, repeat)
            b, r, h, w, c = observations_rgb.shape
            observations_rgb = observations_rgb.reshape(b * r, h, w, c)
            encoder_network_output = self.encoder(observations_rgb)
            b, r, s = observations_joint.shape
            observations_joint = observations_joint.reshape(b * r, s)
            input = torch.cat([encoder_network_output, observations_joint], dim=1)
        else:
            encoder_network_output = self.encoder(observations['rgb'])
            input = torch.cat([encoder_network_output, observations['state']], dim=1)
        base_network_output = self.base_network(input)
        mean, log_std = torch.split(base_network_output, int(self.action_dim), dim=-1)
        log_std = self.log_std_multiplier() * log_std + self.log_std_offset()
        actions, log_probs = self.tanh_gaussian(mean, log_std, deterministic)
        return self.max_action * actions, log_probs

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu"):
        state = {
            k: (v.detach().clone().to(device=device, dtype=torch.float32) if isinstance(v, torch.Tensor) else v)
            for k, v in state.items()
        }
        with torch.no_grad():
            actions, _ = self(state, not self.training)
        return actions.cpu().data.numpy().flatten()


class FullyConnectedQFunction(nn.Module):
    def __init__(
        self,
        envs,
        encoder: EncoderObsWrapper,
        orthogonal_init: bool = False,
        n_hidden_layers: int = 2,
    ):
        super().__init__()
        self.encoder = encoder
        action_dim = np.prod(envs.single_action_space.shape)
        state_size = envs.single_observation_space['state'].shape[0]
        orthogonal_init = orthogonal_init

        layers = [
            nn.Linear(self.encoder.encoder.out_dim + state_size + action_dim, 256),
            nn.ReLU(),
        ]
        for _ in range(n_hidden_layers - 1):
            layers.append(nn.Linear(256, 256))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(256, 1))

        self.network = nn.Sequential(*layers)
        if orthogonal_init:
            self.network.apply(lambda m: init_module_weights(m, True))
        else:
            init_module_weights(self.network[-1], False)

    def forward(self, observations: torch.Tensor, actions: torch.Tensor, detach_encoder: bool = False) -> torch.Tensor:
        multiple_actions = False
        batch_size = observations['rgb'].shape[0]
        if actions.ndim == 3 and observations['rgb'].ndim == 4:
            multiple_actions = True
            observations_rgb = extend_and_repeat(observations['rgb'], 1, actions.shape[1])
            observations_joint = extend_and_repeat(observations['state'], 1, actions.shape[1])
            actions = actions.reshape(-1, actions.shape[-1])
            b, r, h, w, c = observations_rgb.shape
            observations_rgb = observations_rgb.reshape(b * r, h, w, c)
            encoder_states = self.encoder(observations_rgb)
            b, r, s = observations_joint.shape
            observations_joint = observations_joint.reshape(b * r, s)
        else:
            encoder_states = self.encoder(observations['rgb'])
            observations_joint = observations['state']
        if detach_encoder:
            encoder_states = encoder_states.detach()
        input_tensor = torch.cat([encoder_states, observations_joint, actions], dim=-1)
        q_values = torch.squeeze(self.network(input_tensor), dim=-1)
        if multiple_actions:
            q_values = q_values.reshape(batch_size, -1)
        return q_values


class CalQL:
    def __init__(
        self,
        critic_1,
        critic_1_optimizer,
        critic_2,
        critic_2_optimizer,
        actor,
        actor_optimizer,
        target_entropy: float,
        discount: float = 0.99,
        alpha_multiplier: float = 1.0,
        use_automatic_entropy_tuning: bool = True,
        backup_entropy: bool = False,
        policy_lr: bool = 3e-4,
        qf_lr: bool = 3e-4,
        soft_target_update_rate: float = 5e-3,
        bc_steps=100000,
        target_update_period: int = 1,
        cql_n_actions: int = 10,
        cql_importance_sample: bool = True,
        cql_lagrange: bool = False,
        cql_target_action_gap: float = -1.0,
        cql_temp: float = 1.0,
        cql_alpha: float = 5.0,
        cql_max_target_backup: bool = False,
        cql_clip_diff_min: float = -np.inf,
        cql_clip_diff_max: float = np.inf,
        device: str = "cpu",
    ):
        super().__init__()

        self.discount = discount
        self.target_entropy = target_entropy
        self.alpha_multiplier = alpha_multiplier
        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        self.backup_entropy = backup_entropy
        self.policy_lr = policy_lr
        self.qf_lr = qf_lr
        self.soft_target_update_rate = soft_target_update_rate
        self.bc_steps = bc_steps
        self.target_update_period = target_update_period
        self.cql_n_actions = cql_n_actions
        self.cql_importance_sample = cql_importance_sample
        self.cql_lagrange = cql_lagrange
        self.cql_target_action_gap = cql_target_action_gap
        self.cql_temp = cql_temp
        self.cql_alpha = cql_alpha
        self.cql_max_target_backup = cql_max_target_backup
        self.cql_clip_diff_min = cql_clip_diff_min
        self.cql_clip_diff_max = cql_clip_diff_max
        self._device = device

        self.total_it = 0

        self.critic_1 = critic_1
        self.critic_2 = critic_2

        self.target_critic_1 = deepcopy(self.critic_1).to(device)
        self.target_critic_2 = deepcopy(self.critic_2).to(device)

        self.actor = actor

        self.actor_optimizer = actor_optimizer
        self.critic_1_optimizer = critic_1_optimizer
        self.critic_2_optimizer = critic_2_optimizer

        if self.use_automatic_entropy_tuning:
            self.log_alpha = Scalar(0.0)
            self.alpha_optimizer = torch.optim.Adam(
                self.log_alpha.parameters(),
                lr=self.policy_lr,
            )
        else:
            self.log_alpha = None

        self.log_alpha_prime = Scalar(1.0)
        self.alpha_prime_optimizer = torch.optim.Adam(
            self.log_alpha_prime.parameters(),
            lr=self.qf_lr,
        )

        self._calibration_enabled = True
        self.total_it = 0

    def update_target_network(self, soft_target_update_rate: float):
        soft_update(self.target_critic_1, self.critic_1, soft_target_update_rate)
        soft_update(self.target_critic_2, self.critic_2, soft_target_update_rate)

    def switch_calibration(self):
        self._calibration_enabled = not self._calibration_enabled

    def _alpha_and_alpha_loss(self, observations: torch.Tensor, log_pi: torch.Tensor):
        if self.use_automatic_entropy_tuning:
            alpha_loss = -(
                self.log_alpha() * (log_pi + self.target_entropy).detach()
            ).mean()
            alpha = self.log_alpha().exp() * self.alpha_multiplier
        else:
            alpha_loss = observations['rgb'].new_tensor(0.0)
            alpha = observations['rgb'].new_tensor(self.alpha_multiplier)
        return alpha, alpha_loss

    def _policy_loss(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        new_actions: torch.Tensor,
        alpha: torch.Tensor,
        log_pi: torch.Tensor,
    ) -> torch.Tensor:
        if self.total_it <= self.bc_steps:
            log_probs = self.actor.log_prob(observations, actions)
            policy_loss = (alpha * log_pi - log_probs).mean()
        else:
            q_new_actions = torch.min(
                self.critic_1(observations, new_actions, detach_encoder = True),
                self.critic_2(observations, new_actions, detach_encoder = True),
            )
            policy_loss = (alpha * log_pi - q_new_actions).mean()
        return policy_loss

    def _q_loss(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        next_observations: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        mc_returns: torch.Tensor,
        alpha: torch.Tensor,
        log_dict: Dict,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q1_predicted = self.critic_1(observations, actions)
        q2_predicted = self.critic_2(observations, actions)
        if self.cql_max_target_backup:
            new_next_actions, next_log_pi = self.actor(
                next_observations, repeat=self.cql_n_actions
                )
            target_q_values, max_target_indices = torch.max(
                torch.min(
                    self.target_critic_1(next_observations, new_next_actions),
                    self.target_critic_2(next_observations, new_next_actions),
                ),
                dim=-1,
            )
            next_log_pi = torch.gather(
                next_log_pi, -1, max_target_indices.unsqueeze(-1)
            ).squeeze(-1)
        else:
            new_next_actions, next_log_pi = self.actor(next_observations)
            target_q_values = torch.min(
                self.target_critic_1(next_observations, new_next_actions),
                self.target_critic_2(next_observations, new_next_actions),
            )

        if self.backup_entropy:
            target_q_values = target_q_values - alpha * next_log_pi

        td_target = rewards + (1.0 - dones) * self.discount * target_q_values.detach()
        qf1_loss = F.mse_loss(q1_predicted, td_target.detach())
        qf2_loss = F.mse_loss(q2_predicted, td_target.detach())

        # CQL
        batch_size = actions.shape[0]
        action_dim = actions.shape[-1]
        cql_random_actions = actions.new_empty(
            (batch_size, self.cql_n_actions, action_dim), requires_grad=False
        ).uniform_(-1, 1)
        cql_current_actions, cql_current_log_pis = self.actor(
            observations, repeat=self.cql_n_actions
            )
        cql_next_actions, cql_next_log_pis = self.actor(
            next_observations, repeat=self.cql_n_actions
            )
        cql_current_actions, cql_current_log_pis = (
            cql_current_actions.detach(),
            cql_current_log_pis.detach(),
        )
        cql_next_actions, cql_next_log_pis = (
            cql_next_actions.detach(),
            cql_next_log_pis.detach(),
        )


        cql_current_actions = cql_current_actions.reshape(batch_size, self.cql_n_actions, action_dim)
        cql_next_actions = cql_next_actions.reshape(batch_size, self.cql_n_actions, action_dim)
        cql_current_log_pis = cql_current_log_pis.reshape(batch_size, self.cql_n_actions)
        cql_next_log_pis = cql_next_log_pis.reshape(batch_size, self.cql_n_actions)

        cql_q1_rand = self.critic_1(observations, cql_random_actions)
        cql_q2_rand = self.critic_2(observations, cql_random_actions)
        cql_q1_current_actions = self.critic_1(observations, cql_current_actions)
        cql_q2_current_actions = self.critic_2(observations, cql_current_actions)
        cql_q1_next_actions = self.critic_1(observations, cql_next_actions)
        cql_q2_next_actions = self.critic_2(observations, cql_next_actions)


        # Calibration
        lower_bounds = mc_returns.reshape(-1, 1).repeat(
            1, cql_q1_current_actions.shape[1]
        )

        num_vals = torch.sum(lower_bounds == lower_bounds)
        bound_rate_cql_q1_current_actions = (
            torch.sum(cql_q1_current_actions < lower_bounds) / num_vals
        )
        bound_rate_cql_q2_current_actions = (
            torch.sum(cql_q2_current_actions < lower_bounds) / num_vals
        )
        bound_rate_cql_q1_next_actions = (
            torch.sum(cql_q1_next_actions < lower_bounds) / num_vals
        )
        bound_rate_cql_q2_next_actions = (
            torch.sum(cql_q2_next_actions < lower_bounds) / num_vals
        )

        """ Cal-QL: bound Q-values with MC return-to-go """
        if self._calibration_enabled:
            cql_q1_current_actions = torch.maximum(cql_q1_current_actions, lower_bounds)
            cql_q2_current_actions = torch.maximum(cql_q2_current_actions, lower_bounds)
            cql_q1_next_actions = torch.maximum(cql_q1_next_actions, lower_bounds)
            cql_q2_next_actions = torch.maximum(cql_q2_next_actions, lower_bounds)
 

        cql_cat_q1 = torch.cat(
            [
                cql_q1_rand,
                torch.unsqueeze(q1_predicted, 1),
                cql_q1_next_actions,
                cql_q1_current_actions,
            ],
            dim=1,
        )
 
        cql_cat_q2 = torch.cat(
            [
                cql_q2_rand,
                torch.unsqueeze(q2_predicted, 1),
                cql_q2_next_actions,
                cql_q2_current_actions,
            ],
            dim=1,
        )
        cql_std_q1 = torch.std(cql_cat_q1, dim=1)
        cql_std_q2 = torch.std(cql_cat_q2, dim=1)

        if self.cql_importance_sample:
            random_density = np.log(0.5**action_dim)
            cql_cat_q1 = torch.cat(
                [
                    cql_q1_rand - random_density,
                    cql_q1_next_actions - cql_next_log_pis.detach(),
                    cql_q1_current_actions - cql_current_log_pis.detach(),
                ],
                dim=1,
            )
            cql_cat_q2 = torch.cat(
                [
                    cql_q2_rand - random_density,
                    cql_q2_next_actions - cql_next_log_pis.detach(),
                    cql_q2_current_actions - cql_current_log_pis.detach(),
                ],
                dim=1,
            )

        cql_qf1_ood = torch.logsumexp(cql_cat_q1 / self.cql_temp, dim=1) * self.cql_temp
        cql_qf2_ood = torch.logsumexp(cql_cat_q2 / self.cql_temp, dim=1) * self.cql_temp

        """Subtract the log likelihood of data"""
        cql_qf1_diff = torch.clamp(
            cql_qf1_ood - q1_predicted,
            self.cql_clip_diff_min,
            self.cql_clip_diff_max,
        ).mean()
        cql_qf2_diff = torch.clamp(
            cql_qf2_ood - q2_predicted,
            self.cql_clip_diff_min,
            self.cql_clip_diff_max,
        ).mean()

        if self.cql_lagrange:
            alpha_prime = torch.clamp(
                torch.exp(self.log_alpha_prime()), min=0.0, max=1000000.0
            )
            cql_min_qf1_loss = (
                alpha_prime
                * self.cql_alpha
                * (cql_qf1_diff - self.cql_target_action_gap)
            )
            cql_min_qf2_loss = (
                alpha_prime
                * self.cql_alpha
                * (cql_qf2_diff - self.cql_target_action_gap)
            )

            self.alpha_prime_optimizer.zero_grad()
            alpha_prime_loss = (-cql_min_qf1_loss - cql_min_qf2_loss) * 0.5
            alpha_prime_loss.backward(retain_graph=True)
            self.alpha_prime_optimizer.step()
        else:
            cql_min_qf1_loss = cql_qf1_diff * self.cql_alpha
            cql_min_qf2_loss = cql_qf2_diff * self.cql_alpha
            alpha_prime_loss = observations['rgb'].new_tensor(0.0)
            alpha_prime = observations['rgb'].new_tensor(0.0)

        qf_loss = qf1_loss + qf2_loss + cql_min_qf1_loss + cql_min_qf2_loss

        log_dict.update(
            dict(
                qf1_loss=qf1_loss.item(),
                qf2_loss=qf2_loss.item(),
                alpha=alpha.item(),
                average_qf1=q1_predicted.mean().item(),
                average_qf2=q2_predicted.mean().item(),
                average_target_q=target_q_values.mean().item(),
            )
        )

        log_dict.update(
            dict(
                cql_std_q1=cql_std_q1.mean().item(),
                cql_std_q2=cql_std_q2.mean().item(),
                cql_q1_rand=cql_q1_rand.mean().item(),
                cql_q2_rand=cql_q2_rand.mean().item(),
                cql_min_qf1_loss=cql_min_qf1_loss.mean().item(),
                cql_min_qf2_loss=cql_min_qf2_loss.mean().item(),
                cql_qf1_diff=cql_qf1_diff.mean().item(),
                cql_qf2_diff=cql_qf2_diff.mean().item(),
                cql_q1_current_actions=cql_q1_current_actions.mean().item(),
                cql_q2_current_actions=cql_q2_current_actions.mean().item(),
                cql_q1_next_actions=cql_q1_next_actions.mean().item(),
                cql_q2_next_actions=cql_q2_next_actions.mean().item(),
                alpha_prime_loss=alpha_prime_loss.item(),
                alpha_prime=alpha_prime.item(),
                bound_rate_cql_q1_current_actions=bound_rate_cql_q1_current_actions.item(),  # noqa
                bound_rate_cql_q2_current_actions=bound_rate_cql_q2_current_actions.item(),  # noqa
                bound_rate_cql_q1_next_actions=bound_rate_cql_q1_next_actions.item(),
                bound_rate_cql_q2_next_actions=bound_rate_cql_q2_next_actions.item(),
            )
        )

        return qf_loss, alpha_prime, alpha_prime_loss

    def train(self, batch: TensorBatch) -> Dict[str, float]:
        (
            observations,
            next_observations,
            actions,
            rewards,
            dones,
            mc_returns,
        ) = batch
        self.total_it += 1

        new_actions, log_pi = self.actor(observations)

        alpha, alpha_loss = self._alpha_and_alpha_loss(observations, log_pi)

        """ Policy loss """
        policy_loss = self._policy_loss(
            observations, actions, new_actions, alpha, log_pi
        )

        log_dict = dict(
            log_pi=log_pi.mean().item(),
            policy_loss=policy_loss.item(),
            alpha_loss=alpha_loss.item(),
            alpha=alpha.item(),
        )

        """ Q function loss """
        qf_loss, alpha_prime, alpha_prime_loss = self._q_loss(
            observations,
            actions,
            next_observations,
            rewards,
            dones,
            mc_returns,
            alpha,
            log_dict,
        )

        if self.use_automatic_entropy_tuning:
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()


        self.critic_1_optimizer.zero_grad()
        self.critic_2_optimizer.zero_grad()
        qf_loss.backward()

        self.actor_optimizer.zero_grad()
        policy_loss.backward()

        self.critic_1_optimizer.step()
        self.critic_2_optimizer.step()
        self.actor_optimizer.step()

        if self.total_it % self.target_update_period == 0:
            self.update_target_network(self.soft_target_update_rate)

        return log_dict

    def state_dict(self) -> Dict[str, Any]:
        return {
            "actor": self.actor.state_dict(),
            "critic1": self.critic_1.state_dict(),
            "critic2": self.critic_2.state_dict(),
            "critic1_target": self.target_critic_1.state_dict(),
            "critic2_target": self.target_critic_2.state_dict(),
            "critic_1_optimizer": self.critic_1_optimizer.state_dict(),
            "critic_2_optimizer": self.critic_2_optimizer.state_dict(),
            "actor_optim": self.actor_optimizer.state_dict(),
            "sac_log_alpha": self.log_alpha,
            "sac_log_alpha_optim": self.alpha_optimizer.state_dict(),
            "cql_log_alpha": self.log_alpha_prime,
            "cql_log_alpha_optim": self.alpha_prime_optimizer.state_dict(),
            "total_it": self.total_it,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.actor.load_state_dict(state_dict=state_dict["actor"])
        self.critic_1.load_state_dict(state_dict=state_dict["critic1"])
        self.critic_2.load_state_dict(state_dict=state_dict["critic2"])

        self.target_critic_1.load_state_dict(state_dict=state_dict["critic1_target"])
        self.target_critic_2.load_state_dict(state_dict=state_dict["critic2_target"])

        self.critic_1_optimizer.load_state_dict(
            state_dict=state_dict["critic_1_optimizer"]
        )
        self.critic_2_optimizer.load_state_dict(
            state_dict=state_dict["critic_2_optimizer"]
        )
        self.actor_optimizer.load_state_dict(state_dict=state_dict["actor_optim"])

        self.log_alpha = state_dict["sac_log_alpha"]
        self.alpha_optimizer.load_state_dict(
            state_dict=state_dict["sac_log_alpha_optim"]
        )

        self.log_alpha_prime = state_dict["cql_log_alpha"]
        self.alpha_prime_optimizer.load_state_dict(
            state_dict=state_dict["cql_log_alpha_optim"]
        )
        self.total_it = state_dict["total_it"]


config = TrainConfig()  # или подставь свои параметры
config.cuda = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() and config.cuda else "cpu")


env_kwargs = dict(obs_mode=config.obs_mode, 
                  render_mode=config.render_mode, 
                  sim_backend="gpu", 
                  sensor_configs={"width": 128, "height": 128},
)

envs = gym.make(
  "PickCube-v1",
  control_mode="pd_ee_delta_pose",
  num_envs=config.num_envs,
  reconfiguration_freq=config.reconfiguration_freq,
  **env_kwargs
)

eval_envs = gym.make(
  "PickCube-v1",
  control_mode="pd_ee_delta_pose",
  num_envs=config.num_eval_envs,
  reconfiguration_freq=config.eval_reconfiguration_freq,
  **env_kwargs
)

os.environ["PYTHONHASHSEED"] = str(config.seed)
random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.backends.cudnn.deterministic = config.torch_deterministic
envs.action_space.seed(config.seed)
eval_envs.action_space.seed(config.seed)

envs = FlattenRGBDObservationWrapper(envs, rgb=True, depth=False, state=True)
eval_envs = FlattenRGBDObservationWrapper(eval_envs, rgb=True, depth=False, state=True)
if isinstance(envs.action_space, gym.spaces.Dict):
    envs = FlattenActionSpaceWrapper(envs)
    eval_envs = FlattenActionSpaceWrapper(eval_envs)
envs = ManiSkillVectorEnv(envs, config.num_envs, ignore_terminations=not config.partial_reset, record_metrics=True)
eval_envs = ManiSkillVectorEnv(eval_envs, config.num_eval_envs, ignore_terminations=not config.eval_partial_reset, record_metrics=True)


state_dim = envs.single_observation_space
action_dim = np.prod(envs.single_action_space.shape)

offline_buffer = ReplayBuffer(
    env=envs,
    num_envs=config.num_envs,
    buffer_size=config.buffer_size,
    storage_device=torch.device(config.buffer_device),
    sample_device=device
)

online_buffer = ReplayBuffer(
    env=envs,
    num_envs=config.num_envs,
    buffer_size=config.buffer_size,
    storage_device=torch.device(config.buffer_device),
    sample_device=device
)

offline_buffer.load_from_dataset('/home/user10_2/VER_FULL_PickCube-v1', config)

# TRY NOT TO MODIFY: start the game
obs, info = envs.reset(seed=config.seed) # in Gymnasium, seed is given to reset() instead of seed()
eval_obs, _ = eval_envs.reset(seed=config.seed)
max_action = envs.action_space.high[0]

actor = TanhGaussianPolicy(
    envs,
    obs,
    max_action,
    orthogonal_init=config.orthogonal_init,
    device = device
).to(device)
actor_optimizer = torch.optim.Adam(actor.parameters(), config.policy_lr)

critic_1 = FullyConnectedQFunction(
    envs,
    actor.encoder,
    config.orthogonal_init,
    config.q_n_hidden_layers,
).to(device)

critic_2 = FullyConnectedQFunction(
    envs,
    actor.encoder,
    config.orthogonal_init,
    config.q_n_hidden_layers,
).to(device)
critic_1_optimizer = torch.optim.Adam(list(critic_1.parameters()), config.qf_lr)
critic_2_optimizer = torch.optim.Adam(list(critic_2.parameters()), config.qf_lr)

kwargs = {
    "critic_1": critic_1,
    "critic_2": critic_2,
    "critic_1_optimizer": critic_1_optimizer,
    "critic_2_optimizer": critic_2_optimizer,
    "actor": actor,
    "actor_optimizer": actor_optimizer,
    "discount": config.discount,
    "soft_target_update_rate": config.soft_target_update_rate,
    "device": device,
    # CQL
    "target_entropy": -np.prod(envs.action_space.shape).item(),
    "alpha_multiplier": config.alpha_multiplier,
    "use_automatic_entropy_tuning": config.use_automatic_entropy_tuning,
    "backup_entropy": config.backup_entropy,
    "policy_lr": config.policy_lr,
    "qf_lr": config.qf_lr,
    "bc_steps": config.bc_steps,
    "target_update_period": config.target_update_period,
    "cql_n_actions": config.cql_n_actions,
    "cql_importance_sample": config.cql_importance_sample,
    "cql_lagrange": config.cql_lagrange,
    "cql_target_action_gap": config.cql_target_action_gap,
    "cql_temp": config.cql_temp,
    "cql_alpha": config.cql_alpha,
    "cql_max_target_backup": config.cql_max_target_backup,
    "cql_clip_diff_min": config.cql_clip_diff_min,
    "cql_clip_diff_max": config.cql_clip_diff_max,
}

print("---------------------------------------")
print(f"Training Cal-QL, Env: {config.env_id}, Seed: {config.seed}")
print("---------------------------------------")

# Initialize actor
trainer = CalQL(**kwargs)
wandb_init(asdict(config))


evaluations = []
state, _ = envs.reset()
done = False

batch_size_offline = int(config.batch_size * config.mixing_ratio)
batch_size_online = config.batch_size - batch_size_offline

print("Offline pretraining")
for t in range(int(config.offline_iterations) + int(config.online_iterations)):
    if t == config.offline_iterations:
        print("Online tuning")
        trainer.switch_calibration()
        trainer.cql_alpha = config.cql_alpha_online
    if t >= config.offline_iterations:
        state = {k: (v.detach().clone().to(device=device, dtype=torch.float32) if isinstance(v, torch.Tensor) else v)
            for k, v in state.items()}
        action, _ = actor(state)
        next_state, reward, terminated, truncated, env_infos = envs.step(action)
        done = terminated | truncated
        online_buffer.add(state, next_state, action, reward, done)
        state = next_state

    if t < config.offline_iterations:
        batch = offline_buffer.sample(config.batch_size)
    else:
        offline_batch = offline_buffer.sample(batch_size_offline)
        online_batch = online_buffer.sample(batch_size_online)
        batch = []
        for i in range(2):
            merged = {}
            for k in offline_batch[i].keys():
                merged[k] = torch.vstack([offline_batch[i][k], online_batch[i][k]])
            batch.append(merged)
        for i in range(2, len(offline_batch)):
            batch.append(torch.cat([offline_batch[i], online_batch[i]], dim=0))
    log_dict = trainer.train(batch)
    log_dict["offline_iter" if t < config.offline_iterations else "online_iter"] = (
        t if t < config.offline_iterations else t - config.offline_iterations
    )
    wandb.log(log_dict, step=trainer.total_it)
    # Evaluate episode
    if (t + 1) % config.eval_freq == 0:
        print(f"Time steps: {t + 1}")
        eval_scores, eval_successes = eval_actor(
            eval_envs,
            actor,
            device=device,
            n_episodes=config.num_eval_episodes,
            seed=config.seed,
        )
        eval_score = eval_scores.mean()
        eval_success = eval_successes.mean()
        eval_log = {}
        eval_log["eval/score"] = eval_score
        eval_log["eval/sr"] = eval_success
        print("---------------------------------------")
        print(
            f"Evaluation over {config.num_eval_episodes} episodes: "
            f"eval score: {eval_score:.3f}, sr: {eval_success:.3f}"
        )
        torch.cuda.empty_cache()
        print("---------------------------------------")
        if config.checkpoint:
            torch.save(
                trainer.state_dict(),
                os.path.join(config.checkpoint, f"checkpoint_{t}.pt"),
            )
        wandb.log(eval_log, step=trainer.total_it)


