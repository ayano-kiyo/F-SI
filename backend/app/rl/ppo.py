"""Minimal PPO for the racing vec env (follows the SB3/Spinning-Up formulation)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from .env import ACT_DIM, OBS_DIM

@dataclass
class PPOConfig:
    hidden: tuple[int, ...] = (256, 256)
    activation: str = "tanh"
    lr: float = 3e-4
    gamma: float = 0.995
    lam: float = 0.95
    clip: float = 0.2
    epochs: int = 6
    minibatches: int = 8
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    rollout_steps: int = 512
    total_steps: int = 2_000_000

def _activation(name: str) -> nn.Module:
    return nn.Tanh() if name == "tanh" else nn.ReLU()

class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int = OBS_DIM, act_dim: int = ACT_DIM,
                 hidden: tuple[int, ...] = (256, 256), activation: str = "tanh"):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        trunk: list[nn.Module] = []
        last = obs_dim
        for width in hidden:
            layer = nn.Linear(last, width)
            nn.init.orthogonal_(layer.weight, gain=math.sqrt(2.0))
            nn.init.zeros_(layer.bias)
            trunk += [layer, _activation(activation)]
            last = width
        self.trunk = nn.Sequential(*trunk)
        self.mean = nn.Linear(last, act_dim)
        nn.init.orthogonal_(self.mean.weight, gain=0.01)
        nn.init.zeros_(self.mean.bias)
        # Start with measured steering and a positive throttle prior. A unit
        # standard deviation in normalized steering can send a cold policy
        # straight into a wall or another car before PPO has learned anything.
        self.log_std = nn.Parameter(torch.full((act_dim,), math.log(0.35)))
        with torch.no_grad():
            self.mean.bias[1] = 0.7
        self.value = nn.Linear(last, 1)
        nn.init.orthogonal_(self.value.weight, gain=1.0)
        nn.init.zeros_(self.value.bias)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feat = self.trunk(obs)
        return self.mean(feat), self.value(feat).squeeze(-1), feat

    def act(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, value, _ = self.forward(obs)
        std = self.log_std.exp().expand_as(mean)
        dist = torch.distributions.Normal(mean, std)
        raw = dist.sample()
        log_prob = dist.log_prob(raw).sum(-1)
        action = torch.tanh(raw)
        corrected = log_prob - torch.log(1 - action.pow(2) + 1e-6).sum(-1)
        return action, corrected, value, dist.entropy().sum(-1)

    def evaluate(self, obs: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, value, _ = self.forward(obs)
        std = self.log_std.exp().expand_as(mean)
        dist = torch.distributions.Normal(mean, std)
        raw = torch.atanh(torch.clamp(action, -1 + 1e-5, 1 - 1e-5))
        log_prob = dist.log_prob(raw).sum(-1) - torch.log(1 - action.pow(2) + 1e-6).sum(-1)
        return log_prob, value, dist.entropy().sum(-1)

class RolloutBuffer:
    def __init__(self, steps: int, envs: int, obs_dim: int = OBS_DIM):
        self.obs = np.zeros((steps, envs, obs_dim), dtype=np.float32)
        self.actions = np.zeros((steps, envs, ACT_DIM), dtype=np.float32)
        self.log_probs = np.zeros((steps, envs), dtype=np.float32)
        self.rewards = np.zeros((steps, envs), dtype=np.float32)
        self.dones = np.zeros((steps, envs), dtype=np.float32)
        self.values = np.zeros((steps, envs), dtype=np.float32)
        self.valid = np.ones((steps, envs), dtype=bool)
        self.ptr = 0
        self.steps = steps

    def add(self, obs, action, log_prob, reward, done, value, valid=None) -> None:
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.log_probs[self.ptr] = log_prob
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.values[self.ptr] = value
        if valid is not None:
            self.valid[self.ptr] = valid
        self.ptr += 1

    def advantages(self, last_values: np.ndarray, last_dones: np.ndarray, gamma: float, lam: float):
        adv = np.zeros_like(self.rewards)
        gae = np.zeros(self.rewards.shape[1])
        for t in reversed(range(self.steps)):
            nxt_non_terminal = 1.0 - (self.dones[t] if t < self.steps else last_dones)
            if t == self.steps - 1:
                nxt_values = last_values
                nxt_non_terminal = 1.0 - last_dones
            else:
                nxt_values = self.values[t + 1]
            delta = self.rewards[t] + gamma * nxt_values * nxt_non_terminal - self.values[t]
            gae = delta + gamma * lam * nxt_non_terminal * gae
            adv[t] = gae
        returns = adv + self.values
        return adv, returns

def ppo_update(policy: ActorCritic, optimizer: torch.optim.Optimizer, buffer: RolloutBuffer,
               advantages: np.ndarray, returns: np.ndarray, config: PPOConfig, device: torch.device) -> dict:
    obs_dim = buffer.obs.shape[-1]
    valid = buffer.valid.reshape(-1)
    flat_obs = torch.as_tensor(buffer.obs.reshape(-1, obs_dim)[valid], dtype=torch.float32, device=device)
    flat_act = torch.as_tensor(buffer.actions.reshape(-1, ACT_DIM)[valid], dtype=torch.float32, device=device)
    flat_logp = torch.as_tensor(buffer.log_probs.reshape(-1)[valid], dtype=torch.float32, device=device)
    flat_adv = torch.as_tensor(advantages.reshape(-1)[valid], dtype=torch.float32, device=device)
    flat_ret = torch.as_tensor(returns.reshape(-1)[valid], dtype=torch.float32, device=device)
    if flat_obs.shape[0] == 0:
        return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "clip_frac": 0.0, "updates": 0}
    flat_adv = (flat_adv - flat_adv.mean()) / (flat_adv.std() + 1e-8)
    count = flat_obs.shape[0]
    batch = max(1, count // config.minibatches)
    stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "clip_frac": 0.0, "updates": 0}
    for _ in range(config.epochs):
        perm = torch.randperm(count, device=device)
        for start in range(0, count, batch):
            idx = perm[start:start + batch]
            logp, value, entropy = policy.evaluate(flat_obs[idx], flat_act[idx])
            ratio = torch.exp(logp - flat_logp[idx])
            clipped = torch.clamp(ratio, 1 - config.clip, 1 + config.clip)
            policy_loss = -torch.min(ratio * flat_adv[idx], clipped * flat_adv[idx]).mean()
            value_loss = ((flat_ret[idx] - value) ** 2).mean()
            entropy_mean = entropy.mean()
            loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy_mean
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), config.max_grad_norm)
            optimizer.step()
            stats["policy_loss"] += float(policy_loss.detach())
            stats["value_loss"] += float(value_loss.detach())
            stats["entropy"] += float(entropy_mean.detach())
            stats["clip_frac"] += float((torch.abs(ratio - 1) > config.clip).float().mean().detach())
            stats["updates"] += 1
    for key in ("policy_loss", "value_loss", "entropy", "clip_frac"):
        stats[key] /= max(1, stats["updates"])
    return stats
