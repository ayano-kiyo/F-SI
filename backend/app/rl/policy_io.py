"""Load a trained PPO checkpoint for live driving (lazy torch import)."""

from __future__ import annotations

from pathlib import Path

def policy_device(policy) -> str:
    return str(next(policy.parameters()).device)

def find_checkpoint(name: str = "ppo-racing", checkpoint_dir: str = "checkpoints") -> Path | None:
    directory = Path(checkpoint_dir)
    latest = directory / f"{name}-latest.pt"
    if latest.is_file():
        return latest
    candidates = sorted(directory.glob(f"{name}-iter*.pt"))
    return candidates[-1] if candidates else None

def load_policy(checkpoint: Path | str, device: str = "cpu"):
    import torch

    from .ppo import ActorCritic
    path = Path(checkpoint)
    data = torch.load(path, map_location=device, weights_only=False)
    policy = ActorCritic(obs_dim=int(data.get("obs_dim", 25)))
    policy.load_state_dict(data["policy"])
    policy.eval()
    policy.observation_version = int(
        data.get("observation_version", 1 if policy.obs_dim == 25 else 2)
    )
    policy.cars_per_race = int(data.get("cars_per_race", 1 if policy.obs_dim == 25 else 4))
    policy.race_mode = str(data.get("race_mode", "solo" if policy.obs_dim == 25 else "shared_policy_self_play"))
    policy.checkpoint_path = str(path)
    return policy
