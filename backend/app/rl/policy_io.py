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

def validate_policy_metadata(obs_dim: int, cars_per_race: int) -> tuple[int, int]:
    """Reject checkpoint metadata that cannot match the solo or race observation layout."""
    if obs_dim == 25 and cars_per_race != 1:
        raise ValueError("Solo checkpoints must declare cars_per_race=1")
    if obs_dim != 25 and (obs_dim != 31 or not 2 <= cars_per_race <= 4):
        raise ValueError("Race checkpoints must use obs_dim=31 and declare 2-4 cars")
    return obs_dim, cars_per_race


def load_policy(checkpoint: Path | str, device: str = "cpu"):
    import torch

    from .ppo import ActorCritic
    path = Path(checkpoint)
    data = torch.load(path, map_location=device, weights_only=False)
    obs_dim = int(data.get("obs_dim", 25))
    cars_per_race = int(data.get("cars_per_race", 1 if obs_dim == 25 else 4))
    validate_policy_metadata(obs_dim, cars_per_race)
    policy = ActorCritic(obs_dim=obs_dim)
    policy.load_state_dict(data["policy"])
    policy.eval()
    policy.observation_version = int(
        data.get("observation_version", 1 if policy.obs_dim == 25 else 2)
    )
    policy.cars_per_race = cars_per_race
    policy.race_mode = str(data.get("race_mode", "solo" if policy.obs_dim == 25 else "shared_policy_self_play"))
    policy.checkpoint_path = str(path)
    return policy
