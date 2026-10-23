"""Student policy for DAgger: an MLP from deployable observations to the expert's normalised TCP
velocity actions, with per-feature normalisation fitted on the aggregated data and optional
modality masks (no wrench / no vision) for the ablations.

Observation layout (built by contact_env.ContactEnv.observe):
  [0:3]    commanded TCP offset from the episode start (m)
  [3:6]    measured TCP offset from the episode start (m)
  [6:54]   sensed wrench, last 8 policy ticks x [Fx Fy Fz tx ty tz] (N, N m), task frame, about TCP
  [54:59]  previous action
  [59:61]  vision estimate of the ring-to-rod offset (m), noisy stand-in for image features
  [61]     vision available (1/0)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

OBS_DIM, ACT_DIM = 62, 5
WRENCH = slice(6, 54)
VISION = slice(59, 62)


class Student(nn.Module):
    def __init__(self, hidden: int = 256, mask_wrench: bool = False, mask_vision: bool = False):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(OBS_DIM, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, ACT_DIM), nn.Tanh())
        self.register_buffer("mu", torch.zeros(OBS_DIM))
        self.register_buffer("sd", torch.ones(OBS_DIM))
        self.mask_wrench, self.mask_vision = mask_wrench, mask_vision

    def masked(self, obs: torch.Tensor) -> torch.Tensor:
        obs = obs.clone()
        if self.mask_wrench:
            obs[:, WRENCH] = 0
        if self.mask_vision:
            obs[:, VISION] = 0
        return obs

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = (self.masked(obs) - self.mu) / self.sd
        return self.net(x)

    def fit_norm(self, obs: torch.Tensor):
        o = self.masked(obs)
        self.mu.copy_(o.mean(0))
        self.sd.copy_(o.std(0).clamp_min(1e-6))


def load_shards(paths: list[Path]):
    obs, act, w = [], [], []
    for p in paths:
        d = np.load(p)
        obs.append(d["obs"])
        act.append(d["expert_action"])
        w.append(d["valid"].astype(np.float32))
    return np.concatenate(obs), np.concatenate(act), np.concatenate(w)


def train(obs: np.ndarray, act: np.ndarray, valid: np.ndarray, *, epochs: int = 30, lr: float = 1e-3, batch: int = 4096,
          device: str = "cpu", seed: int = 0, **kw) -> tuple[Student, dict]:
    torch.manual_seed(seed)
    keep = valid > 0
    X = torch.tensor(obs[keep], dtype=torch.float32, device=device)
    Y = torch.tensor(act[keep], dtype=torch.float32, device=device)
    n = len(X)
    perm = torch.randperm(n, device=device)
    n_val = max(1, n // 20)
    val, tr = perm[:n_val], perm[n_val:]
    model = Student(**kw).to(device)
    model.fit_norm(X[tr])
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    hist = []
    for ep in range(epochs):
        model.train()
        idx = tr[torch.randperm(len(tr), device=device)]
        tot = 0.0
        for i in range(0, len(idx), batch):
            b = idx[i:i + batch]
            loss = ((model(X[b]) - Y[b]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(b)
        sched.step()
        model.eval()
        with torch.no_grad():
            vl = float(((model(X[val]) - Y[val]) ** 2).mean())
        hist.append(dict(epoch=ep, train=tot / len(tr), val=vl))
    return model, dict(samples=int(n), history=hist)


def save(model: Student, path: Path, info: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state=model.state_dict(), mask_wrench=model.mask_wrench, mask_vision=model.mask_vision), path)
    path.with_suffix(".json").write_text(json.dumps(info, indent=1))


def load(path: Path, device="cpu") -> Student:
    ck = torch.load(path, map_location=device)
    m = Student(mask_wrench=ck["mask_wrench"], mask_vision=ck["mask_vision"]).to(device)
    m.load_state_dict(ck["state"])
    m.eval()
    return m
