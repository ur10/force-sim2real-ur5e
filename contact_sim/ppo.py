"""Minimal PPO for the residual (torch): Gaussian actor on deployable observations, critic with
privileged inputs (asymmetric), running observation normalisation, GAE with time-limit truncation
handled by bootstrapping and invalid (post-terminal) steps masked out.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


class RunningNorm(nn.Module):
    def __init__(self, dim: int, clip: float = 10.0):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.tensor(1e-4))
        self.clip = clip

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        b_mean, b_var, b_n = x.mean(0), x.var(0, unbiased=False), x.shape[0]
        delta, tot = b_mean - self.mean, self.count + b_n
        self.mean += delta * b_n / tot
        self.var = (self.var * self.count + b_var * b_n + delta ** 2 * self.count * b_n / tot) / tot
        self.count = tot

    def forward(self, x):
        return ((x - self.mean) / (self.var.sqrt() + 1e-6)).clamp(-self.clip, self.clip)


def mlp(i, o, h=256, out_act=None):
    layers = [nn.Linear(i, h), nn.ELU(), nn.Linear(h, h), nn.ELU(), nn.Linear(h, h), nn.ELU(), nn.Linear(h, o)]
    if out_act is not None:
        layers.append(out_act)
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, priv_dim: int, act_dim: int, init_log_std: float = -1.2):
        super().__init__()
        self.obs_norm = RunningNorm(obs_dim)
        self.crit_norm = RunningNorm(obs_dim + priv_dim)
        self.actor = mlp(obs_dim, act_dim, out_act=nn.Tanh())
        self.critic = mlp(obs_dim + priv_dim, 1)
        self.log_std = nn.Parameter(torch.full((act_dim,), init_log_std))
        with torch.no_grad():  # start as a near-zero residual
            self.actor[-2].weight.mul_(0.01)
            self.actor[-2].bias.zero_()

    def dist(self, obs):
        mu = self.actor(self.obs_norm(obs))
        return torch.distributions.Normal(mu, self.log_std.exp().expand_as(mu))

    def value(self, obs, priv):
        return self.critic(self.crit_norm(torch.cat([obs, priv], -1)))[:, 0]

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        d = self.dist(obs)
        a = d.mean if deterministic else d.sample()
        return a, d.log_prob(a).sum(-1)


@dataclass
class PPOCfg:
    lr: float = 3e-4
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    epochs: int = 5
    minibatches: int = 4
    vf_coef: float = 0.5
    ent_coef: float = 0.0
    max_grad_norm: float = 1.0


class PPO:
    def __init__(self, model: ActorCritic, cfg: PPOCfg):
        self.m, self.cfg = model, cfg
        self.opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    def gae(self, rew, val, next_val, done, valid, trunc_val):
        """All (T, N). done: terminal at this step (success / failure). valid: env running at the
        start of the step. trunc_val (N): value of the state after the last step for envs that are
        still running (time-limit or rollout boundary bootstrap)."""
        T = rew.shape[0]
        adv = torch.zeros_like(rew)
        last = torch.zeros_like(trunc_val)
        for t in reversed(range(T)):
            nv = trunc_val if t == T - 1 else val[t + 1]
            nv = torch.where(done[t], torch.zeros_like(nv), nv)
            delta = rew[t] + self.cfg.gamma * nv - val[t]
            last = delta + self.cfg.gamma * self.cfg.lam * (~done[t]).float() * last
            last = torch.where(valid[t], last, torch.zeros_like(last))
            adv[t] = last
        return adv, adv + val

    def update(self, obs, priv, act, logp_old, adv, ret, valid) -> dict:
        c = self.cfg
        keep = valid.reshape(-1)
        O, Pv, A = obs.reshape(-1, obs.shape[-1])[keep], priv.reshape(-1, priv.shape[-1])[keep], act.reshape(-1, act.shape[-1])[keep]
        LP, AD, RT = logp_old.reshape(-1)[keep], adv.reshape(-1)[keep], ret.reshape(-1)[keep]
        n = len(O)
        if n < max(2 * c.minibatches, 32):
            # nearly every episode already finished in this chunk (strong base policies): too few samples
            # for a stable update, and the std of a single advantage is NaN (which poisoned a run)
            return dict(pi=0.0, vf=0.0, kl=0.0, clipfrac=0.0, n=n, skipped=1.0)
        AD = (AD - AD.mean()) / (AD.std(unbiased=False) + 1e-8)
        mb = max(1, n // c.minibatches)
        stats = dict(pi=0.0, vf=0.0, kl=0.0, clipfrac=0.0, n=n)
        k = 0
        for _ in range(c.epochs):
            perm = torch.randperm(n, device=O.device)
            for i in range(0, n, mb):
                b = perm[i:i + mb]
                d = self.m.dist(O[b])
                lp = d.log_prob(A[b]).sum(-1)
                ratio = (lp - LP[b]).exp()
                pg = -torch.min(ratio * AD[b], ratio.clamp(1 - c.clip, 1 + c.clip) * AD[b]).mean()
                vf = ((self.m.value(O[b], Pv[b]) - RT[b]) ** 2).mean()
                ent = d.entropy().sum(-1).mean()
                loss = pg + c.vf_coef * vf - c.ent_coef * ent
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.m.parameters(), c.max_grad_norm)
                self.opt.step()
                with torch.no_grad():
                    stats["pi"] += float(pg)
                    stats["vf"] += float(vf)
                    stats["kl"] += float((LP[b] - lp).mean())
                    stats["clipfrac"] += float(((ratio - 1).abs() > c.clip).float().mean())
                k += 1
        for key in ("pi", "vf", "kl", "clipfrac"):
            stats[key] /= max(k, 1)
        stats["std"] = float(self.m.log_std.exp().mean())
        return stats
