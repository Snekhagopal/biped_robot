"""
Custom SB3 callback for BR-1 training metrics.

SB3's own Monitor wrapper already logs ep_rew_mean / ep_len_mean to
TensorBoard. This adds the BR-1-specific numbers from Part 17/19 that
aren't generic RL metrics: fall rate (vs. timeout) and mean tilt, read from
the `info` dict env/biped_env.py attaches to every step.
"""
from __future__ import annotations

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class EpisodeStatsCallback(BaseCallback):
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._fall_count = 0
        self._episode_count = 0
        self._tilt_buf: list[float] = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        for info, done in zip(infos, dones):
            if "tilt_deg" in info:
                self._tilt_buf.append(info["tilt_deg"])
            if done:
                self._episode_count += 1
                if info.get("termination_reason", "").startswith("fell"):
                    self._fall_count += 1
        return True

    def _on_rollout_end(self) -> None:
        if self._episode_count > 0:
            self.logger.record("br1/fall_rate", self._fall_count / self._episode_count)
            self.logger.record("br1/episodes_this_rollout", self._episode_count)
        if self._tilt_buf:
            self.logger.record("br1/mean_tilt_deg", float(np.mean(self._tilt_buf)))
        self._fall_count = 0
        self._episode_count = 0
        self._tilt_buf = []


class SpeedTrackingEvalCallback(BaseCallback):
    """Milestone 6 checkpoint selection. EXP-005's EvalCallback kept the
    highest-*reward* model, which by 4M steps was one stepping in place --
    the gait terms out-earned speed tracking. This runs the pinned-command
    speed sweep (training/evaluate.py) every `eval_freq` env steps and keeps
    the model with the lowest mean |vx error|; a sweep with any fall is
    never kept."""

    def __init__(self, env, save_path: str, eval_freq: int, verbose: int = 1):
        super().__init__(verbose)
        self.env = env
        self.save_path = save_path
        self.eval_freq = eval_freq
        self.best_err = float("inf")
        self._last_eval = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval < self.eval_freq:
            return True
        self._last_eval = self.num_timesteps
        from training.evaluate import speed_sweep  # avoid an import cycle at module load
        sweep = speed_sweep(self.model, self.env)
        self.logger.record("br1/sweep_mean_abs_vx_err", sweep["mean_abs_err"])
        self.logger.record("br1/sweep_vx_slope", sweep["slope"])
        self.logger.record("br1/sweep_falls", sweep["falls"])
        speeds = " ".join(f"{r['vx']:+.2f}" for r in sweep["rows"])
        kept = sweep["falls"] == 0 and sweep["mean_abs_err"] < self.best_err
        if kept:
            self.best_err = sweep["mean_abs_err"]
            self.model.save(self.save_path)
        if self.verbose:
            print(f"[sweep @{self.num_timesteps}] vx {speeds} | |err| {sweep['mean_abs_err']:.3f}"
                  f" slope {sweep['slope']:.2f} falls {sweep['falls']}"
                  f"{'  -> new best' if kept else ''}", flush=True)
        return True
