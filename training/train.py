"""
Milestone 4: train a standing policy with PPO.

  python training/train.py --exp-id EXP-001 --timesteps 300000 --n-envs 8

Reads algorithm/env hyperparameters from config/training.yaml (ppo.*);
CLI flags override the file for quick experiments without editing it.

Logs TensorBoard to          logs/<exp-id>/
Periodic checkpoints to      checkpoints/<exp-id>/
Best-eval + final model to   policies/<exp-id>_best/  and  policies/<exp-id>_final.zip
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from env.biped_env import BipedEnv  # noqa: E402
from training.callbacks import EpisodeStatsCallback, SpeedTrackingEvalCallback  # noqa: E402



def make_env(seed: int, config_path: Path):
    def _init():
        return Monitor(BipedEnv(training_cfg_path=config_path, seed=seed))
    return _init



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", default="EXP-001")
    ap.add_argument("--timesteps", type=int, default=None,
                     help="override config/training.yaml ppo.total_timesteps")
    ap.add_argument("--n-envs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--config", type=str, default="training.yaml",
                    help="Config filename inside the config/ directory")
    ap.add_argument("--init-from", default=None,
                    help="warm-start from a saved policy (e.g. EXP-002_final.zip for M5 stepping)")
    args = ap.parse_args()

    config_path = REPO / "config" / args.config
    
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    ppo_cfg = cfg["ppo"]

    n_envs = args.n_envs or ppo_cfg["n_envs"]
    seed = args.seed if args.seed is not None else ppo_cfg["seed"]
    total_timesteps = args.timesteps or ppo_cfg["total_timesteps"]

    log_dir = REPO / "logs" / args.exp_id
    ckpt_dir = REPO / "checkpoints" / args.exp_id
    policy_dir = REPO / "policies"
    for d in (log_dir, ckpt_dir, policy_dir):
        d.mkdir(parents=True, exist_ok=True)

    env_fns = [make_env(seed + i, config_path) for i in range(n_envs)]
    vec_env = SubprocVecEnv(env_fns) if n_envs > 1 else DummyVecEnv(env_fns)
    eval_env = DummyVecEnv([make_env(seed + 1000, config_path)])

    model = PPO(
        "MlpPolicy", vec_env, verbose=1, seed=seed,
        n_steps=ppo_cfg["n_steps"], batch_size=ppo_cfg["batch_size"],
        n_epochs=ppo_cfg["n_epochs"], gamma=ppo_cfg["gamma"],
        gae_lambda=ppo_cfg["gae_lambda"], learning_rate=ppo_cfg["learning_rate"],
        clip_range=ppo_cfg["clip_range"], ent_coef=ppo_cfg["ent_coef"],
        target_kl=ppo_cfg.get("target_kl"),  # None = SB3 default (no early stop)
        tensorboard_log=str(log_dir),
        policy_kwargs=dict(net_arch=[256, 256]),
    )
    if args.init_from:
        # Same obs/action spaces and net_arch across stages, so the weights
        # (incl. log_std) drop straight in; hyperparameters stay this config's.
        model.set_parameters(args.init_from, exact_match=True, device=model.device)
        print(f"[{args.exp_id}] warm-started from {args.init_from}")

    callbacks = [
        EpisodeStatsCallback(),
        CheckpointCallback(save_freq=max(50_000 // n_envs, 1), save_path=str(ckpt_dir),
                            name_prefix=args.exp_id),
        EvalCallback(eval_env, best_model_save_path=str(policy_dir / f"{args.exp_id}_best"),
                     log_path=str(log_dir), eval_freq=max(10_000 // n_envs, 1),
                     n_eval_episodes=5, deterministic=True),
    ]
    # Milestone 6: with a non-zero forward command range, also keep the model
    # that tracks speed best -> policies/<exp-id>_best_speed.zip
    sweep_every = cfg.get("training", {}).get("speed_sweep_every", 0)
    if sweep_every:
        callbacks.append(SpeedTrackingEvalCallback(
            BipedEnv(training_cfg_path=config_path, seed=seed + 2000),
            save_path=str(policy_dir / f"{args.exp_id}_best_speed.zip"),
            eval_freq=int(sweep_every)))

    print(f"[{args.exp_id}] training {total_timesteps} steps, {n_envs} envs, seed {seed}")
    model.learn(total_timesteps=total_timesteps, callback=CallbackList(callbacks), tb_log_name=args.exp_id)

    final_path = policy_dir / f"{args.exp_id}_final.zip"
    model.save(str(final_path))
    print(f"[{args.exp_id}] saved final model -> {final_path}")


if __name__ == "__main__":
    main()
