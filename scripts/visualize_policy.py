"""
Watch a trained PPO policy drive BR-1 in the MuJoCo viewer.

  python scripts/visualize_policy.py --model policies/EXP-002_final.zip
  python scripts/visualize_policy.py --model policies/EXP-004_final.zip --config training_step.yaml

ctrl+right-click-drag on the robot in the viewer applies a push force --
a quick, informal way to probe push-recovery beyond the Milestone 2 ankle
strategy (the learned policy owns every joint, including balance).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
from stable_baselines3 import PPO

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from env.biped_env import BipedEnv  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stochastic", action="store_true")
    ap.add_argument("--config", default="training.yaml",
                    help="Config filename inside config/ (training_step.yaml for M5 policies)")
    args = ap.parse_args()

    model = PPO.load(args.model)
    env = BipedEnv(training_cfg_path=REPO / "config" / args.config, seed=args.seed)
    obs, _ = env.reset(seed=args.seed)

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        wall = time.perf_counter()
        while viewer.is_running():
            action, _ = model.predict(obs, deterministic=not args.stochastic)
            obs, reward, terminated, truncated, info = env.step(action)
            viewer.sync()
            if terminated or truncated:
                print(f"episode ended ({info['termination_reason'] or 'timeout'}), resetting")
                obs, _ = env.reset()

            wall += env.dt_ctrl
            slack = wall - time.perf_counter()
            if slack > 0:
                time.sleep(slack)
            else:
                wall = time.perf_counter()


if __name__ == "__main__":
    main()
