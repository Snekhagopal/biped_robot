"""
Evaluate a trained BR-1 policy (Part 17 gait/stability metrics).

  python training/evaluate.py --model policies/EXP-001_final.zip --episodes 20
  python training/evaluate.py --model policies/EXP-004_final.zip --config training_step.yaml --episodes 30

Stepping metrics (Milestone 5) are reported over episodes whose command
asked for motion (|cmd| >= gait.min_cmd_speed): forward speed vs command,
distance covered, touchdowns per foot, and how often foot contacts matched
the gait clock.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from env.biped_env import BipedEnv  # noqa: E402

STEP_LIFT_M = 0.02  # swing-foot rise that counts as a step, not contact chatter
SWEEP_CMDS = (0.0, 0.1, 0.2, 0.3, 0.4)  # Milestone 6 speed-tracking sweep, m/s
SWEEP_SETTLE_S = 2.0  # skip the start-up transient when measuring speed


def speed_sweep(model, env: BipedEnv, commands=SWEEP_CMDS, seed: int = 0,
                deterministic: bool = True) -> dict:
    """Milestone 6: one full episode per pinned forward command. Speed is
    measured after SWEEP_SETTLE_S; a fall counts as zero speed for the rest
    of the episode (so falling is never better than walking slowly).
    Returns per-command rows plus mean |vx error| and the least-squares slope
    of achieved vs commanded speed (1.0 = perfect modulation; EXP-005 ~0)."""
    rows = []
    for i, c in enumerate(commands):
        obs, _ = env.reset(seed=seed + i, options={"cmd": [c, 0.0, 0.0]})
        settle = int(round(SWEEP_SETTLE_S / env.dt_ctrl))
        x_settle, steps, fell = None, 0, False
        done = False
        while not done:
            if steps == settle:
                x_settle = float(env.data.qpos[0])
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, _, terminated, truncated, info = env.step(action)
            steps += 1
            fell = info["termination_reason"].startswith("fell")
            done = terminated or truncated
        window_s = (env.max_episode_steps - settle) * env.dt_ctrl
        dx = float(env.data.qpos[0]) - x_settle if x_settle is not None else 0.0
        vx = dx / window_s
        rows.append({"cmd_vx": c, "vx": vx, "err": abs(vx - c), "fell": fell})
    cmds = np.array([r["cmd_vx"] for r in rows])
    vxs = np.array([r["vx"] for r in rows])
    cc = cmds - cmds.mean()
    slope = float(cc @ (vxs - vxs.mean()) / (cc @ cc)) if (cc @ cc) > 0 else float("nan")
    return {"rows": rows, "mean_abs_err": float(np.mean([r["err"] for r in rows])),
            "slope": slope, "falls": int(sum(r["fell"] for r in rows))}


PUSH_MAGS = (0.25, 0.5, 0.75, 1.0)  # Milestone 7 base-velocity kicks, m/s
PUSH_DIRS = {"+x": (1, 0), "-x": (-1, 0), "+y": (0, 1), "-y": (0, -1)}
PUSH_AT_S = 3.0


def push_test(model, env: BipedEnv, cmd_vx: float = 0.0, mags=PUSH_MAGS, seed: int = 0,
              deterministic: bool = True) -> dict:
    """Milestone 7: one episode per (direction, magnitude); kick the base
    velocity once at PUSH_AT_S and record whether the robot survives the
    full episode. M2's hand-tuned ankle controller recovered ~0.25 m/s."""
    survived = {}
    for d, (ux, uy) in PUSH_DIRS.items():
        for mag in mags:
            obs, _ = env.reset(seed=seed, options={"cmd": [cmd_vx, 0.0, 0.0]})
            push_step = int(round(PUSH_AT_S / env.dt_ctrl))
            done, steps, fell = False, 0, False
            while not done:
                if steps == push_step:
                    env.data.qvel[0] += ux * mag
                    env.data.qvel[1] += uy * mag
                action, _ = model.predict(obs, deterministic=deterministic)
                obs, _, terminated, truncated, info = env.step(action)
                steps += 1
                fell = info["termination_reason"].startswith("fell")
                done = terminated or truncated
            survived[(d, mag)] = not fell
    return {"cmd_vx": cmd_vx, "mags": list(mags), "survived": survived,
            "rate": float(np.mean(list(survived.values())))}


def print_push(push: dict) -> None:
    print(f"  -- push recovery (cmd {push['cmd_vx']:.2f} m/s, kick at {PUSH_AT_S:.0f} s) --")
    print("    dir  " + "  ".join(f"{m:>5.2f}" for m in push["mags"]) + "  m/s")
    for d in PUSH_DIRS:
        print(f"    {d:<4} " + "  ".join(f"{'ok' if push['survived'][(d, m)] else 'FELL':>5}"
                                     for m in push["mags"]))
    print(f"  survived {push['rate'] * 100:.0f} % of kicks")


def print_sweep(sweep: dict) -> None:
    print("  -- speed sweep (pinned commands, speed measured after "
          f"{SWEEP_SETTLE_S:.0f} s) --")
    for r in sweep["rows"]:
        print(f"    cmd {r['cmd_vx']:.2f} m/s -> {r['vx']:+.3f} m/s"
              f"  (|err| {r['err']:.3f}){'  FELL' if r['fell'] else ''}")
    print(f"  mean |err| {sweep['mean_abs_err']:.3f} m/s, slope {sweep['slope']:.2f}"
          f" (1.0 = speed follows command), falls {sweep['falls']}/{len(sweep['rows'])}")


def evaluate(
    model_path: str,
    episodes: int,
    seed: int = 1234,
    deterministic: bool = True,
    config_path: Path | None = None,
    sweep: bool = False,
    push: bool = False,
) -> dict:
    model = PPO.load(model_path)
    env = BipedEnv(
    training_cfg_path=config_path if config_path is not None else REPO / "config" / "training.yaml",
    seed=seed,
)

    ep_rewards, ep_lengths_s = [], []
    falls = 0
    tilt_all: list[float] = []
    step_eps: list[dict] = []  # Milestone 5 per-episode stepping stats

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        total_r = 0.0
        steps = 0
        termination_reason = ""
        x0 = float(env.data.qpos[0])
        lifted = [False, False]
        touchdowns = [0, 0]
        gait_match: list[float] = []
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, r, terminated, truncated, info = env.step(action)
            total_r += r
            steps += 1
            tilt_all.append(info["tilt_deg"])
            termination_reason = info["termination_reason"]
            # A touchdown only counts after a real lift (>= STEP_LIFT_M):
            # raw contact flickers tens of times per second even while standing.
            for i, (c, h) in enumerate(zip(info["foot_contact"], info["foot_height"])):
                lifted[i] |= h >= STEP_LIFT_M
                if c and lifted[i]:
                    touchdowns[i] += 1
                    lifted[i] = False
            gait_match.append(info["reward_terms"].get("gait_contact", np.nan))
            done = terminated or truncated
            if done:
                print(f"Episode {ep + 1} ended: {termination_reason or 'timeout'}"
                      f"  cmd_vx={info['cmd'][0]:.2f}  dx={info['base_pos'][0] - x0:+.2f} m"
                      f"  touchdowns L/R={touchdowns[0]}/{touchdowns[1]}")
        if env._stepping():
            t = steps * env.dt_ctrl
            dx = float(env.data.qpos[0]) - x0
            step_eps.append({"cmd_vx": float(env._cmd[0]), "vx": dx / t, "dx": dx,
                             "steps_per_s": sum(touchdowns) / t,
                             "min_foot_touchdowns": min(touchdowns),
                             "gait_match": float(np.nanmean(gait_match))})
        ep_rewards.append(total_r)
        ep_lengths_s.append(steps * env.dt_ctrl)
        if termination_reason.startswith("fell"):
            falls += 1

    result = {
        "mean_reward": float(np.mean(ep_rewards)),
        "std_reward": float(np.std(ep_rewards)),
        "mean_episode_length_s": float(np.mean(ep_lengths_s)),
        "fall_rate": falls / episodes,
        "mean_tilt_deg": float(np.mean(tilt_all)),
        "episodes": episodes,
        "stepping_episodes": len(step_eps),
    }
    if step_eps:
        result.update({
            "mean_cmd_vx": float(np.mean([e["cmd_vx"] for e in step_eps])),
            "mean_vx": float(np.mean([e["vx"] for e in step_eps])),
            "mean_abs_vx_err": float(np.mean([abs(e["vx"] - e["cmd_vx"]) for e in step_eps])),
            "mean_dx_m": float(np.mean([e["dx"] for e in step_eps])),
            "mean_steps_per_s": float(np.mean([e["steps_per_s"] for e in step_eps])),
            "min_foot_touchdowns": int(min(e["min_foot_touchdowns"] for e in step_eps)),
            "mean_gait_match": float(np.mean([e["gait_match"] for e in step_eps])),
        })

    print("=" * 60)
    print(f"EVALUATION  {model_path}  ({episodes} episodes)")
    print("=" * 60)
    print(f"  mean episode reward : {result['mean_reward']:.2f} +/- {result['std_reward']:.2f}")
    print(f"  mean episode length : {result['mean_episode_length_s']:.2f} s")
    print(f"  fall rate           : {result['fall_rate'] * 100:.1f} %  ({falls}/{episodes})")
    print(f"  mean tilt           : {result['mean_tilt_deg']:.2f} deg")
    if step_eps:
        print(f"  -- stepping episodes ({len(step_eps)}/{episodes}, |cmd| >= {env.gait_min_cmd} m/s) --")
        print(f"  forward speed       : {result['mean_vx']:.3f} m/s  (cmd {result['mean_cmd_vx']:.3f},"
              f" mean |err| {result['mean_abs_vx_err']:.3f})")
        print(f"  distance per episode: {result['mean_dx_m']:.2f} m")
        print(f"  step rate           : {result['mean_steps_per_s']:.2f} touchdowns/s"
              f"  (fewest per foot in any episode: {result['min_foot_touchdowns']})")
        print(f"  gait-clock match    : {result['mean_gait_match'] * 100:.1f} %")
    if sweep:
        result["sweep"] = speed_sweep(model, env, seed=seed, deterministic=deterministic)
        print_sweep(result["sweep"])
    if push:
        result["push"] = [push_test(model, env, cmd_vx=c, seed=seed, deterministic=deterministic)
                          for c in (0.0, 0.2)]
        for p in result["push"]:
            print_push(p)
    print("=" * 60)
    return result


def main():
    
    ap = argparse.ArgumentParser()
    
    ap.add_argument("--model", required=True)
    ap.add_argument("--config", type=str, default="training.yaml",
                help="Config filename inside the config/ directory")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--stochastic", action="store_true", help="sample actions instead of argmax-mean")
    ap.add_argument("--sweep", action="store_true",
                    help=f"also run one episode at each pinned command {SWEEP_CMDS} m/s (M6)")
    ap.add_argument("--push", action="store_true",
                    help=f"also run the push-recovery grid {PUSH_MAGS} m/s x {list(PUSH_DIRS)} (M7)")

    args = ap.parse_args()
    config_path = REPO / "config" / args.config
    evaluate(
        args.model,
        args.episodes,
        args.seed,
        deterministic=not args.stochastic,
        config_path=config_path,
        sweep=args.sweep,
        push=args.push,
    )



if __name__ == "__main__":
    main()
