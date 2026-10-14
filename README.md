# BR-1 — Bipedal Humanoid: Simulation + RL Locomotion Framework

First-generation simulated biped (**BR-1**) built in MuJoCo, trained with
reinforcement learning to stand, step, and walk — designed from day 1 to
transfer onto real hardware.

- **Robot:** 1.39 m, 32 kg, 12 actuated DOF (6 per leg) + 6-DOF floating base.
- **Actuation:** torque-source motors + a Python PD position loop (Option C).
- **Policy rate:** 50 Hz over 500 Hz physics (decimation 10).
- Full spec with confidence tags: [`config/robot.yaml`](config/robot.yaml).

## Status

| Milestone | Description | State |
|-----------|-------------|-------|
| M0 | Dev environment | verified |
| M1 | Static model: XML, joints, mass, collision, stable sim | verified |
| M2 | Low-level PD + standing balance controller | verified |
| M3 | Gymnasium environment: obs/action/reward/termination | verified |
| M4 | Standing RL policy (PPO) | achieved — EXP-002, 0% fall rate on 30 held-out episodes |
| M5 | Stepping RL policy (PPO) | achieved — EXP-005, 0% falls, alternating steps in all 25 moving held-out episodes |
| M6 | Walking at the commanded speed | achieved for 0–0.3 m/s — EXP-006, 0/30 falls, mean speed error 0.039 m/s; tops out ~0.25 m/s |
| **M7** | **Robust walking (randomized physics + pushes)** | **stage 1 achieved — EXP-007, 0/30 falls under randomization (EXP-006: 9/30); standing push recovery still 0.25 m/s fore/aft** |
| M8 | Speed range above 0.25 m/s, turning, terrain | not started |
| M9 | Sim-to-real | needs hardware |

**Best policies so far:**
- Standing: `policies/EXP-002_final.zip` — 0/30 held-out falls, mean tilt 2.84°.
- Stepping: `policies/EXP-005_step.zip` — 0/30 held-out falls; walks ~0.13 m/s
  with 12–14 lifts per foot per 10 s whenever commanded to move.
- Walking at commanded speed: `policies/EXP-006_walk.zip` — 0/30 falls;
  0.1 / 0.2 / 0.3 m/s commands → 0.10 / 0.19 / 0.25 m/s.
- Robust walking: `policies/EXP-007_robust.zip` — 0/30 falls with random
  friction, torso mass, motor strength and 0.5 m/s pushes.

See [`experiments/EXP-registry.md`](experiments/EXP-registry.md) for the full
run history, including EXP-001's reward-normalization bug and EXP-004's
PPO collapse. Watch them:
`python scripts/visualize_policy.py --model policies/EXP-002_final.zip`
`python scripts/visualize_policy.py --model policies/EXP-005_step.zip --config training_step.yaml`

## Repository layout

```
config/     robot.yaml (spec), training.yaml (stage 'stand', M3/M4), training_step.yaml (M5),
            training_walk.yaml (M6), training_robust.yaml (M7)
mujoco/     robot.xml (model), scene.xml (model + floor + lights) <- load this
robot/      sensors.py (noise model); kinematics.py/dynamics.py added when M5 needs them
env/        biped_env.py, observations.py, rewards.py, termination.py (M3 — done)
control/    pd_controller.py, actuator.py, balance_controller.py (M2 — done)
training/   train.py, evaluate.py, callbacks.py (added M4)
scripts/    test_robot.py, visualize.py, test_controller.py, test_env.py
tests/      pytest model/reward/obs checks
experiments/EXP-registry.md — one row per training run, append only
```

## Setup (Milestone 0)

Windows PowerShell, from the repo root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

GPU (CUDA 12.1) PyTorch, then the rest:

```powershell
pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

CPU-only is fine through Milestone 6 — just `pip install -r requirements.txt`.

Verify:

```powershell
python -c "import mujoco, gymnasium, stable_baselines3, torch; print('mujoco', mujoco.__version__, '| torch', torch.__version__, '| cuda', torch.cuda.is_available())"
```

## Verify the model (Milestone 1)

```powershell
python scripts/test_robot.py            # structure + contact/GRF + numeric robustness
pytest tests/test_model.py              # fast model checks
```

Expected: `total mass 32.000 kg`, `nu 12 / nv 18 / nq 19`, all checks PASS.
The report also logs "time-to-topple under pure joint PD" (~1.8 s) — that's
expected, not a failure: joint PD holds posture, not balance. See Milestone 2.

## Verify the standing controller (Milestone 2)

```powershell
python scripts/test_controller.py       # posture-topple sanity + 10s hold + push recovery
pytest                                   # full suite (model + controller)
python scripts/visualize.py             # watch it stand indefinitely
python scripts/visualize.py --posture   # compare: posture-only PD, topples ~1.8s
```

Expected: static hold passes (tilt < 10°, both feet loaded, 10 s no fall),
push-recovery passes at 0.15 m/s in both directions, and the info line
reports the measured envelope (~0.25 m/s recoverable in this build). In the
viewer, ctrl+right-click-drag on the robot applies a push you can feel it
resist (or fall to, if you push hard enough — that boundary is real, not a
bug: a fixed-footprint ankle strategy is physically limited by the foot's
support polygon; bigger disturbances need Milestone 7/8 or a learned policy).

## Verify the RL environment (Milestone 3)

```powershell
python scripts/test_env.py     # gymnasium API, determinism, reward sanity, robustness
pytest                          # full suite: model + controller + env
```

Expected: all checks pass, including a zero-action episode that topples at
~1.8 s (the env is not secretly stabilizing anything — Milestone 4's job is
to train a policy that keeps the standing controller's stability without
hard-coding it).

Environment summary: `obs` is 47-dim (projected gravity, base angular
velocity, velocity command, joint positions/velocities relative to nominal,
previous action, gait-phase clock — see `env/observations.py`); `action` is
a 12-dim residual on the nominal joint pose, scaled ±0.30 rad and converted
to torque by the same `PDController` from Milestone 2 (`env/biped_env.py`).
`config/training.yaml` is the curriculum knob: at `curriculum_stage: stand`
all velocity commands are pinned to 0, so `lin_vel_track` simply rewards not
drifting. Later stages widen the command ranges — no environment code
changes.

## Stepping (Milestone 5)

```bash
# train: warm-start from the standing policy, then continue with KL-capped PPO
python training/train.py --config training_step.yaml --exp-id EXP-004 --init-from policies/EXP-002_final.zip
python training/train.py --config training_step.yaml --exp-id EXP-005 --init-from checkpoints/EXP-004/EXP-004_2850000_steps.zip
# evaluate (adds stepping metrics) and watch
python training/evaluate.py --model policies/EXP-005_step.zip --config training_step.yaml --episodes 30 --seed 2026
python scripts/visualize_policy.py --model policies/EXP-005_step.zip --config training_step.yaml
```

`config/training_step.yaml` is the stand config plus the M5 changes. The
environment code only gained optional features, so the stand stage behaves
exactly as before (all M3 checks and `pytest` still pass):

- **Gait clock** at a fixed `gait.frequency_hz` (1.5 Hz) whenever
  `|cmd| >= gait.min_cmd_speed` (0.05 m/s); frozen otherwise. (EXP-003's
  speed-linked clock gave a 20 s step period at 0.05 m/s.)
- **Command** `lin_vel_x` sampled from 0–0.3 m/s (EXP-003's fixed 0.05 m/s
  was indistinguishable from standing in the tracking reward).
- **Rewards** `gait_contact` (foot contacts match the clock; both feet down
  when told to stand) and `swing_clearance` (swing sole rises up to 5 cm,
  so a 1 mm hover doesn't count). Both are weight 0 in `training.yaml`.
- **PPO** `learning_rate` 1e-4 and `target_kl` 0.02. EXP-004 ran at 3e-4
  without a KL cap, learned to step (2.85M ckpt: 0/20 falls) and then
  collapsed to 100% falls within 150k steps.
- **`train.py --init-from`** warm-starts from any saved policy.

Success criteria (set for M5, since none existed): on 30 held-out episodes,
≤10% falls, both feet stepping in every moving episode, forward speed within
50% of the command on average. Result for `EXP-005_step.zip`: 0/30 falls,
≥12 lifts per foot in every moving episode, 0.131 m/s vs 0.169 m/s mean
command, 87% gait-clock match. Steps are counted only when the sole's lowest
corner rises ≥2 cm, because raw contact flickers dozens of times per second
even while standing.

**Known limitation, handed to M6:** the policy walks at one pace (~0.13 m/s)
whatever the command, and continuing training past 2M made it drift toward
stepping in place (final 4M policy: 0.03 m/s). The gait rewards out-earn
speed tracking at these slow commands. M6 needs a sharper velocity-tracking
term and checkpoint selection on velocity error rather than eval reward.

## Walking at the commanded speed (Milestone 6)

```bash
python training/train.py --config training_walk.yaml --exp-id EXP-006 --init-from policies/EXP-005_step.zip
python training/evaluate.py --model policies/EXP-006_walk.zip --config training_walk.yaml --episodes 30 --seed 2026 --sweep
python scripts/visualize_policy.py --model policies/EXP-006_walk.zip --config training_walk.yaml
```

`config/training_walk.yaml` changes only config, plus two optional env features:

- **`reward.lin_vel_sigma: 0.02`**: a narrow kernel for forward-velocity
  tracking only (weight 1.5 → 3.0). At the shared 0.25 width, EXP-005's
  one-pace gait kept ≥89% of the tracking reward at every command, so
  the gait rewards won.
- **Command** `lin_vel_x` 0–0.4 m/s.
- **`training.speed_sweep_every: 100000`**: every 100k steps, run one
  episode at each pinned command 0/0.1/0.2/0.3/0.4 m/s and keep the
  fall-free model with the lowest mean speed error as
  `policies/<exp-id>_best_speed.zip`. This replaces picking by eval reward,
  which picked EXP-005's step-in-place policy.
- `evaluate.py --sweep` prints the same sweep, including the **slope** of
  achieved vs commanded speed (1.0 = speed follows the command; EXP-005:
  0.16, EXP-006: 0.62). `env.reset(options={"cmd": [vx, vy, wz]})` pins a
  command.

Success criteria: ≤10% falls on 30 held-out episodes and mean speed error
≤0.05 m/s. EXP-006 (1.3M checkpoint): 0/30 falls, 0.039 m/s.

**Known limitations:** speed saturates at ~0.25 m/s (a 0.4 m/s command gives
0.23). The likely cause is the fixed 1.5 Hz gait clock, which is untested.
As in EXP-005, training past the best checkpoint drifted back toward one
slow pace (error 0.046 → 0.10 by 2M), so the run was stopped at 2.2M.

## Robust walking (Milestone 7)

```bash
python training/train.py --config training_robust.yaml --exp-id EXP-007 --init-from policies/EXP-006_walk.zip
python training/evaluate.py --model policies/EXP-007_robust.zip --config training_robust.yaml --episodes 30 --seed 2026 --sweep --push
python scripts/visualize_policy.py --model policies/EXP-007_robust.zip --config training_robust.yaml
```

`config/training_robust.yaml` is `training_walk.yaml` plus two optional
blocks, which are off in every earlier config:

- **`domain_rand`**: re-drawn every episode. Floor and sole friction ×0.5–1.25,
  torso mass −1 to +3 kg (inertia scaled to match), and per-joint motor
  strength ×0.85–1.15. These ranges are initial assumptions sized to the
  uncertainty tags in `config/robot.yaml`, not measured hardware spread.
- **`pushes`**: a horizontal base-velocity kick of up to 0.5 m/s per axis
  every 3–6 s.
- `evaluate.py --push` kicks the robot once at 3 s, at 0.25/0.5/0.75/1.0 m/s
  in ±x/±y, while standing and while walking at 0.2 m/s.

Results for `EXP-007_robust.zip` (3.4M checkpoint):

| | EXP-006 | EXP-007 |
|---|---|---|
| Falls, 30 episodes with randomized physics + pushes | 9/30 | **0/30** |
| Push grid on nominal physics, standing | 31% | 38% |
| Push grid on nominal physics, walking 0.2 m/s | 38% | **56%** (survives 0.5 m/s fore/aft) |
| Speed error, nominal physics sweep | 0.045 m/s | 0.053 m/s |

**Known limitations, handed on:**
- Standing push recovery is still only 0.25 m/s fore/aft.
- Training fall rate plateaued at ~30–40% from ~500k steps. Starting pushes at
  0.5 m/s is probably too hard (assumption); next, ramp pushes from 0.3 m/s.
- The `--push` grid on the robust config uses one seed for all 16 cells, so
  a single bad physics draw fails a whole row. Read the grid on
  `training_walk.yaml`.
- The fall-free-only rule for `_best_speed.zip` saved just one checkpoint
  (3.4M) in 40 sweeps under random pushes.

On macOS, the MuJoCo viewer needs `mjpython` instead of `python` for
`scripts/visualize*.py`.
