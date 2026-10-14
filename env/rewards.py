"""
Reward function for BR-1 locomotion.

Reward terms are computed from PRIVILEGED, noiseless simulation state (base
linear velocity included) -- that is standard and correct: the reward
function is a training-time-only oracle, unlike the observation the policy
actually sees (env/observations.py), which must stay realistic.

R = w_vel * lin_vel_track + w_ang * ang_vel_track + w_up * upright
  + w_h * height + w_alive * alive
  - w_tau * torque - w_ar * action_rate - w_ja * joint_acc
  - w_slip * foot_slip - w_col * collision
  + w_gait * gait_contact + w_clear * swing_clearance        (Milestone 5)
  - fall_penalty * (1 if terminated by falling else 0)

REWARD-HACKING GUARDS (Part 10) -- what each penalty is there to stop:
  torque, action_rate, joint_acc  -> jittery, high-frequency, energy-wasting
                                      policies; also discourages "vibrating
                                      in place" to farm the alive bonus.
  foot_slip                        -> sliding feet instead of stepping
                                      (only meaningful once stepping starts,
                                      M5+; harmless at M4 with cmd=0).
  collision (non-foot ground hit)  -> "walking" by dragging a knee/torso,
                                      or falling forward while still racking
                                      up alive/velocity reward before the
                                      termination check fires.
  height term                      -> discourages crouching low to farm a
                                      cheap upright/stability score, or
                                      jumping (a height term alone doesn't
                                      stop jumping -- action_rate/torque and
                                      the foot-contact-based alive/height
                                      combination discourage the aerial
                                      phase a jump requires).
  tracking (not raw speed) reward  -> rewarding raw forward SPEED (instead
                                      of tracking a commanded speed) lets the
                                      policy farm reward by falling forward
                                      with increasing velocity; tracking a
                                      *bounded* command with a Gaussian
                                      kernel caps the achievable reward at
                                      "moving exactly at the requested
                                      speed", removing that incentive.
  gait_contact + swing_clearance   -> (M5) at slow commands, velocity
                                      tracking alone barely separates
                                      "stepping" from "leaning/shuffling"
                                      (0.1 m/s error costs exp(-0.04)=0.96).
                                      These pay for lifting the foot the
                                      gait clock says is in swing; binary
                                      contact alone could be farmed with a
                                      1 mm hover, hence the clearance term.
                                      Both are 0-weight at stage 'stand'.

NORMALIZATION (added after EXP-001, revised after EXP-001b/c):
EXP-001's torque penalty was raw sum(tau^2) in Nm^2, unbounded up to
~172,800 across 12 joints -- under an UNTRAINED policy's near-maximal
exploration noise it regularly swamped the well-behaved-state total of
about +3 to +4 (alive + upright + height + tracking), making "end the
episode immediately" locally better than "keep exploring toward standing."
EXP-001's curve showed exactly that signature: reward/ep_len fell for the
first ~450k steps while mean_tilt_deg stayed flat at ~13-16 deg (well under
the 40 deg fall threshold) -- episodes were ending via height collapse
under penalty pressure, not by toppling.

The first fix (bounding torque + an unmeasured guess at a joint_acc scale)
was incomplete: an actual random-action measurement (500 steps, see
scripts/test_env.py-style probe in the Milestone 4 writeup) showed
joint_acc and foot_slip -- NOT torque -- were the real dominant terms,
reaching unweighted means of -22.7 and -11.6 and single-step worsts of
-64.3 and -88.7 respectively, versus torque's worst of only -0.27 once
normalized. Guessing a normalization constant without measuring was the
mistake; every penalty term below is now EXPLICITLY CLIPPED before being
normalized, so each unweighted term is hard-bounded to [-1, 0] no matter
how chaotic the motion gets -- the clip bounds (ACC_CLIP, SLIP_CLIP) are
INITIAL ASSUMPTIONs (generous "this would already be violent" values), not
measured hardware limits; revisit once gait data from a standing policy
exists. With every penalty bounded to [-1,0] and current weights
(sum ~2.15), the worst possible per-step penalty is now ~-2.15, well below
a single alive/upright/height/tracking step's worth (~+3 to +4) -- so
surviving an additional step can no longer be worse than ending the
episode, removing the perverse incentive by construction rather than by
hoping the scale is small enough.
"""
from __future__ import annotations

import numpy as np

ACC_CLIP = 400.0   # rad/s^2 -- INITIAL ASSUMPTION, "already violently fast"
SLIP_CLIP = 2.0     # m/s -- INITIAL ASSUMPTION, "already clearly sliding"


def compute_reward(state: dict, weights: dict, tracking_sigma: float,
                    fall_penalty: float, lin_vel_sigma: float | None = None) -> tuple[float, dict]:
    r = {}

    # lin_vel_sigma (M6) narrows only the forward-velocity kernel; None keeps
    # the shared tracking_sigma, i.e. the M3-M5 reward exactly.
    lin_sigma = tracking_sigma if lin_vel_sigma is None else lin_vel_sigma
    lin_err = float(np.sum((state["base_linvel"][:2] - state["cmd"][:2]) ** 2))
    r["lin_vel_track"] = float(np.exp(-lin_err / lin_sigma))

    ang_err = float((state["base_angvel_z"] - state["cmd"][2]) ** 2)
    r["ang_vel_track"] = float(np.exp(-ang_err / tracking_sigma))

    r["upright"] = float(np.exp(-np.sum(np.square(state["projected_gravity"][:2])) / tracking_sigma))
    r["height"] = float(np.exp(-((state["base_height"] - state["height_target"]) ** 2) / tracking_sigma))
    r["alive"] = 1.0

    # Every penalty below: clip to a generous physical bound FIRST, then
    # normalize by that same bound -> unweighted term is always in [-1, 0].
    torque_frac = state["torque"] / state["torque_limit"]  # already in [-1,1] (actuator-clamped)
    r["torque"] = -float(np.mean(np.square(torque_frac)))

    action_diff = np.clip(state["action"] - state["prev_action"], -2.0, 2.0) / 2.0
    r["action_rate"] = -float(np.mean(np.square(action_diff)))

    joint_acc = np.clip((state["qvel"] - state["prev_qvel"]) / state["dt"], -ACC_CLIP, ACC_CLIP) / ACC_CLIP
    r["joint_acc"] = -float(np.mean(np.square(joint_acc)))

    slip_terms = []
    for in_contact, foot_xy_vel in zip(state["foot_contact"], state["foot_xy_vel"]):
        if in_contact:
            speed = np.clip(np.linalg.norm(foot_xy_vel), 0.0, SLIP_CLIP) / SLIP_CLIP
            slip_terms.append(speed ** 2)
    r["foot_slip"] = -float(np.mean(slip_terms)) if slip_terms else 0.0

    r["collision"] = -1.0 if state["non_foot_contact"] else 0.0

    # Milestone 5 stepping terms, both in [0, 1]. Keys are optional so older
    # callers (tests, M3/M4 probes) that don't supply gait state still work.
    if "desired_stance" in state:
        matches = [c == d for c, d in zip(state["foot_contact"], state["desired_stance"])]
        r["gait_contact"] = float(np.mean(matches))
        clear = [np.clip(h, 0.0, state["swing_height"]) / state["swing_height"]
                 for h, d in zip(state["foot_height"], state["desired_stance"]) if not d]
        r["swing_clearance"] = float(np.mean(clear)) if clear else 0.0

    total = sum(weights.get(name, 0.0) * value for name, value in r.items())
    if state["terminated_by_fall"]:
        total -= fall_penalty

    r["total"] = float(total)
    return float(total), r
