# UR5 twin-wrist action space

## Locked 9D contract

Legacy evidence shows that normal SpaceMouse control ends in RTDE `speedL`. The adapter therefore preserves the
existing Cartesian-velocity semantics; it does not rewrite the controller as joint targets.

| i | Field | Unit/frame | Positive direction | Representation | Accepted range |
|---:|---|---|---|---|---|
| 0 | UR base TCP `vx` | m/s, UR base | base +X | instantaneous velocity | normal ±0.107; worker hard norm 0.25 |
| 1 | UR base TCP `vy` | m/s, UR base | base +Y | instantaneous velocity | normal ±0.107; worker hard norm 0.25 |
| 2 | UR base TCP `vz` | m/s, UR base | base +Z | instantaneous velocity | normal ±0.107; worker hard norm 0.25 |
| 3 | UR base TCP `wx` | rad/s, UR base | right-hand rule about +X | instantaneous velocity | normal ±0.54; worker hard norm 0.60 |
| 4 | UR base TCP `wy` | rad/s, UR base | right-hand rule about +Y | instantaneous velocity | normal ±0.54; worker hard norm 0.60 |
| 5 | UR base TCP `wz` | rad/s, UR base | right-hand rule about +Z | instantaneous velocity | normal ±0.54; worker hard norm 0.60 |
| 6 | external wrist FE target | rad | firmware/controller +FE | absolute position | board-authoritative; source candidate [-0.9425, 1.0996] |
| 7 | external wrist RU target | rad | firmware/controller +RU | absolute position | board-authoritative; source candidate [-0.4538, 0.7330] |
| 8 | gripper target | normalized | closing | absolute position | [0,1], 0=open, 1=closed |

The linear and angular limits are vector-norm limits, not independent simultaneous maxima. The FE/RU values above
are candidates from the audited firmware source (`-54..+63°`, `-26..+42°`); the active OpenRB board returns the
authoritative bounds through `GET_OUTPUT_CL`. Anatomical FE/RU positive directions still require physical verification.

State uses exactly:

```text
[UR actual joint q 6 rad, wrist actual FE/RU 2 rad, measured gripper position 1 normalized]
```

Capture is 30 Hz, accepted HDF5 samples default to 10 Hz, and the low-level teleoperation loop runs at 125 Hz.

## SpaceMouse-to-action path

```text
spnav raw axes [x,y,z,rx,ry,rz]
  → divide by 500 → clip [-1,1] → deadzone 0.12
  → [-z,x,y,-rz,rx,ry]
  → normal translation / Shift rotation / Ctrl FE-RU routing
  → vector speed limits and legacy workspace guard
  → speedL RPC
  → accepted command snapshot
  → action[0:6]

wrist controller bounded absolute target → action[6:8]
single-owner gripper commanded target    → action[8]
```

Raw axes are telemetry only. They are never saved as training action.

The legacy UR worker can still floor-clamp negative Z after the parent constructs a twist, and HOME or keys 1/2 use
`speedJ`. To keep every accepted action consistent with the hardware command, the new sink:

- publishes one immutable motion/twist/`target_qd` receipt only after each 125 Hz command cycle completes;
- rejects an episode if any nonzero `target_qd` occurs in its task window;
- rejects commands whose projected Z could activate the worker floor clamp;
- rejects speed/numeric/state/synchronization violations;
- seals at Fit before reset/HOME motion.

Thus an accepted task frame has `speedL` semantics and cannot mix fields from adjacent command cycles. The legacy RPC
ack still does not echo the worker's post-floor-guard twist, so equality with that low-level value is established by the
conservative floor invariant. Returning the applied twist in the worker ack remains a desirable future hardening step.

## π0.5 transform

The first six dimensions are already velocities, so no delta-action transform is applied. Wrist and gripper remain
absolute. `PadStatesAndActions` pads the 9 real dimensions to model `action_dim=32`; policy output is cropped back to
the first 9 dimensions. Model horizon is 10.
