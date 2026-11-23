# Hardware evaluation: publishing checklist and protocol

## Available footage

[The hardware gallery](media/hardware/README.md) contains all 15 supplied files in three numbered source groups, with checksums and three previews. The supplied files include multiple camera views; previews use view `4`. Policy identities and logs were not provided. Do not treat the number of files as an evaluation denominator or infer outcomes from filenames.

## Summary to fill after trial metadata and logs are finalized

- Robot / gripper / hardware software version: [fill]
- Physical part and *measured radial* clearance: [fill]
- Evaluated policy checkpoints: DAgger [hash]; residual actor [hash]; frozen residual base [hash]
- Feature source: kinematic offset / measured image tracker / other [fill]
- Policy runtime and safety thresholds (including any changes): [fill]
- Held-out start distribution, offset/tilt, grasp preparation: [fill]
- Pilot/training/calibration runs excluded from test evaluation: [fill]
- Number of attempted *paired* rollouts, ordering/randomization and evaluation date(s): [fill]
- Operator intervention and objective success definition: [fill]

## Results (do not publish a fabricated denominator)

| Policy | Successful / attempted | Median time (all vs successful trials) | Median / p95 peak force | Stops / interventions | Failure modes |
| --- | --- | --- | --- | --- | --- |
| Replay | [ ] / [ ] | [ ] / [ ] s | [ ] / [ ] N | [ ] | [ ] |
| DAgger (same deployable features) | [ ] / [ ] | [ ] / [ ] s | [ ] / [ ] N | [ ] | [ ] |
| Residual RL on frozen DAgger | [ ] / [ ] | [ ] / [ ] s | [ ] / [ ] N | [ ] | [ ] |

Add Wilson intervals for proportions; for matched pairs report the 2x2 discordance table (DAgger only succeeds / residual only succeeds), since the same starts are paired. Also report policy seeds and whether RL's higher success required higher contact force.

## Three decisive tests for the hardware gap

1. **Matched RL versus DAgger:** same object geometry, initial offsets, gripper setup, control rate, observation source, force stops and allowed retries. Interleave policy order to reduce drift/learning or reset bias.
2. **Contact-recovery mechanism:** group failures by jam, loss of grasp, false force trigger, stall, timeout. Compare the wrench sequence and XY corrections immediately after first contact for each policy. Check whether RL truly recovers more often rather than merely pushing harder.
3. **Calibration ablation:** train or evaluate measured, deliberately mismatched and measurement-centered randomized sensor/contact configurations; keep network and training budget constant. Evaluate on held-out hardware starts and avoid tuning on the final test set.

## Real rollout CSV schema (one row per attempt)

`trial_id,session_date,condition_id,policy,checkpoint_hash,seed,init_offset_x_mm,init_offset_y_mm,init_tilt_deg,measured_radial_clearance_mm,grasp_id,observation_source,success,completion_time_s,peak_force_n,peak_torque_nm,time_over_soft_force_s,termination_reason,operator_intervention,video_path,log_path,notes`

Preserve original 500 Hz sensor timestamps and 30 Hz action timestamps in the detailed logs, noting frame, reference point, force compensation and filter latency. Use one scripted file for aggregate metrics to avoid manual cherry-picking.

## Video guidance

Use a consistent side/wrist angle, minimal crop, 1× playback for primary results, small policy badges (not fabricated stats), synchronized contact force plot and a timestamp. Ensure the footage matches the logged trial and state whether repeated or cherry-picked. If presenting a comparison, prefer simultaneous side-by-side **matched conditions**, not arbitrary visually similar clips.

Suggested file names after adding final edited copies:

- `docs/media/hardware/01_matched_comparison.mp4`
- `docs/media/hardware/02_rl_contact_recovery.mp4`
- `docs/media/hardware/03_heldout_robustness.mp4`

GitHub's README player support varies: use a linked thumbnail/poster from README; host web-optimized MP4s on the portfolio site and embed there with an HTML `<video>` element with `controls`, captions, and a poster image. Preserve original experiment videos separately.
