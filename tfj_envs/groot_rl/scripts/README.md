# GROOT RL Scripts

This directory stores copied user-facing scripts related to the GROOT RL and SO101 deployment work.

Files:

- `run_groot_so101_infer.sh`
  - Direct no-save GROOT inference on SO101
- `train_groot_grasp_block_in_bin1_offline_rl_stage2.sh`
  - Canonical dataset-only offline RL stage-2 launcher
- `train_groot_grasp_block_in_bin1_offline_stage2_rl.sh`
  - Compatibility alias for the stage-2 launcher
- `openclaw_groot_server.py`
  - OpenClaw GROOT server entrypoint
- `run_so101_pickplace_infer.sh`
  - Backend router for SO101 inference
- `run_so101_policy_record.sh`
  - SO101 record-mode launcher when dataset capture is still desired

Notes:

- These are copied snapshots placed here for centralized lookup.
- The authoritative runtime files still live at the repository root `scripts/` paths.
