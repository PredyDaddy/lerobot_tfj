# GROOT RL Bundle

This bundle collects the GROOT RL training and SO101 deployment artifacts that were discussed and updated in the current work session.

The bundle is organized for quick lookup:

- `docs/`: human-readable notes and indexes
- `scripts/`: copied user-facing launch scripts
- `reference_src/`: copied Python source snapshots for the core implementation files

Current status summary:

- Stage-2 GROOT offline RL training is bundled here as a dataset-only workflow.
- Real-robot default inference remains on the safer stage-1 GROOT weights.
- Stage-2 RL weights are opt-in only and should not be treated as the default robot deployment policy.
- The SO101 GROOT runtime now has a direct no-save path and no longer needs dataset recording for inference.

Directory layout:

- `docs/README.md`
- `docs/groot_rl_how_rl_is_added_20260324_zh.md`
- `docs/groot_rl_complete_knowledge_for_ppt_20260318_zh.md`
- `docs/groot_rl_architecture_and_operations_20260318_zh.md`
- `docs/groot_rl_python_source_index_20260318_zh.md`
- `docs/openclaw_groot_server.md`
- `scripts/README.md`
- `scripts/run_groot_so101_infer.sh`
- `scripts/run_so101_pickplace_infer.sh`
- `scripts/run_so101_policy_record.sh`
- `scripts/openclaw_groot_server.py`
- `scripts/train_groot_grasp_block_in_bin1_offline_rl_stage2.sh`
- `scripts/train_groot_grasp_block_in_bin1_offline_stage2_rl.sh`
- `reference_src/src/lerobot/scripts/lerobot_run_so101_pickplace.py`
- `reference_src/src/lerobot/scripts/lerobot_record_so101_policy.py`
- `reference_src/src/lerobot/scripts/lerobot_train_groot_hybrid.py`
- `reference_src/src/lerobot/configs/train_groot_hybrid.py`
- `reference_src/src/lerobot/rl/groot_hybrid/offline_replay.py`
- `reference_src/src/lerobot/rl/groot_hybrid/trainer.py`

Recommended reading order:

1. `docs/groot_rl_architecture_and_operations_20260318_zh.md`
2. `docs/groot_rl_complete_knowledge_for_ppt_20260318_zh.md`
3. `docs/groot_rl_python_source_index_20260318_zh.md`
4. `scripts/README.md`
