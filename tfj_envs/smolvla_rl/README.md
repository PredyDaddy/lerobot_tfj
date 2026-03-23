# SmolVLA RL Bundle

This bundle archives the current SmolVLA hybrid RL work and related training / runtime tooling under a single directory.

Status summary:

- The repository currently contains a first-pass `SmolVLA hybrid training` implementation.
- This implementation is not a full real-robot SO101 online RL stack.
- The current code path is best understood as:
  - offline SmolVLA supervised training
  - then hybrid RL fine-tuning in a standard simulated env
  - then SO101 policy-on-robot evaluation / recording

Directory layout:

- `docs/README.md`
- `docs/smolvla_rl_architecture_and_integration_20260315_zh.md`
- `docs/smolvla_rl_training_and_operations_20260315_zh.md`
- `docs/reviews/`
- `scripts/README.md`
- `scripts/launch_smolvla_offline_trimmed_train.sh`
- `scripts/start_smolvla_offline_trimmed_train_nohup.sh`
- `scripts/launch_smolvla_hybrid_train.sh`
- `scripts/start_smolvla_hybrid_train_nohup.sh`
- `scripts/monitor_training_process.sh`
- `scripts/run_so101_policy_record.sh`
- `scripts/show_trimmed_dataset_meta.sh`

Recommended reading order:

1. `docs/smolvla_rl_architecture_and_integration_20260315_zh.md`
2. `docs/smolvla_rl_training_and_operations_20260315_zh.md`
3. `scripts/README.md`
