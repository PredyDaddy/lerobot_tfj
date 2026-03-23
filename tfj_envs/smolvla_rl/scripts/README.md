# SmolVLA RL Scripts

This directory collects the scripts that are directly relevant to the current SmolVLA hybrid RL workflow.

Scripts:

- `launch_smolvla_offline_trimmed_train.sh`
  - Launch a single foreground offline SmolVLA training run on the trimmed SO101 dataset.

- `start_smolvla_offline_trimmed_train_nohup.sh`
  - Start the same offline training in background and launch a monitor process.

- `launch_smolvla_hybrid_train.sh`
  - Template launcher for the current hybrid SmolVLA RL trainer.
  - Requires an env type and task because the current hybrid trainer only works when `env` is configured.

- `start_smolvla_hybrid_train_nohup.sh`
  - Background wrapper for hybrid training plus monitor process.

- `monitor_training_process.sh`
  - Generic background monitor that samples process state, GPU state, and latest log signals.

- `run_so101_policy_record.sh`
  - Calls the new SO101-focused policy recording entrypoint with safe offline defaults.
  - Now defaults to the completed RL checkpoint at `smolvla_hybrid_aloha_live_20260316_111221/checkpoints/last/pretrained_model`.
  - Normalizes shell-style bool inputs such as `CLEAR_DATASET_ROOT=0/1` into `false/true` for `draccus`.

- `show_trimmed_dataset_meta.sh`
  - Prints `info.json` and `tasks.parquet` for the trimmed dataset.

Notes:

- The offline training scripts default to `pyav` because the trimmed dataset videos are AV1 and the verified stable path here was `pyav`.
- The SO101 policy recording wrapper defaults to offline Hugging Face / Transformers mode to avoid the tokenizer fetch retries that previously blocked inference on a network-isolated machine.
- None of these scripts delete output directories by default.
