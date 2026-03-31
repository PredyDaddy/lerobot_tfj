# ACT Distillation Launch Notes

## Scope

This run package starts ACT distillation from the teacher:

- `/data/tfj/lerobot_tfj/outputs/act_grasp_block_in_bin1/checkpoints/last`

It uses the teacher's own training line as the canonical base:

- dataset: `/home/cqy/.cache/huggingface/lerobot/admin123/grasp_block_in_bin1`
- base config: `/data/tfj/lerobot_tfj/outputs/act_grasp_block_in_bin1/checkpoints/last/pretrained_model/train_config.json`

## Student choice

The current Stage-2 decoder KD implementation compares `decoder_out` directly and does not wire projection into the default training path. Because of that, the student keeps:

- `dim_model=512`
- `n_heads=8`
- `chunk_size=100`
- `n_action_steps=100`

Compression is applied through:

- `n_encoder_layers: 4 -> 2`
- `dim_feedforward: 3200 -> 1024`
- `latent_dim: 32 -> 16`
- `n_vae_encoder_layers: 4 -> 2`

This reduces the ACT policy from about `51.6M` params to about `23.1M` params while staying decoder-KD compatible.

## Run types

- `run_smoke.sh`: short launch to validate startup, processor compatibility, KD wiring, backward, and logging.
- `run_full.sh`: full Stage-2 launch with conservative decoder-KD schedule.

## Output dirs

- smoke: `/data/tfj/lerobot_tfj/outputs/act_distill_grasp_block_in_bin1_stage2_smoke_20260315_153740`
- full: `/data/tfj/lerobot_tfj/outputs/act_distill_grasp_block_in_bin1_stage2_20260315_153740`

## Notes

- Always use repo-local entry:
  - `PYTHONPATH=/data/tfj/lerobot_tfj/src python -m lerobot.scripts.lerobot_train`
- Do not use the globally installed `lerobot-train` wrapper for this run.
- Force `--dataset.video_backend=pyav` for this dataset in the current environment.
  - The cached `AV1` videos opened with `PyAV/libdav1d`, while `torchcodec` failed during the smoke run.
