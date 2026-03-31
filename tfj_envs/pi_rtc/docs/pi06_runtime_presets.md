# PI06 Runtime Presets

## Recommended launch commands

Use the repo launcher and let `POLICY_PRESET` pick the checkpoint. Do not keep an old `POLICY_PATH` in your shell, or it will override the preset.

```bash
unset POLICY_PATH
unset TASK_TEXT
POLICY_PRESET=restart_145737 bash /data/tfj/lerobot_tfj/tfj_envs/pi_rtc/scripts/run_pi06_torch_infer_so101.sh
```

Try the earlier checkpoint like this:

```bash
unset POLICY_PATH
unset TASK_TEXT
POLICY_PRESET=restart_016193 bash /data/tfj/lerobot_tfj/tfj_envs/pi_rtc/scripts/run_pi06_torch_infer_so101.sh
```

The task text should stay aligned with training:

```bash
TASK_TEXT='Put the block in the bin'
```

## Speed benchmark

Measure pure torch inference without robot I/O:

```bash
/home/cqy/miniconda3/envs/lerobot/bin/python /data/tfj/lerobot_tfj/tfj_envs/pi_rtc/scripts/benchmark_pi_torch_runtime.py \
  --policy-path /data/tfj/Evo-RL/outputs/train/pi06_stage2_restart_9epoch_bs2_20260327_103000/stage2/checkpoints/145737/pretrained_model \
  --device cuda:0 \
  --use-amp \
  --policy-num-inference-steps 4
```

PI05 baseline:

```bash
/home/cqy/miniconda3/envs/lerobot/bin/python /data/tfj/lerobot_tfj/tfj_envs/pi_rtc/scripts/benchmark_pi_torch_runtime.py \
  --policy-path /data/tfj/lerobot_tfj/pi_model/pretrained_model \
  --device cuda:0 \
  --policy-num-inference-steps 10
```

## Measured results

Measured on this machine on 2026-03-31:

- PI06 `145737`, `num_inference_steps=4`, `use_amp=true`:
  `predict_action_chunk` mean `60.12 ms`, amortized `15.03 ms/step`, implied pure-model ceiling about `66.53 Hz`.
- PI06 `145737`, `num_inference_steps=4`, `use_amp=false`:
  `predict_action_chunk` mean `57.70 ms`, amortized `14.42 ms/step`, implied pure-model ceiling about `69.33 Hz`.
- PI05 base model, `num_inference_steps=10`:
  `predict_action_chunk` mean `94.85 ms`, but one chunk covers `50` actions, so the amortized cost is only `1.90 ms/step`.

Saved reports:

- `/data/tfj/lerobot_tfj/tfj_envs/pi_rtc/docs/results/pi_torch_benchmark_pretrained_model_20260331_113452/report.json`
- `/data/tfj/lerobot_tfj/tfj_envs/pi_rtc/docs/results/pi_torch_benchmark_pretrained_model_20260331_113616/report.json`
- `/data/tfj/lerobot_tfj/tfj_envs/pi_rtc/docs/results/pi_torch_benchmark_pretrained_model_20260331_113532/report.json`

## Notes

- `select_action` reports the averaged per-step cost after queue amortization.
- `predict_action_chunk` reports the true chunk refresh latency; this is the number that usually explains whether PI06 feels hesitant.
- The speed gap between PI05 and PI06 is mostly about `chunk_size=50` vs `chunk_size=4`, not about the weights file being damaged or the GPU suddenly becoming slow.
- If the robot still looks slow or sticky while these pure-model numbers are fast, the next places to inspect are camera capture cadence, control loop scheduling, and motor communication retries.
- `145737`, `016193`, `129544`, and `097158` share the same PI06 runtime shape (`chunk_size=4`, `n_action_steps=4`), so their speed should be nearly identical. The difference is mostly behavior quality, not raw throughput.
