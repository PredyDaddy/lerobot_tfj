# ACT TRT Workspace

## Structure

- `scripts/`
  - export, compare, and real-robot runtime scripts
- `docs/`
  - comparison results, debugging notes, and usage guides

## Main Entry Points

- export checkpoint to ONNX + TensorRT:
  - [scripts/export_act_checkpoint_engine.py](/data/tfj/lerobot_tfj/tfj_envs/act_trt/scripts/export_act_checkpoint_engine.py)
- compare `016000 safetensors` vs my exported `016000 TRT`:
  - [scripts/compare_act_016000_my_exported_trt_vs_safetensors.py](/data/tfj/lerobot_tfj/tfj_envs/act_trt/scripts/compare_act_016000_my_exported_trt_vs_safetensors.py)
- pure TRT real-robot inference:
  - [scripts/run_act_trt_infer_so101.py](/data/tfj/lerobot_tfj/tfj_envs/act_trt/scripts/run_act_trt_infer_so101.py)
- TRT real-robot eval and recording:
  - [scripts/run_act_trt_so101_eval.py](/data/tfj/lerobot_tfj/tfj_envs/act_trt/scripts/run_act_trt_so101_eval.py)

## Main Guide

Read:

- [docs/ACT_TRT_PYTHON_SCRIPT_USAGE.md](/data/tfj/lerobot_tfj/tfj_envs/act_trt/docs/ACT_TRT_PYTHON_SCRIPT_USAGE.md)
- [docs/ACT_TRT_EXPORT_AND_DEPLOY_GUIDE.md](/data/tfj/lerobot_tfj/tfj_envs/act_trt/docs/ACT_TRT_EXPORT_AND_DEPLOY_GUIDE.md)
- [docs/ACT_TRT_EXPORT_VERIFY_QUICKSTART.md](/data/tfj/lerobot_tfj/tfj_envs/act_trt/docs/ACT_TRT_EXPORT_VERIFY_QUICKSTART.md)




