#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch

from act_trt_paths import REPO_ROOT, SRC_DIR

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from act_model_utils import load_act_policy, load_model_spec, make_case_inputs, torch_core_forward


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export ACT core model to ONNX.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", "--output-dir", dest="output_dir", type=Path, required=True)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--verify", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--simplify", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dynamo", action="store_true")
    return parser.parse_args()


def maybe_simplify_onnx(onnx_path: Path) -> None:
    try:
        import onnx
        from onnxsim import simplify
    except Exception:
        return

    model = onnx.load(str(onnx_path))
    simplified, ok = simplify(model)
    if ok:
        onnx.save(simplified, str(onnx_path))


def require_export_dependencies() -> None:
    missing = []
    if importlib.util.find_spec("onnx") is None:
        missing.append("onnx")

    if missing:
        missing_text = ", ".join(missing)
        raise ModuleNotFoundError(
            "ACT ONNX 导出缺少依赖: "
            f"{missing_text}\n"
            "当前脚本会调用 torch.onnx.export，因此必须先安装 onnx。\n"
            "建议直接在 lerobot_flex 环境运行导出命令，例如：\n"
            "conda run --live-stream -n lerobot_flex python "
            "/data/tfj/lerobot_tfj/tfj_envs/act_trt/scripts/export_act_checkpoint_engine.py "
            "--checkpoint /data/tfj/lerobot_tfj/outputs/act_grasp_block_in_bin1/checkpoints/016000/pretrained_model\n"
            "如果你坚持在当前环境导出，也可以先安装 onnx。"
        )


def main() -> int:
    args = parse_args()
    require_export_dependencies()
    checkpoint = args.checkpoint.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    spec = load_model_spec(checkpoint)
    policy = load_act_policy(checkpoint, device=args.device)
    obs_state, img0, img1 = make_case_inputs(spec=spec, seed=0, case="random")
    obs_state = obs_state.to(args.device)
    img0 = img0.to(args.device)
    img1 = img1.to(args.device)

    onnx_path = output_dir / "act_single.onnx"
    metadata_path = output_dir / "export_metadata.json"

    class ExportWrapper(torch.nn.Module):
        def __init__(self, act_policy):
            super().__init__()
            self.policy = act_policy

        def forward(self, obs_state_norm, img0_norm, img1_norm):
            return torch_core_forward(self.policy, obs_state_norm, img0_norm, img1_norm)

    wrapper = ExportWrapper(policy).eval()

    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            (obs_state, img0, img1),
            str(onnx_path),
            export_params=True,
            opset_version=args.opset,
            do_constant_folding=True,
            input_names=["obs_state_norm", "img0_norm", "img1_norm"],
            output_names=["actions_norm"],
            dynamic_axes=None,
            dynamo=args.dynamo,
        )

    if args.simplify:
        maybe_simplify_onnx(onnx_path)

    metadata = {
        "checkpoint": str(checkpoint),
        "onnx_path": str(onnx_path),
        "camera_order_visual_keys": spec.visual_keys,
        "io_mapping": {
            "obs_state_norm": "observation.state (mean/std normalized)",
            "img0_norm": f"{spec.visual_keys[0]} (mean/std normalized)",
            "img1_norm": f"{spec.visual_keys[1]} (mean/std normalized)",
            "actions_norm": "action chunk (mean/std normalized)",
        },
        "shapes": {
            "obs_state_norm": list(spec.obs_state_shape),
            "img0_norm": list(spec.image_shape),
            "img1_norm": list(spec.image_shape),
            "actions_norm": list(spec.action_shape),
        },
        "export": {
            "opset": args.opset,
            "dynamic_batch": False,
            "dynamo": bool(args.dynamo),
            "device": args.device,
        },
        "verification": {
            "enabled": bool(args.verify),
            "skipped": not bool(args.verify),
        },
        "notes": [
            "This ONNX exports ACT core network only (policy.model). Action queue logic stays in Python.",
            "Inputs/outputs are normalized (mean/std). Use policy processors for runtime normalization.",
            "Camera order must match config.json VISUAL input_features order.",
        ],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(onnx_path)
    print(metadata_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
