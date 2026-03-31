#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from common import write_json
from groot_compare_common_local import (
    build_runner_dict,
    build_synthetic_inputs,
    compare_outputs,
    load_compare_context,
    run_compare_pipeline,
    tensor_summary,
)


class OnnxSession:
    def __init__(self, model_path: str | Path, *, device: str, providers: list[str]) -> None:
        self.model_path = Path(model_path).expanduser().resolve()
        if not self.model_path.is_file():
            raise FileNotFoundError(f"ONNX model not found: {self.model_path}")
        self.device = torch.device(device)
        self.session = ort.InferenceSession(self.model_path.as_posix(), providers=providers)
        self.input_names = [item.name for item in self.session.get_inputs()]
        self.output_names = [item.name for item in self.session.get_outputs()]
        self.providers = list(self.session.get_providers())

    def run(self, feed_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        ort_inputs = {
            name: np.ascontiguousarray(feed_dict[name].detach().to("cpu").numpy())
            for name in self.input_names
        }
        outputs = self.session.run(self.output_names, ort_inputs)
        return {
            name: torch.from_numpy(output).to(self.device)
            for name, output in zip(self.output_names, outputs, strict=True)
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local Torch-vs-ONNX compare for the fixed 7-subgraph GROOT export.")
    parser.add_argument("--policy-path", required=True)
    parser.add_argument("--onnx-dir", required=True)
    parser.add_argument("--seq-len", type=int, required=True)
    parser.add_argument("--video-views", type=int, required=True)
    parser.add_argument("--vit-dtype", default="fp16", choices=["fp16"])
    parser.add_argument("--llm-dtype", default="fp16", choices=["fp16"])
    parser.add_argument("--dit-dtype", default="fp16")
    parser.add_argument("--seed", type=int, default=20260303)
    parser.add_argument("--device", default="cuda", choices=["cuda"])
    parser.add_argument("--json-out", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    ctx = load_compare_context(args.policy_path, device=args.device)
    inputs = build_synthetic_inputs(
        ctx,
        seq_len=args.seq_len,
        video_views=args.video_views,
        seed=args.seed,
    )

    available = ort.get_available_providers()
    preferred = [name for name in ["CUDAExecutionProvider", "CPUExecutionProvider"] if name in available]
    if not preferred:
        preferred = available
    if not preferred:
        raise RuntimeError("No ONNX Runtime providers are available.")

    onnx_dir = Path(args.onnx_dir).expanduser().resolve()
    sessions = {
        "vit": OnnxSession(onnx_dir / "eagle2" / f"vit_{args.vit_dtype}.onnx", device=args.device, providers=preferred),
        "llm": OnnxSession(onnx_dir / "eagle2" / f"llm_{args.llm_dtype}.onnx", device=args.device, providers=preferred),
        "vlln": OnnxSession(onnx_dir / "action_head" / "vlln_vl_self_attention.onnx", device=args.device, providers=preferred),
        "state_encoder": OnnxSession(onnx_dir / "action_head" / "state_encoder.onnx", device=args.device, providers=preferred),
        "action_encoder": OnnxSession(onnx_dir / "action_head" / "action_encoder.onnx", device=args.device, providers=preferred),
        "dit": OnnxSession(onnx_dir / "action_head" / f"DiT_{args.dit_dtype}.onnx", device=args.device, providers=preferred),
        "action_decoder": OnnxSession(onnx_dir / "action_head" / "action_decoder.onnx", device=args.device, providers=preferred),
    }

    torch_outputs = run_compare_pipeline(ctx, build_runner_dict(ctx), inputs)
    onnx_outputs = run_compare_pipeline(
        ctx,
        {
            "vit": lambda pixel_values, position_ids: sessions["vit"].run(
                {"pixel_values": pixel_values, "position_ids": position_ids}
            )["vit_embeds"],
            "llm": lambda inputs_embeds, attention_mask: sessions["llm"].run(
                {"inputs_embeds": inputs_embeds, "attention_mask": attention_mask}
            )["embeddings"],
            "vlln": lambda backbone_features: sessions["vlln"].run({"backbone_features": backbone_features})["output"],
            "state_encoder": lambda state, embodiment_id: sessions["state_encoder"].run(
                {"state": state, "embodiment_id": embodiment_id}
            )["output"],
            "action_encoder": lambda actions, timesteps_tensor, embodiment_id: sessions["action_encoder"].run(
                {
                    "actions": actions,
                    "timesteps_tensor": timesteps_tensor,
                    "embodiment_id": embodiment_id,
                }
            )["output"],
            "dit": lambda sa_embs, vl_embs, timesteps_tensor: sessions["dit"].run(
                {
                    "sa_embs": sa_embs,
                    "vl_embs": vl_embs,
                    "timesteps_tensor": timesteps_tensor,
                }
            )["output"],
            "action_decoder": lambda model_output, embodiment_id: sessions["action_decoder"].run(
                {"model_output": model_output, "embodiment_id": embodiment_id}
            )["output"],
        },
        inputs,
    )

    report = {
        "policy_path": ctx.policy_dir.as_posix(),
        "onnx_dir": onnx_dir.as_posix(),
        "providers": sessions["vit"].providers,
        "seed": args.seed,
        "seq_len": args.seq_len,
        "video_views": args.video_views,
        "inputs": {key: tensor_summary(value) for key, value in inputs.items()},
        "missing": [],
        "results": compare_outputs(torch_outputs, onnx_outputs),
    }
    write_json(Path(args.json_out).expanduser().resolve(), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
