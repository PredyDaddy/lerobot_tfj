#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
from pathlib import Path

import tensorrt as trt

from common import write_json
from groot_compare_common_local import (
    build_runner_dict,
    build_synthetic_inputs,
    compare_outputs,
    load_compare_context,
    run_compare_pipeline,
    tensor_summary,
)
from trt_runtime_local import TrtSession


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local Torch-vs-TensorRT compare for the fixed 7-engine GROOT export.")
    parser.add_argument("--policy-path", required=True)
    parser.add_argument("--engine-dir", required=True)
    parser.add_argument("--seq-len", type=int, required=True)
    parser.add_argument("--video-views", type=int, required=True)
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

    engine_dir = Path(args.engine_dir).expanduser().resolve()
    sessions = {
        "vit": TrtSession(engine_dir / "vit_fp16.engine", device=args.device),
        "llm": TrtSession(engine_dir / "llm_fp16.engine", device=args.device),
        "vlln": TrtSession(engine_dir / "vlln_vl_self_attention.engine", device=args.device),
        "state_encoder": TrtSession(engine_dir / "state_encoder.engine", device=args.device),
        "action_encoder": TrtSession(engine_dir / "action_encoder.engine", device=args.device),
        "dit": TrtSession(engine_dir / "DiT_fp16.engine", device=args.device),
        "action_decoder": TrtSession(engine_dir / "action_decoder.engine", device=args.device),
    }

    torch_outputs = run_compare_pipeline(ctx, build_runner_dict(ctx), inputs)
    trt_outputs = run_compare_pipeline(
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
        "engine_dir": engine_dir.as_posix(),
        "tensorrt_version": trt.__version__,
        "seed": args.seed,
        "seq_len": args.seq_len,
        "video_views": args.video_views,
        "inputs": {key: tensor_summary(value) for key, value in inputs.items()},
        "results": compare_outputs(torch_outputs, trt_outputs),
    }
    write_json(Path(args.json_out).expanduser().resolve(), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
