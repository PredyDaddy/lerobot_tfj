#!/usr/bin/env python

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from common import ensure_dir, load_policy, resolve_policy_dir


class VLLNVLSelfAttention(torch.nn.Module):
    def __init__(self, vlln: torch.nn.Module, vl_self_attention: torch.nn.Module) -> None:
        super().__init__()
        self.vlln = vlln
        self.vl_self_attention = vl_self_attention

    def forward(self, backbone_features: torch.Tensor) -> torch.Tensor:
        return self.vl_self_attention(self.vlln(backbone_features))


class DiTForOnnx(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, sa_embs: torch.Tensor, vl_embs: torch.Tensor, timesteps_tensor: torch.Tensor) -> torch.Tensor:
        return self.model(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embs,
            timestep=timesteps_tensor,
            return_all_hidden_states=False,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export current-checkout GROOT action-head modules to ONNX.")
    parser.add_argument("--policy-path", required=True)
    parser.add_argument("--onnx-out-dir", required=True)
    parser.add_argument("--seq-len", type=int, default=296)
    parser.add_argument("--state-horizon", type=int, default=1)
    parser.add_argument("--opset", type=int, default=19)
    parser.add_argument("--device", default="cuda", choices=["cuda"])
    return parser


@torch.inference_mode()
def main() -> None:
    args = build_parser().parse_args()

    policy_dir = resolve_policy_dir(args.policy_path)
    out_dir = Path(args.onnx_out_dir).expanduser().resolve()
    action_head_dir = ensure_dir(out_dir / "action_head")

    _, _, policy = load_policy(policy_dir, device=args.device, strict=False)
    action_head = policy._groot_model.action_head.to(dtype=torch.float16)
    action_head.eval()

    process_backbone_model = VLLNVLSelfAttention(action_head.vlln, action_head.vl_self_attention).to(
        device=args.device, dtype=torch.float16
    )
    process_backbone_model.eval()
    backbone_features = torch.randn(
        (1, args.seq_len, int(action_head.config.backbone_embedding_dim)),
        dtype=torch.float16,
        device=args.device,
    )
    torch.onnx.export(
        process_backbone_model,
        (backbone_features,),
        (action_head_dir / "vlln_vl_self_attention.onnx").as_posix(),
        export_params=True,
        do_constant_folding=True,
        opset_version=args.opset,
        input_names=["backbone_features"],
        output_names=["output"],
        dynamic_axes={
            "backbone_features": {0: "batch_size", 1: "sequence_length"},
            "output": {0: "batch_size", 1: "sequence_length"},
        },
    )

    state_encoder = action_head.state_encoder.to(device=args.device, dtype=torch.float16)
    state_encoder.eval()
    state_tensor = torch.randn(
        (1, args.state_horizon, int(action_head.config.max_state_dim)),
        dtype=torch.float16,
        device=args.device,
    )
    embodiment_id = torch.ones((1,), dtype=torch.int64, device=args.device)
    torch.onnx.export(
        state_encoder,
        (state_tensor, embodiment_id),
        (action_head_dir / "state_encoder.onnx").as_posix(),
        export_params=True,
        do_constant_folding=True,
        opset_version=args.opset,
        input_names=["state", "embodiment_id"],
        output_names=["output"],
        dynamic_axes={
            "state": {0: "batch_size"},
            "embodiment_id": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )

    action_encoder = action_head.action_encoder.to(device=args.device, dtype=torch.float16)
    action_encoder.eval()
    actions_tensor = torch.randn(
        (1, int(action_head.config.action_horizon), int(action_head.config.action_dim)),
        dtype=torch.float16,
        device=args.device,
    )
    timesteps_tensor = torch.ones((1,), dtype=torch.int64, device=args.device)
    torch.onnx.export(
        action_encoder,
        (actions_tensor, timesteps_tensor, embodiment_id),
        (action_head_dir / "action_encoder.onnx").as_posix(),
        export_params=True,
        do_constant_folding=True,
        opset_version=args.opset,
        input_names=["actions", "timesteps_tensor", "embodiment_id"],
        output_names=["output"],
        dynamic_axes={
            "actions": {0: "batch_size"},
            "timesteps_tensor": {0: "batch_size"},
            "embodiment_id": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )

    dit = DiTForOnnx(action_head.model).to(device=args.device, dtype=torch.float16)
    dit.eval()
    sa_seq_len = args.state_horizon + int(action_head.config.action_horizon) + int(
        action_head.config.num_target_vision_tokens
    )
    sa_embs = torch.randn(
        (1, sa_seq_len, int(action_head.config.input_embedding_dim)),
        dtype=torch.float16,
        device=args.device,
    )
    vl_embs = torch.randn(
        (1, args.seq_len, int(action_head.config.backbone_embedding_dim)),
        dtype=torch.float16,
        device=args.device,
    )
    torch.onnx.export(
        dit,
        (sa_embs, vl_embs, timesteps_tensor),
        (action_head_dir / "DiT_fp16.onnx").as_posix(),
        export_params=True,
        do_constant_folding=True,
        opset_version=args.opset,
        input_names=["sa_embs", "vl_embs", "timesteps_tensor"],
        output_names=["output"],
        dynamic_axes={
            "sa_embs": {0: "batch_size"},
            "vl_embs": {0: "batch_size", 1: "sequence_length"},
            "timesteps_tensor": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )

    action_decoder = action_head.action_decoder.to(device=args.device, dtype=torch.float16)
    action_decoder.eval()
    model_output = torch.randn(
        (1, sa_seq_len, int(action_head.config.hidden_size)),
        dtype=torch.float16,
        device=args.device,
    )
    torch.onnx.export(
        action_decoder,
        (model_output, embodiment_id),
        (action_head_dir / "action_decoder.onnx").as_posix(),
        export_params=True,
        do_constant_folding=True,
        opset_version=args.opset,
        input_names=["model_output", "embodiment_id"],
        output_names=["output"],
        dynamic_axes={
            "model_output": {0: "batch_size"},
            "embodiment_id": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )

    print(f"[OK] Exported action-head ONNX to: {action_head_dir}")


if __name__ == "__main__":
    main()
