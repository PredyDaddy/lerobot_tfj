#!/usr/bin/env python

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from common import ensure_dir, load_policy, resolve_policy_dir


class VisionModelForOnnx(torch.nn.Module):
    def __init__(self, vision_model: torch.nn.Module) -> None:
        super().__init__()
        self.vision_model = vision_model

    def forward(self, pixel_values: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        outputs = self.vision_model(pixel_values=pixel_values, output_hidden_states=False, return_dict=True)
        vit_embeds = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
        dummy = position_ids.to(dtype=vit_embeds.dtype).sum() * 0.0
        return vit_embeds + dummy


class LanguageModelForOnnx(torch.nn.Module):
    def __init__(
        self,
        language_model: torch.nn.Module,
        eagle_linear: torch.nn.Module,
        select_layer: int,
    ) -> None:
        super().__init__()
        self.language_model = language_model
        self.eagle_linear = eagle_linear
        self.select_layer = int(select_layer)

    def forward(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = attention_mask.shape
        cache_position = torch.arange(seq_len, device=inputs_embeds.device)
        position_ids = cache_position.unsqueeze(0).expand(batch_size, -1)
        neg_inf = torch.finfo(inputs_embeds.dtype).min
        causal_mask = torch.full(
            (seq_len, seq_len),
            fill_value=neg_inf,
            dtype=inputs_embeds.dtype,
            device=inputs_embeds.device,
        )
        causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, seq_len, seq_len)
        valid_tokens = attention_mask[:, None, None, :].to(dtype=inputs_embeds.dtype)
        causal_mask = causal_mask + (1.0 - valid_tokens) * neg_inf
        outputs = self.language_model.model(
            inputs_embeds=inputs_embeds,
            attention_mask=causal_mask,
            position_ids=position_ids,
            cache_position=cache_position,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
        embeddings = outputs.hidden_states[self.select_layer]
        return self.eagle_linear(embeddings)


def num_patches(backbone: torch.nn.Module) -> int:
    vision_model = backbone.eagle_model.vision_model
    if hasattr(vision_model, "vision_model"):
        return int(vision_model.vision_model.embeddings.num_patches)
    return int(vision_model.embeddings.num_patches)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export current-checkout GROOT backbone modules to ONNX.")
    parser.add_argument("--policy-path", required=True)
    parser.add_argument("--onnx-out-dir", required=True)
    parser.add_argument("--seq-len", type=int, default=296)
    parser.add_argument("--video-views", type=int, default=1)
    parser.add_argument("--vit-dtype", default="fp16", choices=["fp16"])
    parser.add_argument("--llm-dtype", default="fp16", choices=["fp16"])
    parser.add_argument("--opset", type=int, default=19)
    parser.add_argument("--device", default="cuda", choices=["cuda"])
    return parser


@torch.inference_mode()
def main() -> None:
    args = build_parser().parse_args()

    policy_dir = resolve_policy_dir(args.policy_path)
    out_dir = Path(args.onnx_out_dir).expanduser().resolve()
    eagle2_dir = ensure_dir(out_dir / "eagle2")

    _, _, policy = load_policy(policy_dir, device=args.device, strict=False)
    backbone = policy._groot_model.backbone
    backbone.eval()

    if hasattr(backbone.eagle_model.vision_model, "config"):
        backbone.eagle_model.vision_model.config._attn_implementation = "eager"
    if hasattr(backbone.eagle_model.language_model, "config"):
        backbone.eagle_model.language_model.config._attn_implementation = "eager"

    patch_count = num_patches(backbone)
    hidden_size = int(backbone.eagle_model.language_model.config.hidden_size)

    vit_wrapper = VisionModelForOnnx(backbone.eagle_model.vision_model).to(device=args.device, dtype=torch.float16)
    vit_wrapper.eval()
    pixel_values = torch.randn((args.video_views, 3, 224, 224), dtype=torch.float16, device=args.device)
    position_ids = torch.arange(patch_count, dtype=torch.int64, device=args.device).expand((args.video_views, -1))
    vit_path = eagle2_dir / f"vit_{args.vit_dtype}.onnx"
    torch.onnx.export(
        vit_wrapper,
        (pixel_values, position_ids),
        vit_path.as_posix(),
        export_params=True,
        do_constant_folding=True,
        opset_version=args.opset,
        input_names=["pixel_values", "position_ids"],
        output_names=["vit_embeds"],
        dynamic_axes={
            "pixel_values": {0: "batch_size"},
            "position_ids": {0: "batch_size"},
            "vit_embeds": {0: "batch_size"},
        },
    )

    llm_wrapper = LanguageModelForOnnx(
        backbone.eagle_model.language_model,
        backbone.eagle_linear,
        backbone.select_layer,
    ).to(device=args.device, dtype=torch.float16)
    llm_wrapper.eval()
    llm_inputs_embeds = torch.randn((1, args.seq_len, hidden_size), dtype=torch.float16, device=args.device)
    llm_attention_mask = torch.ones((1, args.seq_len), dtype=torch.int64, device=args.device)
    llm_path = eagle2_dir / f"llm_{args.llm_dtype}.onnx"
    torch.onnx.export(
        llm_wrapper,
        (llm_inputs_embeds, llm_attention_mask),
        llm_path.as_posix(),
        export_params=True,
        do_constant_folding=True,
        opset_version=args.opset,
        input_names=["inputs_embeds", "attention_mask"],
        output_names=["embeddings"],
        dynamic_axes={
            "inputs_embeds": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "embeddings": {0: "batch_size", 1: "sequence_length"},
        },
    )

    print(f"[OK] Exported backbone ONNX to: {eagle2_dir}")


if __name__ == "__main__":
    main()
