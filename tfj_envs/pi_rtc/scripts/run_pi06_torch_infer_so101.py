#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pi06.modeling_pi06 import PI06Policy

import run_pi05_torch_infer_so101 as base


DEFAULT_POLICY_PATH = Path(
    "/data/tfj/Evo-RL/outputs/train/pi06_stage2_restart_9epoch_bs2_20260327_103000/stage2/checkpoints/145737/pretrained_model"
)
_ORIG_BUILD_PARSER = base.build_parser


def _expected_pi06(policy_cfg: object) -> None:
    if getattr(policy_cfg, "type", None) != "pi06":
        raise ValueError(f"Expected PI06 policy, got {getattr(policy_cfg, 'type', None)!r}")


def resolve_rtc_runtime_config(args, policy_cfg):
    _expected_pi06(policy_cfg)
    checkpoint_cfg = getattr(policy_cfg, "rtc_config", None)
    if checkpoint_cfg is not None and not isinstance(checkpoint_cfg, base.RTCConfig):
        raise TypeError(f"Expected rtc_config to be RTCConfig or None, got {type(checkpoint_cfg)}")

    base_cfg = checkpoint_cfg or base.RTCConfig()
    override_applied = base._rtc_override_requested(args)
    enabled = bool(getattr(args, "rtc_enable", False) or override_applied)
    schedule = (
        base.parse_rtc_attention_schedule(getattr(args, "rtc_prefix_attention_schedule", None))
        or base_cfg.prefix_attention_schedule
    )
    max_guidance_weight = (
        float(args.rtc_max_guidance_weight)
        if getattr(args, "rtc_max_guidance_weight", None) is not None
        else float(base_cfg.max_guidance_weight)
    )
    execution_horizon = (
        int(args.rtc_execution_horizon)
        if getattr(args, "rtc_execution_horizon", None) is not None
        else int(base_cfg.execution_horizon)
    )
    debug = bool(getattr(args, "rtc_debug", False) or (enabled and bool(base_cfg.debug) and not override_applied))
    debug_maxlen = (
        int(args.rtc_debug_maxlen)
        if getattr(args, "rtc_debug_maxlen", None) is not None
        else int(base_cfg.debug_maxlen)
    )

    if execution_horizon <= 0:
        raise ValueError(f"--rtc-execution-horizon must be positive, got {execution_horizon}")
    if max_guidance_weight <= 0.0:
        raise ValueError(f"--rtc-max-guidance-weight must be positive, got {max_guidance_weight}")
    if debug_maxlen <= 0:
        raise ValueError(f"--rtc-debug-maxlen must be positive, got {debug_maxlen}")

    resolved_cfg = base.RTCConfig(
        enabled=enabled,
        prefix_attention_schedule=schedule,
        max_guidance_weight=max_guidance_weight,
        execution_horizon=execution_horizon,
        debug=debug,
        debug_maxlen=debug_maxlen,
    )
    policy_cfg.rtc_config = resolved_cfg
    return base.ResolvedRTCRuntimeConfig(
        checkpoint_enabled=(checkpoint_cfg.enabled if checkpoint_cfg is not None else None),
        config=resolved_cfg,
        enabled_by_cli=bool(getattr(args, "rtc_enable", False)),
        override_applied=override_applied,
    )


def build_parser():
    parser = _ORIG_BUILD_PARSER()
    parser.description = "Run PI0.6 PyTorch/model.safetensors inference on a real SO101 follower robot."
    parser.set_defaults(
        policy_path=str(DEFAULT_POLICY_PATH),
        trt_path=None,
        trt_metadata_path=None,
        task="Put the block in the bin",
    )
    return parser


def apply_pi_runtime_overrides(args, policy_cfg):
    _expected_pi06(policy_cfg)

    if hasattr(policy_cfg, "gradient_checkpointing"):
        policy_cfg.gradient_checkpointing = False

    chunk_size = int(policy_cfg.chunk_size)
    if args.policy_n_action_steps is not None:
        if not 1 <= args.policy_n_action_steps <= chunk_size:
            raise ValueError(
                f"--policy-n-action-steps must be within [1, {chunk_size}], got {args.policy_n_action_steps}"
            )
        policy_cfg.n_action_steps = int(args.policy_n_action_steps)

    if args.policy_num_inference_steps is not None:
        if args.policy_num_inference_steps <= 0:
            raise ValueError("--policy-num-inference-steps must be positive")
        policy_cfg.num_inference_steps = int(args.policy_num_inference_steps)

    if args.policy_temporal_ensemble_coeff is not None:
        raise ValueError("PI06 PyTorch runtime does not support temporal ensembling.")

    policy_cfg.use_amp = bool(args.policy_use_amp)
    return resolve_rtc_runtime_config(args, policy_cfg)


def load_policy_config(policy_dir: Path, policy_device: str) -> PreTrainedConfig:
    policy_cfg = PreTrainedConfig.from_pretrained(str(policy_dir))
    policy_cfg.pretrained_path = str(policy_dir)
    policy_cfg.device = policy_device
    _expected_pi06(policy_cfg)
    return policy_cfg


def preflight_policy(policy_dir: Path, policy_cfg: object):
    policy = PI06Policy.from_pretrained(
        str(policy_dir),
        config=policy_cfg,
        local_files_only=True,
        strict=False,
    )
    policy.eval()
    base.info(
        "PI06 PyTorch policy OK: "
        f"device={policy_cfg.device}, use_amp={policy_cfg.use_amp}, "
        f"rtc_enabled={getattr(getattr(policy_cfg, 'rtc_config', None), 'enabled', False)}"
    )
    base.info(
        "PI06 runtime: "
        f"chunk_size={policy_cfg.chunk_size}, "
        f"n_action_steps={policy_cfg.n_action_steps}, "
        f"num_inference_steps={policy_cfg.num_inference_steps}"
    )
    return policy


def print_summary(
    args,
    policy_dir: Path,
    calib_dir: Path,
    tokenizer_dir: Path | None,
    policy_cfg: object,
    preprocessor_details: dict,
    rtc_runtime,
) -> None:
    base.info(f"Python: {base.sys.executable}")
    base.info(f"Policy path: {policy_dir}")
    base.info(f"Policy type: {getattr(policy_cfg, 'type', '<unknown>')}")
    base.info(
        "PI06 runtime config: "
        f"chunk_size={policy_cfg.chunk_size}, "
        f"n_action_steps={policy_cfg.n_action_steps}, "
        f"num_inference_steps={policy_cfg.num_inference_steps}, "
        f"use_amp={policy_cfg.use_amp}"
    )
    base.info(f"Resolved RTC config: {rtc_runtime.as_dict()}")
    if rtc_runtime.checkpoint_enabled and not rtc_runtime.config.enabled:
        base.warn(
            "Checkpoint RTC config is enabled, but launcher runtime keeps RTC off by default "
            "unless --rtc-enable/--rtc-enabled or another --rtc-* override is provided."
        )
    base.info(f"Policy device: {args.policy_device}")
    base.info(f"Calibration dir: {calib_dir}")
    base.info(f"Robot port: {args.robot_port}")
    base.info(f"Robot max_relative_target: {args.robot_max_relative_target}")
    base.info(f"Cameras: top={args.top_cam_index}, wrist={args.wrist_cam_index}")
    base.info(
        "FPS settings: "
        f"camera_fps={args.camera_fps}, "
        f"control_fps={args.control_fps if args.control_fps is not None else args.camera_fps}"
    )
    base.info(f"Task: {args.task}")
    base.info(f"run_time_s: {args.run_time_s} (<=0 means until Ctrl+C)")
    base.info(f"sync_refill_timeout_s: {args.sync_refill_timeout_s}")
    base.info(
        "prefetch_threshold: "
        f"{args.prefetch_threshold if args.prefetch_threshold is not None else '<latency-aware>'}"
    )
    base.info(
        "Tokenizer path: "
        f"{preprocessor_details.get('effective_tokenizer_name') or tokenizer_dir or '<unresolved>'}"
    )
    base.info(
        "TRT compatibility args: "
        f"path={Path(args.trt_path).expanduser().resolve(strict=False) if args.trt_path else '<unset>'}, "
        f"metadata={Path(args.trt_metadata_path).expanduser().resolve(strict=False) if args.trt_metadata_path else '<unset>'}, "
        f"trt_device={args.trt_device}"
    )
    base.info("Torch runtime note: --trt-* flags are accepted for preflight/compatibility only; live inference uses pi06 weights.")
    base.info(f"Script joint_delta_limit: {args.joint_delta_limit}")
    base.info(
        "Script gripper_delta_limit: "
        f"{args.gripper_delta_limit if args.gripper_delta_limit is not None else args.joint_delta_limit}"
    )
    base.info(f"Script joint_action_alpha: {args.joint_action_alpha}")
    base.info(
        "Script gripper_action_alpha: "
        f"{args.gripper_action_alpha if args.gripper_action_alpha is not None else args.joint_action_alpha}"
    )


base.DEFAULT_POLICY_PATH = DEFAULT_POLICY_PATH
base.build_parser = build_parser
base.resolve_rtc_runtime_config = resolve_rtc_runtime_config
base.apply_pi_runtime_overrides = apply_pi_runtime_overrides
base.load_policy_config = load_policy_config
base.preflight_policy = preflight_policy
base.print_summary = print_summary


def main() -> int:
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
