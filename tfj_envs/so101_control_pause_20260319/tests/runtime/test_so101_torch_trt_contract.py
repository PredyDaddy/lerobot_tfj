from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = REPO_ROOT / "tfj_envs" / "groot_trt" / "scripts" / "run_groot_trt_infer_so101.py"
RUNNER_DIR = RUNNER_PATH.parent


def _load_runner_module():
    module_name = "run_groot_trt_infer_so101_test"
    for path in [str(REPO_ROOT), str(REPO_ROOT / "src"), str(RUNNER_DIR)]:
        if path not in sys.path:
            sys.path.insert(0, path)
    for module_key in list(sys.modules):
        if module_key == "lerobot" or module_key.startswith("lerobot."):
            sys.modules.pop(module_key, None)
    if str(RUNNER_DIR) not in sys.path:
        sys.path.insert(0, str(RUNNER_DIR))
    spec = importlib.util.spec_from_file_location(module_name, RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _make_args(**overrides):
    defaults = {
        "task": "pick the blue block",
        "intent_json": "",
        "safety_profile": "default",
        "events_jsonl": "",
        "dry_run": False,
        "preflight_only": False,
        "allow_unsafe": False,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


runner = _load_runner_module()


def test_resolve_runtime_contract_prefers_intent_json_over_task():
    args = _make_args(
        task="stale free-form task",
        intent_json=json.dumps(
            {
                "task": "pick_place red_block left_bin",
                "verb": "pick_place",
                "target_object": "red_block",
                "target_container": "left_bin",
            }
        ),
    )

    contract = runner.resolve_runtime_contract(args)

    assert contract.task_text == "pick_place red_block left_bin"
    assert contract.resolved_intent_source == "intent_json"
    assert contract.safety_profile == "default"
    assert contract.max_relative_target == pytest.approx(8.0)


def test_resolve_runtime_contract_rejects_unsafe_profile_for_real_robot_run():
    args = _make_args(safety_profile="off")

    with pytest.raises(ValueError, match="Unsafe safety profiles are rejected"):
        runner.resolve_runtime_contract(args)


def test_resolve_runtime_contract_allows_unsafe_profile_only_for_explicit_dry_run():
    args = _make_args(safety_profile="off", dry_run=True, allow_unsafe=True)

    contract = runner.resolve_runtime_contract(args)

    assert contract.mode == "dry_run_contract_only"
    assert contract.safety_profile == "off"
    assert contract.max_relative_target is None


def test_joint_safety_guard_clamps_large_joint_deltas():
    guard = runner.JointSafetyGuard(max_relative_target=4.0)

    result = guard.validate(
        action={
            "shoulder.pos": 25.0,
            "elbow.pos": -6.0,
        },
        obs={
            "shoulder.pos": 10.0,
            "elbow.pos": -5.0,
        },
    )

    assert result.status == "CLAMP_AND_ACCEPT"
    assert result.action["shoulder.pos"] == pytest.approx(14.0)
    assert result.action["elbow.pos"] == pytest.approx(-6.0)
    assert result.details["clamped_joints"]["shoulder.pos"]["requested"] == pytest.approx(25.0)


def test_jsonl_event_logger_writes_contract_schema(tmp_path: Path):
    events_path = tmp_path / "events.jsonl"
    logger = runner.JsonlEventLogger(events_path, run_id="run123")

    logger.log("run_start", mode="robot_run", safety_profile="default", task_text="pick block")

    records = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 1
    assert records[0]["schema_version"] == runner.CONTRACT_SCHEMA_VERSION
    assert records[0]["run_id"] == "run123"
    assert records[0]["event_type"] == "run_start"
    assert records[0]["safety_profile"] == "default"
