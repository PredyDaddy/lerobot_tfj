from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SERVER_PATH = REPO_ROOT / "scripts" / "openclaw_groot_server.py"
ROUTER_SCRIPT = REPO_ROOT / "scripts" / "run_so101_pickplace_infer.sh"


def load_server_module():
    spec = importlib.util.spec_from_file_location("openclaw_groot_server_test", SERVER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def server_module():
    module = load_server_module()
    with module.jobs_lock:
        module.jobs.clear()
        module.robot_sessions.clear()
        module.reserved_job_ids.clear()
    yield module
    with module.jobs_lock:
        module.jobs.clear()
        module.robot_sessions.clear()
        module.reserved_job_ids.clear()


class FakeProcess:
    def __init__(self, pid: int = 4321, return_code: int | None = None):
        self.pid = pid
        self._return_code = return_code

    def poll(self) -> int | None:
        return self._return_code


def run_script(script: Path, *, env: dict[str, str], args: tuple[str, ...] = ()) -> subprocess.CompletedProcess[str]:
    merged_env = os.environ.copy()
    merged_env.update(env)
    return subprocess.run(
        ["bash", str(script), *args],
        cwd=REPO_ROOT,
        env=merged_env,
        capture_output=True,
        text=True,
        check=True,
    )


def test_server_normalizes_legacy_payload_aliases(server_module, tmp_path):
    job_dir = tmp_path / "job-a"
    raw_payload = {
        "backend": "ACT_DISTILL",
        "task_text": "Place the red block into the left bin",
        "intent": {"verb": "pick_place", "target_object": "red block"},
        "events_path": str(job_dir / "custom-events.jsonl"),
        "clear_dataset_root": True,
        "robot_id": "robot-a",
    }

    payload, compat_aliases = server_module._normalize_payload(raw_payload, job_id="job-a", job_dir=job_dir)

    assert payload["backend"] == "act"
    assert payload["task"] == "Place the red block into the left bin"
    assert json.loads(payload["intent_json"]) == {"target_object": "red block", "verb": "pick_place"}
    assert payload["safety_profile"] == "default"
    assert payload["events_jsonl_path"] == str(job_dir / "custom-events.jsonl")
    assert payload["clear_dataset_root"] is True
    assert compat_aliases == {
        "task": "task_text",
        "intent_json": "intent",
        "events_jsonl_path": "events_path",
    }

    env = server_module._build_env(payload, job_id="job-a", job_dir=job_dir)
    assert env["BACKEND"] == "act"
    assert env["TASK_TEXT"] == payload["task"]
    assert env["INTENT_JSON"] == payload["intent_json"]
    assert env["TASK_INTENT_JSON"] == payload["intent_json"]
    assert env["SAFETY_PROFILE"] == "default"
    assert env["EVENTS_JSONL_PATH"] == payload["events_jsonl_path"]
    assert env["EVENTS_PATH"] == payload["events_jsonl_path"]
    assert env["CLEAR_DATASET_ROOT"] == "1"
    assert env["ROBOT_ID"] == "robot-a"


def test_server_reservation_blocks_duplicate_job_id_and_robot_reuse(server_module, tmp_path):
    log_path = tmp_path / "run.log"
    log_path.write_text("", encoding="utf-8")
    running_job = server_module.Job(
        job_id="sameid",
        process=FakeProcess(return_code=None),
        log_path=log_path,
        payload={"backend": "groot", "task": "pick block"},
        robot_id="robot-a",
    )

    with server_module.jobs_lock:
        assert server_module._reserve_job_start_locked("pending-job", "robot-z") is None
        assert server_module._reserve_job_start_locked("pending-job", "robot-y") == ("job_id", "pending-job")
        assert server_module._reserve_job_start_locked("other-job", "robot-z") == ("robot_id", "pending-job")
        server_module._release_job_reservation_locked("robot-z", "pending-job")

        server_module.jobs[running_job.job_id] = running_job
        server_module.robot_sessions[running_job.robot_id] = running_job.job_id

        assert server_module._reserve_job_start_locked("sameid", "robot-b") == ("job_id", "sameid")
        assert server_module._reserve_job_start_locked("job-2", "robot-a") == ("robot_id", "sameid")
        assert server_module._acquire_robot_session_locked("robot-a", "sameid") == "sameid"


def test_server_robot_lock_released_after_job_finishes(server_module, tmp_path):
    log_path = tmp_path / "run.log"
    log_path.write_text("", encoding="utf-8")
    process = FakeProcess(return_code=None)
    job = server_module.Job(
        job_id="job-1",
        process=process,
        log_path=log_path,
        payload={"backend": "groot", "task": "pick block"},
        robot_id="robot-a",
    )

    with server_module.jobs_lock:
        server_module.jobs[job.job_id] = job
        server_module.robot_sessions[job.robot_id] = job.job_id
        assert server_module._acquire_robot_session_locked("robot-a", "job-2") == "job-1"

        process._return_code = 0
        server_module._reap_finished_jobs_locked()

        assert "robot-a" not in server_module.robot_sessions
        assert server_module._acquire_robot_session_locked("robot-a", "job-2") is None


def test_router_canonicalizes_contract_flags_for_policy_record():
    intent_json = '{"task":"place red block","action_delta":{"joint_1.pos":0.25}}'
    result = run_script(
        ROUTER_SCRIPT,
        env={
            "BACKEND": "policy_record",
            "PYTHON_BIN": "echo",
        },
        args=(
            "--dry_run=true",
            "--clear_dataset_root=0",
            "--safety_profile=off",
            "--events_jsonl_path=/tmp/policy-events.jsonl",
            f"--intent_json={intent_json}",
        ),
    )

    output = result.stdout
    assert "backend=policy_record" in output
    assert "lerobot_run_so101_pickplace.py" in output
    assert "--clear_dataset_root=false" in output
    assert '--safety_profile="off"' in output
    assert "--events_jsonl_path=/tmp/policy-events.jsonl" in output
    assert '--intent_json="' in output
    assert "action_delta" in output
    assert "place red block" in output


def test_router_bridges_generic_envs_for_act_to_guarded_entrypoint():
    result = run_script(
        ROUTER_SCRIPT,
        env={
            "BACKEND": "act",
            "PYTHON_BIN": "echo",
            "TASK_TEXT": "Stack the cube",
            "TOP_CAMERA_INDEX": "11",
            "WRIST_CAMERA_INDEX": "22",
            "CAMERA_FPS": "15",
            "EPISODE_TIME_S": "9",
            "DISPLAY_DATA": "false",
        },
    )

    output = result.stdout
    assert "backend=act" in output
    assert "lerobot_run_so101_pickplace.py" in output
    assert "--top_camera_index=11" in output
    assert "--wrist_camera_index=22" in output
    assert "--dataset_fps=15" in output
    assert "--episode_time_s=9" in output
    assert "--task=Stack the cube" in output
    assert '--safety_profile="default"' in output
