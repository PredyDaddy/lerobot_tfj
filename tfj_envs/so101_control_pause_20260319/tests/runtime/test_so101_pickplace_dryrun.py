import json
import sys
from pathlib import Path

import pytest

REPO_SRC = Path(__file__).resolve().parents[2] / "src"
REPO_SRC_STR = str(REPO_SRC)
if REPO_SRC_STR not in sys.path:
    sys.path.insert(0, REPO_SRC_STR)

from lerobot.scripts.lerobot_run_so101_pickplace import SO101PickPlaceConfig, run


def test_pickplace_requires_guard_in_real_robot_mode(tmp_path):
    cfg = SO101PickPlaceConfig(
        dry_run=False,
        safety_profile="off",
        leader_port="/dev/null",
        dataset_root=tmp_path / "real_robot",
        events_jsonl_path=tmp_path / "real_robot" / "events.jsonl",
    )

    with pytest.raises(ValueError, match="safety_profile"):
        run(cfg)


def test_pickplace_dry_run_allows_guard_off_and_writes_events(tmp_path):
    events_path = tmp_path / "dryrun" / "events.jsonl"
    cfg = SO101PickPlaceConfig(
        dry_run=True,
        safety_profile="off",
        dataset_root=tmp_path / "dryrun",
        events_jsonl_path=events_path,
        num_episodes=1,
        episode_time_s=0.05,
        dataset_fps=20,
        intent_json='{"task": "place red block"}',
    )

    result = run(cfg)

    assert result["dry_run"] is True
    assert result["halted"] is False
    assert events_path.exists()

    records = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
    events = [record["event"] for record in records]

    assert "run_start" in events
    assert "step" in events
    assert "run_end" in events
    assert all(record["safety_profile"] == "off" for record in records)
    assert any(record.get("task") == "place red block" for record in records if "task" in record)


def test_pickplace_dry_run_strict_guard_halt_logs_guard_reject_and_stops(tmp_path):
    events_path = tmp_path / "dryrun_strict" / "events.jsonl"
    cfg = SO101PickPlaceConfig(
        dry_run=True,
        safety_profile="strict",
        dataset_root=tmp_path / "dryrun_strict",
        events_jsonl_path=events_path,
        num_episodes=1,
        episode_time_s=0.05,
        dataset_fps=20,
        intent_json='{"task": "place red block", "dry_run_action": {"joint_1.pos": 200.0, "joint_2.pos": 0.0, "joint_3.pos": 0.0}}',
    )

    result = run(cfg)

    assert result["dry_run"] is True
    assert result["halted"] is True
    assert events_path.exists()

    records = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
    events = [record["event"] for record in records]

    assert "run_start" in events
    assert "guard_reject" in events
    assert "step" in events
    assert "run_end" in events
    assert any(record.get("halt") is True for record in records if record["event"] == "guard_reject")
    assert any(record.get("halted") is True for record in records if record["event"] == "run_end")
