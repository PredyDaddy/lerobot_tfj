#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Dataset tools utilities for LeRobotDataset.

This module provides utilities for:
- Deleting episodes from datasets
- Splitting datasets into multiple smaller datasets
- Adding/removing features from datasets
- Merging datasets (wrapper around aggregate functionality)
"""

import logging
import shutil
from collections.abc import Callable
from pathlib import Path

import datasets
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from tqdm import tqdm

from lerobot.datasets.aggregate import aggregate_datasets
from lerobot.datasets.compute_stats import aggregate_stats, compute_episode_stats
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import (
    DATA_DIR,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_FEATURES,
    DEFAULT_DATA_FILE_SIZE_IN_MB,
    DEFAULT_DATA_PATH,
    DEFAULT_EPISODES_PATH,
    get_parquet_file_size_in_mb,
    load_episodes,
    update_chunk_file_indices,
    write_info,
    write_stats,
    write_tasks,
)
from lerobot.utils.constants import HF_LEROBOT_HOME


def _load_episode_with_stats(src_dataset: LeRobotDataset, episode_idx: int) -> dict:
    """Load a single episode's metadata including stats from parquet file.

    Args:
        src_dataset: Source dataset
        episode_idx: Episode index to load

    Returns:
        dict containing episode metadata and stats
    """
    ep_meta = src_dataset.meta.episodes[episode_idx]
    chunk_idx = ep_meta["meta/episodes/chunk_index"]
    file_idx = ep_meta["meta/episodes/file_index"]

    parquet_path = src_dataset.root / DEFAULT_EPISODES_PATH.format(chunk_index=chunk_idx, file_index=file_idx)
    df = pd.read_parquet(parquet_path)

    episode_row = df[df["episode_index"] == episode_idx].iloc[0]

    return episode_row.to_dict()


def delete_episodes(
    dataset: LeRobotDataset,
    episode_indices: list[int],
    output_dir: str | Path | None = None,
    repo_id: str | None = None,
) -> LeRobotDataset:
    """Delete episodes from a LeRobotDataset and create a new dataset.

    Args:
        dataset: The source LeRobotDataset.
        episode_indices: List of episode indices to delete.
        output_dir: Directory to save the new dataset. If None, uses default location.
        repo_id: Repository ID for the new dataset. If None, appends "_modified" to original.
    """
    if not episode_indices:
        raise ValueError("No episodes to delete")

    valid_indices = set(range(dataset.meta.total_episodes))
    invalid = set(episode_indices) - valid_indices
    if invalid:
        raise ValueError(f"Invalid episode indices: {invalid}")

    logging.info(f"Deleting {len(episode_indices)} episodes from dataset")

    if repo_id is None:
        repo_id = f"{dataset.repo_id}_modified"
    output_dir = Path(output_dir) if output_dir is not None else HF_LEROBOT_HOME / repo_id

    episodes_to_keep = [i for i in range(dataset.meta.total_episodes) if i not in episode_indices]
    if not episodes_to_keep:
        raise ValueError("Cannot delete all episodes from dataset")

    new_meta = LeRobotDatasetMetadata.create(
        repo_id=repo_id,
        fps=dataset.meta.fps,
        features=dataset.meta.features,
        robot_type=dataset.meta.robot_type,
        root=output_dir,
        use_videos=len(dataset.meta.video_keys) > 0,
    )

    episode_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(episodes_to_keep)}

    video_metadata = None
    if dataset.meta.video_keys:
        video_metadata = _copy_and_reindex_videos(dataset, new_meta, episode_mapping)

    data_metadata = _copy_and_reindex_data(dataset, new_meta, episode_mapping)

    _copy_and_reindex_episodes_metadata(dataset, new_meta, episode_mapping, data_metadata, video_metadata)

    new_dataset = LeRobotDataset(
        repo_id=repo_id,
        root=output_dir,
        image_transforms=dataset.image_transforms,
        delta_timestamps=dataset.delta_timestamps,
        tolerance_s=dataset.tolerance_s,
    )

    logging.info(f"Created new dataset with {len(episodes_to_keep)} episodes")
    return new_dataset


def _resolve_frames_to_delete(
    dataset: LeRobotDataset,
    episode_index: int | None = None,
    frame_indices: list[int] | None = None,
    global_indices: list[int] | None = None,
) -> dict[int, set[int]]:
    using_episode_frames = episode_index is not None or frame_indices is not None
    using_global_indices = global_indices is not None

    if using_episode_frames == using_global_indices:
        raise ValueError(
            "Specify either (episode_index + frame_indices) or global_indices when deleting frames"
        )

    if using_episode_frames:
        if episode_index is None:
            raise ValueError("episode_index must be provided when frame_indices are used")
        if not frame_indices:
            raise ValueError("frame_indices must contain at least one frame index")
        if episode_index < 0 or episode_index >= dataset.meta.total_episodes:
            raise ValueError(f"Invalid episode index: {episode_index}")

        episode_length = int(dataset.meta.episodes[episode_index]["length"])
        invalid = sorted({frame_idx for frame_idx in frame_indices if frame_idx < 0 or frame_idx >= episode_length})
        if invalid:
            raise ValueError(
                f"Invalid frame indices for episode {episode_index}: {invalid}. "
                f"Episode length is {episode_length}."
            )

        return {episode_index: set(frame_indices)}

    assert global_indices is not None
    if len(global_indices) == 0:
        raise ValueError("global_indices must contain at least one frame index")

    invalid = sorted({idx for idx in global_indices if idx < 0 or idx >= dataset.meta.total_frames})
    if invalid:
        raise ValueError(
            f"Invalid global frame indices: {invalid}. Dataset contains {dataset.meta.total_frames} frames."
        )

    frames_to_delete: dict[int, set[int]] = {}
    for global_idx in sorted(set(global_indices)):
        item = dataset.hf_dataset[int(global_idx)]
        ep_idx = int(item["episode_index"])
        frame_idx = int(item["frame_index"])
        frames_to_delete.setdefault(ep_idx, set()).add(frame_idx)

    return frames_to_delete


def _rebuild_dataset_without_frames(
    src_dataset: LeRobotDataset,
    frames_to_delete: dict[int, set[int]],
    output_dir: str | Path | None = None,
    repo_id: str | None = None,
) -> LeRobotDataset:
    for ep_idx, frame_ids in sorted(frames_to_delete.items()):
        episode_length = int(src_dataset.meta.episodes[ep_idx]["length"])
        if len(frame_ids) >= episode_length:
            raise ValueError(
                f"Cannot delete all frames from episode {ep_idx}. Episode length is {episode_length}."
            )

    total_deleted = sum(len(frame_ids) for frame_ids in frames_to_delete.values())
    logging.info(f"Deleting {total_deleted} frame(s) across {len(frames_to_delete)} episode(s)")

    if repo_id is None:
        repo_id = f"{src_dataset.repo_id}_modified"
    output_dir = Path(output_dir) if output_dir is not None else HF_LEROBOT_HOME / repo_id

    new_dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=src_dataset.meta.fps,
        features=src_dataset.meta.features,
        robot_type=src_dataset.meta.robot_type,
        root=output_dir,
        use_videos=len(src_dataset.meta.video_keys) > 0,
        tolerance_s=src_dataset.tolerance_s,
        video_backend=src_dataset.video_backend,
    )
    new_dataset.meta.update_chunk_settings(
        chunks_size=src_dataset.meta.chunks_size,
        data_files_size_in_mb=src_dataset.meta.data_files_size_in_mb,
        video_files_size_in_mb=src_dataset.meta.video_files_size_in_mb,
    )

    feature_keys = [key for key in src_dataset.features if key not in DEFAULT_FEATURES]

    for ep_idx in tqdm(range(src_dataset.meta.total_episodes), desc="Rebuilding episodes"):
        ep_meta = src_dataset.meta.episodes[ep_idx]
        from_idx = int(ep_meta["dataset_from_index"])
        to_idx = int(ep_meta["dataset_to_index"])
        delete_local_indices = frames_to_delete.get(ep_idx, set())

        for global_idx in range(from_idx, to_idx):
            local_frame_idx = global_idx - from_idx
            if local_frame_idx in delete_local_indices:
                continue

            item = src_dataset[global_idx]
            frame = {}
            for key in feature_keys:
                value = item[key]
                if src_dataset.features[key]["dtype"] in ["image", "video"] and hasattr(value, "shape"):
                    expected_shape = tuple(src_dataset.features[key]["shape"])
                    chw_shape = (expected_shape[2], expected_shape[0], expected_shape[1])
                    if tuple(value.shape) == chw_shape:
                        value = value.permute(1, 2, 0) if isinstance(value, torch.Tensor) else np.transpose(value, (1, 2, 0))
                frame[key] = value
            frame["task"] = item["task"]
            new_dataset.add_frame(frame)

        new_dataset.save_episode()

    new_dataset.finalize()

    logging.info(
        "Created new dataset with %s episodes and %s frames",
        new_dataset.meta.total_episodes,
        new_dataset.meta.total_frames,
    )

    return LeRobotDataset(
        repo_id=repo_id,
        root=output_dir,
        image_transforms=src_dataset.image_transforms,
        delta_timestamps=src_dataset.delta_timestamps,
        tolerance_s=src_dataset.tolerance_s,
        download_videos=False,
        video_backend=src_dataset.video_backend,
    )


def _resolve_static_tail_frames_to_delete(
    dataset: LeRobotDataset,
    action_key: str = "action",
    change_threshold: float = 0.0,
    min_static_frames: int = 1,
    diff_mode: str = "max_abs",
) -> dict[int, set[int]]:
    if dataset.episodes is not None:
        raise ValueError("trim_static_tail_frames requires loading the full dataset, not a subset of episodes")
    if action_key not in dataset.hf_dataset.column_names:
        raise ValueError(f"Action key '{action_key}' not found in dataset columns")
    if change_threshold < 0:
        raise ValueError("change_threshold must be non-negative")
    if min_static_frames < 1:
        raise ValueError("min_static_frames must be at least 1")
    if diff_mode not in {"max_abs", "l2"}:
        raise ValueError("diff_mode must be one of: max_abs, l2")

    frames_to_delete: dict[int, set[int]] = {}

    for ep_idx in range(dataset.meta.total_episodes):
        ep_meta = dataset.meta.episodes[ep_idx]
        from_idx = int(ep_meta["dataset_from_index"])
        to_idx = int(ep_meta["dataset_to_index"])
        actions = np.asarray(dataset.hf_dataset[action_key][from_idx:to_idx], dtype=np.float32)

        if len(actions) <= 1:
            continue

        action_deltas = np.diff(actions, axis=0)
        if diff_mode == "max_abs":
            delta_values = np.abs(action_deltas).max(axis=1)
        else:
            delta_values = np.linalg.norm(action_deltas, axis=1)

        tail_static_transitions = 0
        for delta in delta_values[::-1]:
            if delta <= change_threshold:
                tail_static_transitions += 1
            else:
                break

        if tail_static_transitions < min_static_frames:
            continue

        first_deleted_frame = len(actions) - tail_static_transitions
        frames_to_delete[ep_idx] = set(range(first_deleted_frame, len(actions)))

    return frames_to_delete


def _trim_static_tail_video_dataset(
    src_dataset: LeRobotDataset,
    frames_to_delete: dict[int, set[int]],
    output_dir: str | Path | None = None,
    repo_id: str | None = None,
) -> LeRobotDataset:
    if repo_id is None:
        repo_id = f"{src_dataset.repo_id}_modified"
    output_dir = Path(output_dir) if output_dir is not None else HF_LEROBOT_HOME / repo_id

    dst_meta = LeRobotDatasetMetadata.create(
        repo_id=repo_id,
        fps=src_dataset.meta.fps,
        features=src_dataset.meta.features,
        robot_type=src_dataset.meta.robot_type,
        root=output_dir,
        use_videos=len(src_dataset.meta.video_keys) > 0,
    )
    dst_meta.update_chunk_settings(
        chunks_size=src_dataset.meta.chunks_size,
        data_files_size_in_mb=src_dataset.meta.data_files_size_in_mb,
        video_files_size_in_mb=src_dataset.meta.video_files_size_in_mb,
    )

    if src_dataset.meta.tasks is not None:
        write_tasks(src_dataset.meta.tasks, dst_meta.root)
        dst_meta.tasks = src_dataset.meta.tasks.copy()

    if src_dataset.meta.episodes is None:
        src_dataset.meta.episodes = load_episodes(src_dataset.meta.root)

    numeric_features = {
        key: ft for key, ft in src_dataset.meta.features.items() if ft["dtype"] not in ["image", "video"]
    }
    file_to_episodes: dict[Path, list[int]] = {}
    for ep_idx in range(src_dataset.meta.total_episodes):
        file_path = src_dataset.meta.get_data_file_path(ep_idx)
        file_to_episodes.setdefault(file_path, []).append(ep_idx)

    global_index = 0
    episode_lengths: dict[int, int] = {}
    episode_stats: dict[int, dict[str, dict]] = {}

    for src_path in tqdm(sorted(file_to_episodes.keys()), desc="Processing trimmed data files"):
        df = pd.read_parquet(src_dataset.root / src_path)
        rewritten_frames = []

        for ep_idx in sorted(file_to_episodes[src_path]):
            ep_df = df[df["episode_index"] == ep_idx].sort_values("frame_index").copy().reset_index(drop=True)
            delete_local_indices = frames_to_delete.get(ep_idx, set())
            if delete_local_indices:
                ep_df = ep_df[~ep_df["frame_index"].isin(delete_local_indices)].copy().reset_index(drop=True)

            if len(ep_df) == 0:
                raise ValueError(f"Episode {ep_idx} became empty after trimming")

            ep_df["frame_index"] = np.arange(len(ep_df), dtype=np.int64)
            ep_df["timestamp"] = ep_df["frame_index"].to_numpy(dtype=np.float32) / src_dataset.meta.fps
            ep_df["index"] = np.arange(global_index, global_index + len(ep_df), dtype=np.int64)

            stats_input = {}
            for key, ft in numeric_features.items():
                if key not in ep_df.columns:
                    continue
                shape = tuple(ft["shape"])
                if shape == (1,):
                    stats_input[key] = ep_df[key].to_numpy().reshape(-1, 1)
                else:
                    stats_input[key] = np.stack(ep_df[key].map(np.asarray).to_list())

            episode_stats[ep_idx] = compute_episode_stats(stats_input, numeric_features)
            episode_lengths[ep_idx] = len(ep_df)
            rewritten_frames.append(ep_df)
            global_index += len(ep_df)

        rewritten_df = pd.concat(rewritten_frames, ignore_index=True)
        dst_path = dst_meta.root / src_path
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        _write_parquet(rewritten_df, dst_path, dst_meta)

    video_metadata: dict[int, dict[str, int | float]] = {ep_idx: {} for ep_idx in range(src_dataset.meta.total_episodes)}
    for video_key in src_dataset.meta.video_keys:
        file_to_episodes: dict[tuple[int, int], list[int]] = {}
        for ep_idx in range(src_dataset.meta.total_episodes):
            src_ep = src_dataset.meta.episodes[ep_idx]
            chunk_idx = int(src_ep[f"videos/{video_key}/chunk_index"])
            file_idx = int(src_ep[f"videos/{video_key}/file_index"])
            file_to_episodes.setdefault((chunk_idx, file_idx), []).append(ep_idx)

        for (chunk_idx, file_idx), episode_indices in tqdm(
            sorted(file_to_episodes.items()), desc=f"Trimming {video_key} video files"
        ):
            if src_dataset.meta.video_path is None or dst_meta.video_path is None:
                raise ValueError("Source or destination metadata has no video_path defined")

            src_video_path = src_dataset.root / src_dataset.meta.video_path.format(
                video_key=video_key, chunk_index=chunk_idx, file_index=file_idx
            )
            dst_video_path = dst_meta.root / dst_meta.video_path.format(
                video_key=video_key, chunk_index=chunk_idx, file_index=file_idx
            )
            dst_video_path.parent.mkdir(parents=True, exist_ok=True)

            kept_ranges: list[tuple[float, float]] = []
            current_out_ts = 0.0
            for ep_idx in sorted(episode_indices):
                src_ep = src_dataset.meta.episodes[ep_idx]
                src_from_ts = float(src_ep[f"videos/{video_key}/from_timestamp"])
                duration = episode_lengths[ep_idx] / src_dataset.meta.fps
                kept_ranges.append((src_from_ts, src_from_ts + duration))
                video_metadata[ep_idx][f"videos/{video_key}/chunk_index"] = chunk_idx
                video_metadata[ep_idx][f"videos/{video_key}/file_index"] = file_idx
                video_metadata[ep_idx][f"videos/{video_key}/from_timestamp"] = current_out_ts
                video_metadata[ep_idx][f"videos/{video_key}/to_timestamp"] = current_out_ts + duration
                current_out_ts += duration

            _keep_episodes_from_video_with_av(
                src_video_path,
                dst_video_path,
                kept_ranges,
                fps=src_dataset.meta.fps,
            )

    for ep_idx in tqdm(range(src_dataset.meta.total_episodes), desc="Writing trimmed episode metadata"):
        episode_dict = _load_episode_with_stats(src_dataset, ep_idx)
        episode_dict["episode_index"] = ep_idx
        episode_dict["length"] = episode_lengths[ep_idx]

        for video_key in src_dataset.meta.video_keys:
            episode_dict[f"videos/{video_key}/chunk_index"] = video_metadata[ep_idx][
                f"videos/{video_key}/chunk_index"
            ]
            episode_dict[f"videos/{video_key}/file_index"] = video_metadata[ep_idx][
                f"videos/{video_key}/file_index"
            ]
            episode_dict[f"videos/{video_key}/from_timestamp"] = video_metadata[ep_idx][
                f"videos/{video_key}/from_timestamp"
            ]
            episode_dict[f"videos/{video_key}/to_timestamp"] = video_metadata[ep_idx][
                f"videos/{video_key}/to_timestamp"
            ]

        for feature_name, feature_stats in episode_stats[ep_idx].items():
            for stat_name, stat_value in feature_stats.items():
                episode_dict[f"stats/{feature_name}/{stat_name}"] = stat_value

        dst_meta._save_episode_metadata(episode_dict)

    dst_meta._close_writer()
    dst_meta.info.update(
        {
            "total_episodes": src_dataset.meta.total_episodes,
            "total_frames": global_index,
            "total_tasks": len(dst_meta.tasks) if dst_meta.tasks is not None else 0,
            "splits": src_dataset.meta.info.get("splits", {"train": f"0:{src_dataset.meta.total_episodes}"}),
        }
    )
    for key in dst_meta.video_keys:
        dst_meta.info["features"][key]["info"] = src_dataset.meta.info["features"][key].get("info", {})
    write_info(dst_meta.info, dst_meta.root)

    if src_dataset.meta.stats:
        updated_stats = {key: value for key, value in src_dataset.meta.stats.items() if key in dst_meta.features}
        aggregated_numeric_stats = aggregate_stats([episode_stats[idx] for idx in range(src_dataset.meta.total_episodes)])
        updated_stats.update(aggregated_numeric_stats)
        write_stats(updated_stats, dst_meta.root)

    return LeRobotDataset(
        repo_id=repo_id,
        root=output_dir,
        image_transforms=src_dataset.image_transforms,
        delta_timestamps=src_dataset.delta_timestamps,
        tolerance_s=src_dataset.tolerance_s,
        download_videos=False,
        video_backend=src_dataset.video_backend,
    )


def trim_static_tail_frames(
    dataset: LeRobotDataset,
    output_dir: str | Path | None = None,
    repo_id: str | None = None,
    action_key: str = "action",
    change_threshold: float = 0.0,
    min_static_frames: int = 1,
    diff_mode: str = "max_abs",
) -> LeRobotDataset:
    """Trim the static tail of each episode based on small action changes.

    This rebuilds the dataset and removes trailing frames whose action change from the previous
    frame stays below ``change_threshold`` for at least ``min_static_frames`` consecutive tail frames.

    Args:
        dataset: The source LeRobotDataset. Must be the full dataset, not a subset of episodes.
        output_dir: Directory to save the new dataset. If None, uses the default cache location.
        repo_id: Repository ID for the new dataset. If None, appends "_modified" to the original repo id.
        action_key: Dataset key used to measure action changes.
        change_threshold: Maximum per-step action change still considered static.
        min_static_frames: Minimum number of trailing frames required before trimming starts.
        diff_mode: Metric used to measure action deltas. Supported values are "max_abs" and "l2".
    """
    src_dataset = LeRobotDataset(
        dataset.repo_id,
        root=dataset.root,
        tolerance_s=dataset.tolerance_s,
        download_videos=False,
        video_backend=dataset.video_backend,
    )

    frames_to_delete = _resolve_static_tail_frames_to_delete(
        src_dataset,
        action_key=action_key,
        change_threshold=change_threshold,
        min_static_frames=min_static_frames,
        diff_mode=diff_mode,
    )

    trimmed_episodes = len(frames_to_delete)
    trimmed_frames = sum(len(frame_ids) for frame_ids in frames_to_delete.values())
    logging.info(
        "Trimming static tails with action_key=%s, threshold=%s, min_static_frames=%s, diff_mode=%s",
        action_key,
        change_threshold,
        min_static_frames,
        diff_mode,
    )
    logging.info("Detected %s trailing static frame(s) across %s episode(s)", trimmed_frames, trimmed_episodes)

    if src_dataset.meta.video_keys:
        logging.info("Using metadata-only trim path for video-backed dataset")
        return _trim_static_tail_video_dataset(
            src_dataset,
            frames_to_delete=frames_to_delete,
            output_dir=output_dir,
            repo_id=repo_id,
        )

    return _rebuild_dataset_without_frames(
        src_dataset,
        frames_to_delete=frames_to_delete,
        output_dir=output_dir,
        repo_id=repo_id,
    )


def delete_frames(
    dataset: LeRobotDataset,
    output_dir: str | Path | None = None,
    repo_id: str | None = None,
    episode_index: int | None = None,
    frame_indices: list[int] | None = None,
    global_indices: list[int] | None = None,
) -> LeRobotDataset:
    """Delete specific frames from a LeRobotDataset by rebuilding a consistent dataset.

    This operation intentionally rewrites the dataset instead of editing parquet/video files in place.
    Removing a single frame changes global indices, per-episode frame indices, timestamps, video offsets,
    and aggregated statistics. Rebuilding through the official LeRobot reader/writer path keeps all of
    these artifacts consistent.

    Args:
        dataset: The source dataset. Must be the full dataset, not a subset of episodes.
        output_dir: Directory to save the new dataset. If None, uses the default cache location.
        repo_id: Repository ID for the new dataset. If None, appends "_modified" to the original repo id.
        episode_index: Episode index containing the frame(s) to delete.
        frame_indices: Frame indices within ``episode_index`` to delete.
        global_indices: Absolute frame indices in dataset order to delete.
    """
    if dataset.episodes is not None:
        raise ValueError("delete_frames requires loading the full dataset, not a subset of episodes")

    src_dataset = LeRobotDataset(
        dataset.repo_id,
        root=dataset.root,
        tolerance_s=dataset.tolerance_s,
        download_videos=False,
        video_backend=dataset.video_backend,
    )

    frames_to_delete = _resolve_frames_to_delete(
        src_dataset,
        episode_index=episode_index,
        frame_indices=frame_indices,
        global_indices=global_indices,
    )

    return _rebuild_dataset_without_frames(
        src_dataset,
        frames_to_delete=frames_to_delete,
        output_dir=output_dir,
        repo_id=repo_id,
    )


def _count_trailing_static_frames(actions: torch.Tensor | np.ndarray, delta_threshold: float) -> int:
    if len(actions) < 2:
        return 0

    action_values = actions.detach().cpu().numpy() if isinstance(actions, torch.Tensor) else np.asarray(actions)
    action_values = action_values.astype(np.float64, copy=False).reshape(len(action_values), -1)
    action_deltas = np.linalg.norm(np.diff(action_values, axis=0), axis=1)

    trailing_static_frames = 0
    for delta in action_deltas[::-1]:
        if delta <= delta_threshold:
            trailing_static_frames += 1
        else:
            break

    return trailing_static_frames


def _resolve_trailing_static_global_indices(
    dataset: LeRobotDataset,
    action_key: str = "action",
    delta_threshold: float = 1e-6,
    min_static_frames: int = 1,
) -> list[int]:
    if dataset.episodes is not None:
        raise ValueError("trim_trailing_static_frames requires loading the full dataset, not a subset of episodes")
    if action_key not in dataset.features:
        raise ValueError(f"Unknown action key: {action_key}")
    if dataset.features[action_key]["dtype"] in ["image", "video"]:
        raise ValueError(f"Action key must be numeric, got visual feature: {action_key}")
    if delta_threshold < 0:
        raise ValueError("delta_threshold must be non-negative")
    if min_static_frames < 1:
        raise ValueError("min_static_frames must be at least 1")

    global_indices: list[int] = []
    trimmed_episode_count = 0

    for ep_idx in tqdm(range(dataset.meta.total_episodes), desc="Scanning trailing static frames"):
        ep_meta = dataset.meta.episodes[ep_idx]
        from_idx = int(ep_meta["dataset_from_index"])
        to_idx = int(ep_meta["dataset_to_index"])

        action_batch = dataset.hf_dataset[from_idx:to_idx][action_key]
        if len(action_batch) < 2:
            continue

        if isinstance(action_batch[0], torch.Tensor):
            actions = torch.stack(action_batch)
        else:
            actions = np.stack(action_batch)

        trailing_static_frames = _count_trailing_static_frames(actions, delta_threshold)
        if trailing_static_frames < min_static_frames:
            continue

        trimmed_episode_count += 1
        global_indices.extend(range(to_idx - trailing_static_frames, to_idx))

    logging.info(
        "Detected %s trailing static frame(s) across %s episode(s) using %s (delta_threshold=%s, min_static_frames=%s)",
        len(global_indices),
        trimmed_episode_count,
        action_key,
        delta_threshold,
        min_static_frames,
    )
    return global_indices


def trim_trailing_static_frames(
    dataset: LeRobotDataset,
    output_dir: str | Path | None = None,
    repo_id: str | None = None,
    action_key: str = "action",
    delta_threshold: float = 1e-6,
    min_static_frames: int = 1,
) -> LeRobotDataset:
    """Trim trailing frames whose action changes stay below a threshold.

    For each episode, this keeps the first frame of the final static segment and deletes the
    subsequent trailing frames whose L2 action delta against the previous frame is less than or
    equal to ``delta_threshold``.

    Args:
        dataset: The source dataset. Must be the full dataset, not a subset of episodes.
        output_dir: Directory to save the new dataset. If None, uses the default cache location.
        repo_id: Repository ID for the new dataset. If None, appends "_trimmed" to the original repo id.
        action_key: Numeric feature used to detect static tails. Defaults to ``action``.
        delta_threshold: Maximum per-frame L2 action delta considered static.
        min_static_frames: Minimum number of trailing frames to delete for an episode to be trimmed.
    """
    global_indices = _resolve_trailing_static_global_indices(
        dataset,
        action_key=action_key,
        delta_threshold=delta_threshold,
        min_static_frames=min_static_frames,
    )
    if not global_indices:
        raise ValueError("No trailing static frames matched the configured action threshold")

    if repo_id is None:
        repo_id = f"{dataset.repo_id}_trimmed"

    return delete_frames(
        dataset,
        output_dir=output_dir,
        repo_id=repo_id,
        global_indices=global_indices,
    )


def find_trailing_static_frame_indices(
    dataset: LeRobotDataset,
    action_key: str = "action",
    delta_threshold: float = 0.0,
    keep_last_n: int = 1,
    min_tail_length: int = 2,
) -> dict[int, list[int]]:
    """Find global frame indices belonging to trailing static tails in each episode.

    A trailing static tail is defined as the final contiguous suffix of an episode where the
    L2 distance between consecutive ``action_key`` values is below or equal to ``delta_threshold``.
    The earliest ``keep_last_n`` frames of that suffix are preserved, and the rest are returned as
    global dataset indices to delete.

    Args:
        dataset: The source dataset. Must be the full dataset, not an episode subset.
        action_key: Feature name used to detect static tails.
        delta_threshold: Maximum allowed L2 delta between consecutive actions inside the tail.
        keep_last_n: Number of frames to preserve at the beginning of the static suffix.
        min_tail_length: Minimum detected suffix length before trimming is applied.

    Returns:
        Mapping from episode index to sorted global indices to delete.
    """
    if dataset.episodes is not None:
        raise ValueError("find_trailing_static_frame_indices requires loading the full dataset")
    if action_key not in dataset.meta.features:
        raise ValueError(f"Feature '{action_key}' not found in dataset")
    if delta_threshold < 0:
        raise ValueError("delta_threshold must be >= 0")
    if keep_last_n < 0:
        raise ValueError("keep_last_n must be >= 0")
    if min_tail_length < 1:
        raise ValueError("min_tail_length must be >= 1")

    episodes_by_file: dict[Path, list[int]] = {}
    for ep_idx in range(dataset.meta.total_episodes):
        data_path = dataset.root / dataset.meta.get_data_file_path(ep_idx)
        episodes_by_file.setdefault(data_path, []).append(ep_idx)

    frames_to_delete: dict[int, list[int]] = {}
    columns = ["episode_index", "frame_index", "index", action_key]

    for data_path, episode_indices in episodes_by_file.items():
        file_df = pd.read_parquet(data_path, columns=columns)

        for ep_idx in episode_indices:
            ep_df = file_df[file_df["episode_index"] == ep_idx].sort_values("frame_index")
            if len(ep_df) <= keep_last_n:
                continue

            actions = np.stack(ep_df[action_key].map(np.asarray).to_list())
            tail_length = 1

            for frame_idx in range(len(actions) - 1, 0, -1):
                delta = float(np.linalg.norm(actions[frame_idx] - actions[frame_idx - 1]))
                if delta <= delta_threshold:
                    tail_length += 1
                else:
                    break

            if tail_length < min_tail_length:
                continue

            delete_count = tail_length - keep_last_n
            if delete_count <= 0:
                continue

            frames_to_delete[ep_idx] = ep_df["index"].iloc[-delete_count:].astype(int).tolist()

    return frames_to_delete


def split_dataset(
    dataset: LeRobotDataset,
    splits: dict[str, float | list[int]],
    output_dir: str | Path | None = None,
) -> dict[str, LeRobotDataset]:
    """Split a LeRobotDataset into multiple smaller datasets.

    Args:
        dataset: The source LeRobotDataset to split.
        splits: Either a dict mapping split names to episode indices, or a dict mapping
                split names to fractions (must sum to <= 1.0).
        output_dir: Base directory for output datasets. If None, uses default location.

    Examples:
      Split by specific episodes
        splits = {"train": [0, 1, 2], "val": [3, 4]}
        datasets = split_dataset(dataset, splits)

      Split by fractions
        splits = {"train": 0.8, "val": 0.2}
        datasets = split_dataset(dataset, splits)
    """
    if not splits:
        raise ValueError("No splits provided")

    if all(isinstance(v, float) for v in splits.values()):
        splits = _fractions_to_episode_indices(dataset.meta.total_episodes, splits)

    all_episodes = set()
    for split_name, episodes in splits.items():
        if not episodes:
            raise ValueError(f"Split '{split_name}' has no episodes")
        episode_set = set(episodes)
        if episode_set & all_episodes:
            raise ValueError("Episodes cannot appear in multiple splits")
        all_episodes.update(episode_set)

    valid_indices = set(range(dataset.meta.total_episodes))
    invalid = all_episodes - valid_indices
    if invalid:
        raise ValueError(f"Invalid episode indices: {invalid}")

    if output_dir is not None:
        output_dir = Path(output_dir)

    result_datasets = {}

    for split_name, episodes in splits.items():
        logging.info(f"Creating split '{split_name}' with {len(episodes)} episodes")

        split_repo_id = f"{dataset.repo_id}_{split_name}"

        split_output_dir = (
            output_dir / split_name if output_dir is not None else HF_LEROBOT_HOME / split_repo_id
        )

        episode_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted(episodes))}

        new_meta = LeRobotDatasetMetadata.create(
            repo_id=split_repo_id,
            fps=dataset.meta.fps,
            features=dataset.meta.features,
            robot_type=dataset.meta.robot_type,
            root=split_output_dir,
            use_videos=len(dataset.meta.video_keys) > 0,
            chunks_size=dataset.meta.chunks_size,
            data_files_size_in_mb=dataset.meta.data_files_size_in_mb,
            video_files_size_in_mb=dataset.meta.video_files_size_in_mb,
        )

        video_metadata = None
        if dataset.meta.video_keys:
            video_metadata = _copy_and_reindex_videos(dataset, new_meta, episode_mapping)

        data_metadata = _copy_and_reindex_data(dataset, new_meta, episode_mapping)

        _copy_and_reindex_episodes_metadata(dataset, new_meta, episode_mapping, data_metadata, video_metadata)

        new_dataset = LeRobotDataset(
            repo_id=split_repo_id,
            root=split_output_dir,
            image_transforms=dataset.image_transforms,
            delta_timestamps=dataset.delta_timestamps,
            tolerance_s=dataset.tolerance_s,
        )

        result_datasets[split_name] = new_dataset

    return result_datasets


def merge_datasets(
    datasets: list[LeRobotDataset],
    output_repo_id: str,
    output_dir: str | Path | None = None,
) -> LeRobotDataset:
    """Merge multiple LeRobotDatasets into a single dataset.

    This is a wrapper around the aggregate_datasets functionality with a cleaner API.

    Args:
        datasets: List of LeRobotDatasets to merge.
        output_repo_id: Repository ID for the merged dataset.
        output_dir: Directory to save the merged dataset. If None, uses default location.
    """
    if not datasets:
        raise ValueError("No datasets to merge")

    output_dir = Path(output_dir) if output_dir is not None else HF_LEROBOT_HOME / output_repo_id

    repo_ids = [ds.repo_id for ds in datasets]
    roots = [ds.root for ds in datasets]

    aggregate_datasets(
        repo_ids=repo_ids,
        aggr_repo_id=output_repo_id,
        roots=roots,
        aggr_root=output_dir,
    )

    merged_dataset = LeRobotDataset(
        repo_id=output_repo_id,
        root=output_dir,
        image_transforms=datasets[0].image_transforms,
        delta_timestamps=datasets[0].delta_timestamps,
        tolerance_s=datasets[0].tolerance_s,
    )

    return merged_dataset


def modify_features(
    dataset: LeRobotDataset,
    add_features: dict[str, tuple[np.ndarray | torch.Tensor | Callable, dict]] | None = None,
    remove_features: str | list[str] | None = None,
    output_dir: str | Path | None = None,
    repo_id: str | None = None,
) -> LeRobotDataset:
    """Modify a LeRobotDataset by adding and/or removing features in a single pass.

    This is the most efficient way to modify features, as it only copies the dataset once
    regardless of how many features are being added or removed.

    Args:
        dataset: The source LeRobotDataset.
        add_features: Optional dict mapping feature names to (feature_values, feature_info) tuples.
        remove_features: Optional feature name(s) to remove. Can be a single string or list.
        output_dir: Directory to save the new dataset. If None, uses default location.
        repo_id: Repository ID for the new dataset. If None, appends "_modified" to original.

    Returns:
        New dataset with features modified.

    Example:
        new_dataset = modify_features(
            dataset,
            add_features={
                "reward": (reward_array, {"dtype": "float32", "shape": [1], "names": None}),
            },
            remove_features=["old_feature"],
            output_dir="./output",
        )
    """
    if add_features is None and remove_features is None:
        raise ValueError("Must specify at least one of add_features or remove_features")

    remove_features_list: list[str] = []
    if remove_features is not None:
        remove_features_list = [remove_features] if isinstance(remove_features, str) else remove_features

    if add_features:
        required_keys = {"dtype", "shape"}
        for feature_name, (_, feature_info) in add_features.items():
            if feature_name in dataset.meta.features:
                raise ValueError(f"Feature '{feature_name}' already exists in dataset")

            if not required_keys.issubset(feature_info.keys()):
                raise ValueError(f"feature_info for '{feature_name}' must contain keys: {required_keys}")

    if remove_features_list:
        for name in remove_features_list:
            if name not in dataset.meta.features:
                raise ValueError(f"Feature '{name}' not found in dataset")

        required_features = {"timestamp", "frame_index", "episode_index", "index", "task_index"}
        if any(name in required_features for name in remove_features_list):
            raise ValueError(f"Cannot remove required features: {required_features}")

    if repo_id is None:
        repo_id = f"{dataset.repo_id}_modified"
    output_dir = Path(output_dir) if output_dir is not None else HF_LEROBOT_HOME / repo_id

    new_features = dataset.meta.features.copy()

    if remove_features_list:
        for name in remove_features_list:
            new_features.pop(name, None)

    if add_features:
        for feature_name, (_, feature_info) in add_features.items():
            new_features[feature_name] = feature_info

    video_keys_to_remove = [name for name in remove_features_list if name in dataset.meta.video_keys]
    remaining_video_keys = [k for k in dataset.meta.video_keys if k not in video_keys_to_remove]

    new_meta = LeRobotDatasetMetadata.create(
        repo_id=repo_id,
        fps=dataset.meta.fps,
        features=new_features,
        robot_type=dataset.meta.robot_type,
        root=output_dir,
        use_videos=len(remaining_video_keys) > 0,
    )

    _copy_data_with_feature_changes(
        dataset=dataset,
        new_meta=new_meta,
        add_features=add_features,
        remove_features=remove_features_list if remove_features_list else None,
    )

    if new_meta.video_keys:
        _copy_videos(dataset, new_meta, exclude_keys=video_keys_to_remove if video_keys_to_remove else None)

    new_dataset = LeRobotDataset(
        repo_id=repo_id,
        root=output_dir,
        image_transforms=dataset.image_transforms,
        delta_timestamps=dataset.delta_timestamps,
        tolerance_s=dataset.tolerance_s,
    )

    return new_dataset


def add_features(
    dataset: LeRobotDataset,
    features: dict[str, tuple[np.ndarray | torch.Tensor | Callable, dict]],
    output_dir: str | Path | None = None,
    repo_id: str | None = None,
) -> LeRobotDataset:
    """Add multiple features to a LeRobotDataset in a single pass.

    This is more efficient than calling add_feature() multiple times, as it only
    copies the dataset once regardless of how many features are being added.

    Args:
        dataset: The source LeRobotDataset.
        features: Dictionary mapping feature names to (feature_values, feature_info) tuples.
        output_dir: Directory to save the new dataset. If None, uses default location.
        repo_id: Repository ID for the new dataset. If None, appends "_modified" to original.

    Returns:
        New dataset with all features added.

    Example:
        features = {
            "task_embedding": (task_emb_array, {"dtype": "float32", "shape": [384], "names": None}),
            "cam1_embedding": (cam1_emb_array, {"dtype": "float32", "shape": [768], "names": None}),
            "cam2_embedding": (cam2_emb_array, {"dtype": "float32", "shape": [768], "names": None}),
        }
        new_dataset = add_features(dataset, features, output_dir="./output", repo_id="my_dataset")
    """
    if not features:
        raise ValueError("No features provided")

    return modify_features(
        dataset=dataset,
        add_features=features,
        remove_features=None,
        output_dir=output_dir,
        repo_id=repo_id,
    )


def remove_feature(
    dataset: LeRobotDataset,
    feature_names: str | list[str],
    output_dir: str | Path | None = None,
    repo_id: str | None = None,
) -> LeRobotDataset:
    """Remove features from a LeRobotDataset.

    Args:
        dataset: The source LeRobotDataset.
        feature_names: Name(s) of features to remove. Can be a single string or list.
        output_dir: Directory to save the new dataset. If None, uses default location.
        repo_id: Repository ID for the new dataset. If None, appends "_modified" to original.

    Returns:
        New dataset with features removed.
    """
    return modify_features(
        dataset=dataset,
        add_features=None,
        remove_features=feature_names,
        output_dir=output_dir,
        repo_id=repo_id,
    )


def _fractions_to_episode_indices(
    total_episodes: int,
    splits: dict[str, float],
) -> dict[str, list[int]]:
    """Convert split fractions to episode indices."""
    if sum(splits.values()) > 1.0:
        raise ValueError("Split fractions must sum to <= 1.0")

    indices = list(range(total_episodes))
    result = {}
    start_idx = 0

    for split_name, fraction in splits.items():
        num_episodes = int(total_episodes * fraction)
        if num_episodes == 0:
            logging.warning(f"Split '{split_name}' has no episodes, skipping...")
            continue
        end_idx = start_idx + num_episodes
        if split_name == list(splits.keys())[-1]:
            end_idx = total_episodes
        result[split_name] = indices[start_idx:end_idx]
        start_idx = end_idx

    return result


def _copy_and_reindex_data(
    src_dataset: LeRobotDataset,
    dst_meta: LeRobotDatasetMetadata,
    episode_mapping: dict[int, int],
) -> dict[int, dict]:
    """Copy and filter data files, only modifying files with deleted episodes.

    Args:
        src_dataset: Source dataset to copy from
        dst_meta: Destination metadata object
        episode_mapping: Mapping from old episode indices to new indices

    Returns:
        dict mapping episode index to its data file metadata (chunk_index, file_index, etc.)
    """
    if src_dataset.meta.episodes is None:
        src_dataset.meta.episodes = load_episodes(src_dataset.meta.root)

    file_to_episodes: dict[Path, set[int]] = {}
    for old_idx in episode_mapping:
        file_path = src_dataset.meta.get_data_file_path(old_idx)
        if file_path not in file_to_episodes:
            file_to_episodes[file_path] = set()
        file_to_episodes[file_path].add(old_idx)

    global_index = 0
    episode_data_metadata: dict[int, dict] = {}

    if dst_meta.tasks is None:
        all_task_indices = set()
        for src_path in file_to_episodes:
            df = pd.read_parquet(src_dataset.root / src_path)
            mask = df["episode_index"].isin(list(episode_mapping.keys()))
            task_series: pd.Series = df[mask]["task_index"]
            all_task_indices.update(task_series.unique().tolist())
        tasks = [src_dataset.meta.tasks.iloc[idx].name for idx in all_task_indices]
        dst_meta.save_episode_tasks(list(set(tasks)))

    task_mapping = {}
    for old_task_idx in range(len(src_dataset.meta.tasks)):
        task_name = src_dataset.meta.tasks.iloc[old_task_idx].name
        new_task_idx = dst_meta.get_task_index(task_name)
        if new_task_idx is not None:
            task_mapping[old_task_idx] = new_task_idx

    for src_path in tqdm(sorted(file_to_episodes.keys()), desc="Processing data files"):
        df = pd.read_parquet(src_dataset.root / src_path)

        all_episodes_in_file = set(df["episode_index"].unique())
        episodes_to_keep = file_to_episodes[src_path]

        if all_episodes_in_file == episodes_to_keep:
            df["episode_index"] = df["episode_index"].replace(episode_mapping)
            df["index"] = range(global_index, global_index + len(df))
            df["task_index"] = df["task_index"].replace(task_mapping)

            first_ep_old_idx = min(episodes_to_keep)
            src_ep = src_dataset.meta.episodes[first_ep_old_idx]
            chunk_idx = src_ep["data/chunk_index"]
            file_idx = src_ep["data/file_index"]
        else:
            mask = df["episode_index"].isin(list(episode_mapping.keys()))
            df = df[mask].copy().reset_index(drop=True)

            if len(df) == 0:
                continue

            df["episode_index"] = df["episode_index"].replace(episode_mapping)
            df["index"] = range(global_index, global_index + len(df))
            df["task_index"] = df["task_index"].replace(task_mapping)

            first_ep_old_idx = min(episodes_to_keep)
            src_ep = src_dataset.meta.episodes[first_ep_old_idx]
            chunk_idx = src_ep["data/chunk_index"]
            file_idx = src_ep["data/file_index"]

        dst_path = dst_meta.root / DEFAULT_DATA_PATH.format(chunk_index=chunk_idx, file_index=file_idx)
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        _write_parquet(df, dst_path, dst_meta)

        for ep_old_idx in episodes_to_keep:
            ep_new_idx = episode_mapping[ep_old_idx]
            ep_df = df[df["episode_index"] == ep_new_idx]
            episode_data_metadata[ep_new_idx] = {
                "data/chunk_index": chunk_idx,
                "data/file_index": file_idx,
                "dataset_from_index": int(ep_df["index"].min()),
                "dataset_to_index": int(ep_df["index"].max() + 1),
            }

        global_index += len(df)

    return episode_data_metadata


def _keep_episodes_from_video_with_av(
    input_path: Path,
    output_path: Path,
    episodes_to_keep: list[tuple[float, float]],
    fps: float,
    vcodec: str = "libsvtav1",
    pix_fmt: str = "yuv420p",
) -> None:
    """Keep only specified episodes from a video file using PyAV.

    This function decodes frames from specified time ranges and re-encodes them with
    properly reset timestamps to ensure monotonic progression.

    Args:
        input_path: Source video file path.
        output_path: Destination video file path.
        episodes_to_keep: List of (start_time, end_time) tuples for episodes to keep.
        fps: Frame rate of the video.
        vcodec: Video codec to use for encoding.
        pix_fmt: Pixel format for output video.
    """
    from fractions import Fraction

    import av

    if not episodes_to_keep:
        raise ValueError("No episodes to keep")

    in_container = av.open(str(input_path))

    # Check if video stream exists.
    if not in_container.streams.video:
        raise ValueError(
            f"No video streams found in {input_path}. "
            "The video file may be corrupted or empty. "
            "Try re-downloading the dataset or checking the video file."
        )

    v_in = in_container.streams.video[0]

    out = av.open(str(output_path), mode="w")

    # Convert fps to Fraction for PyAV compatibility.
    fps_fraction = Fraction(fps).limit_denominator(1000)
    v_out = out.add_stream(vcodec, rate=fps_fraction)

    # PyAV type stubs don't distinguish video streams from audio/subtitle streams.
    v_out.width = v_in.codec_context.width
    v_out.height = v_in.codec_context.height
    v_out.pix_fmt = pix_fmt

    # Set time_base to match the frame rate for proper timestamp handling.
    v_out.time_base = Fraction(1, int(fps))

    out.start_encoding()

    # Create set of (start, end) ranges for fast lookup.
    # Convert to a sorted list for efficient checking.
    time_ranges = sorted(episodes_to_keep)

    # Track frame index for setting PTS and current range being processed.
    frame_count = 0
    range_idx = 0

    # Read through entire video once and filter frames.
    for packet in in_container.demux(v_in):
        for frame in packet.decode():
            if frame is None:
                continue

            # Get frame timestamp.
            frame_time = float(frame.pts * frame.time_base) if frame.pts is not None else 0.0

            # Check if frame is in any of our desired time ranges.
            # Skip ranges that have already passed.
            while range_idx < len(time_ranges) and frame_time >= time_ranges[range_idx][1]:
                range_idx += 1

            # If we've passed all ranges, stop processing.
            if range_idx >= len(time_ranges):
                break

            # Check if frame is in current range.
            start_ts, end_ts = time_ranges[range_idx]
            if frame_time < start_ts:
                continue

            # Frame is in range - create a new frame with reset timestamps.
            # We need to create a copy to avoid modifying the original.
            new_frame = frame.reformat(width=v_out.width, height=v_out.height, format=v_out.pix_fmt)
            new_frame.pts = frame_count
            new_frame.time_base = Fraction(1, int(fps))

            # Encode and mux the frame.
            for pkt in v_out.encode(new_frame):
                out.mux(pkt)

            frame_count += 1

    # Flush encoder.
    for pkt in v_out.encode():
        out.mux(pkt)

    out.close()
    in_container.close()


def _copy_and_reindex_videos(
    src_dataset: LeRobotDataset,
    dst_meta: LeRobotDatasetMetadata,
    episode_mapping: dict[int, int],
    vcodec: str = "libsvtav1",
    pix_fmt: str = "yuv420p",
) -> dict[int, dict]:
    """Copy and filter video files, only re-encoding files with deleted episodes.

    For video files that only contain kept episodes, we copy them directly.
    For files with mixed kept/deleted episodes, we use PyAV filters to efficiently
    re-encode only the desired segments.

    Args:
        src_dataset: Source dataset to copy from
        dst_meta: Destination metadata object
        episode_mapping: Mapping from old episode indices to new indices

    Returns:
        dict mapping episode index to its video metadata (chunk_index, file_index, timestamps)
    """
    if src_dataset.meta.episodes is None:
        src_dataset.meta.episodes = load_episodes(src_dataset.meta.root)

    episodes_video_metadata: dict[int, dict] = {new_idx: {} for new_idx in episode_mapping.values()}

    for video_key in src_dataset.meta.video_keys:
        logging.info(f"Processing videos for {video_key}")

        if dst_meta.video_path is None:
            raise ValueError("Destination metadata has no video_path defined")

        file_to_episodes: dict[tuple[int, int], list[int]] = {}
        for old_idx in episode_mapping:
            src_ep = src_dataset.meta.episodes[old_idx]
            chunk_idx = src_ep[f"videos/{video_key}/chunk_index"]
            file_idx = src_ep[f"videos/{video_key}/file_index"]
            file_key = (chunk_idx, file_idx)
            if file_key not in file_to_episodes:
                file_to_episodes[file_key] = []
            file_to_episodes[file_key].append(old_idx)

        for (src_chunk_idx, src_file_idx), episodes_in_file in tqdm(
            sorted(file_to_episodes.items()), desc=f"Processing {video_key} video files"
        ):
            all_episodes_in_file = [
                ep_idx
                for ep_idx in range(src_dataset.meta.total_episodes)
                if src_dataset.meta.episodes[ep_idx].get(f"videos/{video_key}/chunk_index") == src_chunk_idx
                and src_dataset.meta.episodes[ep_idx].get(f"videos/{video_key}/file_index") == src_file_idx
            ]

            episodes_to_keep_set = set(episodes_in_file)
            all_in_file_set = set(all_episodes_in_file)

            if all_in_file_set == episodes_to_keep_set:
                assert src_dataset.meta.video_path is not None
                src_video_path = src_dataset.root / src_dataset.meta.video_path.format(
                    video_key=video_key, chunk_index=src_chunk_idx, file_index=src_file_idx
                )
                dst_video_path = dst_meta.root / dst_meta.video_path.format(
                    video_key=video_key, chunk_index=src_chunk_idx, file_index=src_file_idx
                )
                dst_video_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(src_video_path, dst_video_path)

                for old_idx in episodes_in_file:
                    new_idx = episode_mapping[old_idx]
                    src_ep = src_dataset.meta.episodes[old_idx]
                    episodes_video_metadata[new_idx][f"videos/{video_key}/chunk_index"] = src_chunk_idx
                    episodes_video_metadata[new_idx][f"videos/{video_key}/file_index"] = src_file_idx
                    episodes_video_metadata[new_idx][f"videos/{video_key}/from_timestamp"] = src_ep[
                        f"videos/{video_key}/from_timestamp"
                    ]
                    episodes_video_metadata[new_idx][f"videos/{video_key}/to_timestamp"] = src_ep[
                        f"videos/{video_key}/to_timestamp"
                    ]
            else:
                # Build list of time ranges to keep, in sorted order.
                sorted_keep_episodes = sorted(episodes_in_file, key=lambda x: episode_mapping[x])
                episodes_to_keep_ranges: list[tuple[float, float]] = []

                for old_idx in sorted_keep_episodes:
                    src_ep = src_dataset.meta.episodes[old_idx]
                    from_ts = src_ep[f"videos/{video_key}/from_timestamp"]
                    to_ts = src_ep[f"videos/{video_key}/to_timestamp"]
                    episodes_to_keep_ranges.append((from_ts, to_ts))

                # Use PyAV filters to efficiently re-encode only the desired segments.
                assert src_dataset.meta.video_path is not None
                src_video_path = src_dataset.root / src_dataset.meta.video_path.format(
                    video_key=video_key, chunk_index=src_chunk_idx, file_index=src_file_idx
                )
                dst_video_path = dst_meta.root / dst_meta.video_path.format(
                    video_key=video_key, chunk_index=src_chunk_idx, file_index=src_file_idx
                )
                dst_video_path.parent.mkdir(parents=True, exist_ok=True)

                logging.info(
                    f"Re-encoding {video_key} (chunk {src_chunk_idx}, file {src_file_idx}) "
                    f"with {len(episodes_to_keep_ranges)} episodes"
                )
                _keep_episodes_from_video_with_av(
                    src_video_path,
                    dst_video_path,
                    episodes_to_keep_ranges,
                    src_dataset.meta.fps,
                    vcodec,
                    pix_fmt,
                )

                cumulative_ts = 0.0
                for old_idx in sorted_keep_episodes:
                    new_idx = episode_mapping[old_idx]
                    src_ep = src_dataset.meta.episodes[old_idx]
                    ep_length = src_ep["length"]
                    ep_duration = ep_length / src_dataset.meta.fps

                    episodes_video_metadata[new_idx][f"videos/{video_key}/chunk_index"] = src_chunk_idx
                    episodes_video_metadata[new_idx][f"videos/{video_key}/file_index"] = src_file_idx
                    episodes_video_metadata[new_idx][f"videos/{video_key}/from_timestamp"] = cumulative_ts
                    episodes_video_metadata[new_idx][f"videos/{video_key}/to_timestamp"] = (
                        cumulative_ts + ep_duration
                    )

                    cumulative_ts += ep_duration

    return episodes_video_metadata


def _copy_and_reindex_episodes_metadata(
    src_dataset: LeRobotDataset,
    dst_meta: LeRobotDatasetMetadata,
    episode_mapping: dict[int, int],
    data_metadata: dict[int, dict],
    video_metadata: dict[int, dict] | None = None,
) -> None:
    """Copy and reindex episodes metadata using provided data and video metadata.

    Args:
        src_dataset: Source dataset to copy from
        dst_meta: Destination metadata object
        episode_mapping: Mapping from old episode indices to new indices
        data_metadata: Dict mapping new episode index to its data file metadata
        video_metadata: Optional dict mapping new episode index to its video metadata
    """
    from lerobot.datasets.utils import flatten_dict

    if src_dataset.meta.episodes is None:
        src_dataset.meta.episodes = load_episodes(src_dataset.meta.root)

    all_stats = []
    total_frames = 0

    for old_idx, new_idx in tqdm(
        sorted(episode_mapping.items(), key=lambda x: x[1]), desc="Processing episodes metadata"
    ):
        src_episode_full = _load_episode_with_stats(src_dataset, old_idx)

        src_episode = src_dataset.meta.episodes[old_idx]

        episode_meta = data_metadata[new_idx].copy()

        if video_metadata and new_idx in video_metadata:
            episode_meta.update(video_metadata[new_idx])

        # Extract episode statistics from parquet metadata.
        # Note (maractingi): When pandas/pyarrow serializes numpy arrays with shape (3, 1, 1) to parquet,
        # they are being deserialized as nested object arrays like:
        #   array([array([array([0.])]), array([array([0.])]), array([array([0.])])])
        # This happens particularly with image/video statistics. We need to detect and flatten
        # these nested structures back to proper (3, 1, 1) arrays so aggregate_stats can process them.
        episode_stats = {}
        for key in src_episode_full:
            if key.startswith("stats/"):
                stat_key = key.replace("stats/", "")
                parts = stat_key.split("/")
                if len(parts) == 2:
                    feature_name, stat_name = parts
                    if feature_name not in episode_stats:
                        episode_stats[feature_name] = {}

                    value = src_episode_full[key]

                    if feature_name in src_dataset.meta.features:
                        feature_dtype = src_dataset.meta.features[feature_name]["dtype"]
                        if feature_dtype in ["image", "video"] and stat_name != "count":
                            if isinstance(value, np.ndarray) and value.dtype == object:
                                flat_values = []
                                for item in value:
                                    while isinstance(item, np.ndarray):
                                        item = item.flatten()[0]
                                    flat_values.append(item)
                                value = np.array(flat_values, dtype=np.float64).reshape(3, 1, 1)
                            elif isinstance(value, np.ndarray) and value.shape == (3,):
                                value = value.reshape(3, 1, 1)

                    episode_stats[feature_name][stat_name] = value

        all_stats.append(episode_stats)

        episode_dict = {
            "episode_index": new_idx,
            "tasks": src_episode["tasks"],
            "length": src_episode["length"],
        }
        episode_dict.update(episode_meta)
        episode_dict.update(flatten_dict({"stats": episode_stats}))
        dst_meta._save_episode_metadata(episode_dict)

        total_frames += src_episode["length"]

    dst_meta._close_writer()

    dst_meta.info.update(
        {
            "total_episodes": len(episode_mapping),
            "total_frames": total_frames,
            "total_tasks": len(dst_meta.tasks) if dst_meta.tasks is not None else 0,
            "splits": {"train": f"0:{len(episode_mapping)}"},
        }
    )
    write_info(dst_meta.info, dst_meta.root)

    if not all_stats:
        logging.warning("No statistics found to aggregate")
        return

    logging.info(f"Aggregating statistics for {len(all_stats)} episodes")
    aggregated_stats = aggregate_stats(all_stats)
    filtered_stats = {k: v for k, v in aggregated_stats.items() if k in dst_meta.features}
    write_stats(filtered_stats, dst_meta.root)


def _write_parquet(df: pd.DataFrame, path: Path, meta: LeRobotDatasetMetadata) -> None:
    """Write DataFrame to parquet

    This ensures images are properly embedded and the file can be loaded correctly by HF datasets.
    """
    from lerobot.datasets.utils import embed_images, get_hf_features_from_features

    hf_features = get_hf_features_from_features(meta.features)
    ep_dataset = datasets.Dataset.from_dict(df.to_dict(orient="list"), features=hf_features, split="train")

    if len(meta.image_keys) > 0:
        ep_dataset = embed_images(ep_dataset)

    table = ep_dataset.with_format("arrow")[:]
    writer = pq.ParquetWriter(path, schema=table.schema, compression="snappy", use_dictionary=True)
    writer.write_table(table)
    writer.close()


def _save_data_chunk(
    df: pd.DataFrame,
    meta: LeRobotDatasetMetadata,
    chunk_idx: int = 0,
    file_idx: int = 0,
) -> tuple[int, int, dict[int, dict]]:
    """Save a data chunk and return updated indices and episode metadata.

    Returns:
        tuple: (next_chunk_idx, next_file_idx, episode_metadata_dict)
            where episode_metadata_dict maps episode_index to its data file metadata
    """
    path = meta.root / DEFAULT_DATA_PATH.format(chunk_index=chunk_idx, file_index=file_idx)
    path.parent.mkdir(parents=True, exist_ok=True)

    _write_parquet(df, path, meta)

    episode_metadata = {}
    for ep_idx in df["episode_index"].unique():
        ep_df = df[df["episode_index"] == ep_idx]
        episode_metadata[ep_idx] = {
            "data/chunk_index": chunk_idx,
            "data/file_index": file_idx,
            "dataset_from_index": int(ep_df["index"].min()),
            "dataset_to_index": int(ep_df["index"].max() + 1),
        }

    file_size = get_parquet_file_size_in_mb(path)
    if file_size >= DEFAULT_DATA_FILE_SIZE_IN_MB * 0.9:
        chunk_idx, file_idx = update_chunk_file_indices(chunk_idx, file_idx, DEFAULT_CHUNK_SIZE)

    return chunk_idx, file_idx, episode_metadata


def _copy_data_with_feature_changes(
    dataset: LeRobotDataset,
    new_meta: LeRobotDatasetMetadata,
    add_features: dict[str, tuple] | None = None,
    remove_features: list[str] | None = None,
) -> None:
    """Copy data while adding or removing features."""
    data_dir = dataset.root / DATA_DIR
    parquet_files = sorted(data_dir.glob("*/*.parquet"))

    if not parquet_files:
        raise ValueError(f"No parquet files found in {data_dir}")

    frame_idx = 0

    for src_path in tqdm(parquet_files, desc="Processing data files"):
        df = pd.read_parquet(src_path).reset_index(drop=True)

        relative_path = src_path.relative_to(dataset.root)
        chunk_dir = relative_path.parts[1]
        file_name = relative_path.parts[2]

        chunk_idx = int(chunk_dir.split("-")[1])
        file_idx = int(file_name.split("-")[1].split(".")[0])

        if remove_features:
            df = df.drop(columns=remove_features, errors="ignore")

        if add_features:
            end_idx = frame_idx + len(df)
            for feature_name, (values, _) in add_features.items():
                if callable(values):
                    feature_values = []
                    for _, row in df.iterrows():
                        ep_idx = row["episode_index"]
                        frame_in_ep = row["frame_index"]
                        value = values(row.to_dict(), ep_idx, frame_in_ep)
                        if isinstance(value, np.ndarray) and value.size == 1:
                            value = value.item()
                        feature_values.append(value)
                    df[feature_name] = feature_values
                else:
                    feature_slice = values[frame_idx:end_idx]
                    if len(feature_slice.shape) > 1 and feature_slice.shape[1] == 1:
                        df[feature_name] = feature_slice.flatten()
                    else:
                        df[feature_name] = feature_slice
            frame_idx = end_idx

        # Write using the same chunk/file structure as source
        dst_path = new_meta.root / DEFAULT_DATA_PATH.format(chunk_index=chunk_idx, file_index=file_idx)
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        _write_parquet(df, dst_path, new_meta)

    _copy_episodes_metadata_and_stats(dataset, new_meta)


def _copy_videos(
    src_dataset: LeRobotDataset,
    dst_meta: LeRobotDatasetMetadata,
    exclude_keys: list[str] | None = None,
) -> None:
    """Copy video files, optionally excluding certain keys."""
    if exclude_keys is None:
        exclude_keys = []

    for video_key in src_dataset.meta.video_keys:
        if video_key in exclude_keys:
            continue

        video_files = set()
        for ep_idx in range(len(src_dataset.meta.episodes)):
            try:
                video_files.add(src_dataset.meta.get_video_file_path(ep_idx, video_key))
            except KeyError:
                continue

        for src_path in tqdm(sorted(video_files), desc=f"Copying {video_key} videos"):
            dst_path = dst_meta.root / src_path
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src_dataset.root / src_path, dst_path)


def _copy_episodes_metadata_and_stats(
    src_dataset: LeRobotDataset,
    dst_meta: LeRobotDatasetMetadata,
) -> None:
    """Copy episodes metadata and recalculate stats."""
    if src_dataset.meta.tasks is not None:
        write_tasks(src_dataset.meta.tasks, dst_meta.root)
        dst_meta.tasks = src_dataset.meta.tasks.copy()

    episodes_dir = src_dataset.root / "meta/episodes"
    dst_episodes_dir = dst_meta.root / "meta/episodes"
    if episodes_dir.exists():
        shutil.copytree(episodes_dir, dst_episodes_dir, dirs_exist_ok=True)

    dst_meta.info.update(
        {
            "total_episodes": src_dataset.meta.total_episodes,
            "total_frames": src_dataset.meta.total_frames,
            "total_tasks": src_dataset.meta.total_tasks,
            "splits": src_dataset.meta.info.get("splits", {"train": f"0:{src_dataset.meta.total_episodes}"}),
        }
    )

    if dst_meta.video_keys and src_dataset.meta.video_keys:
        for key in dst_meta.video_keys:
            if key in src_dataset.meta.features:
                dst_meta.info["features"][key]["info"] = src_dataset.meta.info["features"][key].get(
                    "info", {}
                )

    write_info(dst_meta.info, dst_meta.root)

    if set(dst_meta.features.keys()) != set(src_dataset.meta.features.keys()):
        logging.info("Recalculating dataset statistics...")
        if src_dataset.meta.stats:
            new_stats = {}
            for key in dst_meta.features:
                if key in src_dataset.meta.stats:
                    new_stats[key] = src_dataset.meta.stats[key]
            write_stats(new_stats, dst_meta.root)
    else:
        if src_dataset.meta.stats:
            write_stats(src_dataset.meta.stats, dst_meta.root)
