# SmolVLA RL Docs

This directory contains the detailed Chinese documentation for the current SmolVLA hybrid RL work.

Files:

- `smolvla_rl_architecture_and_integration_20260315_zh.md`
  - Explains the original SmolVLA action generation mechanism.
  - Explains how RL is connected into SmolVLA in this repository.
  - Explains the exact role of `value_head`, `prefix context`, `compute_fm_score`, replay buffer, and collector.
  - Explains what this implementation is and is not.

- `smolvla_rl_training_and_operations_20260315_zh.md`
  - Explains the practical staged workflow.
  - Explains the successful offline training run on the trimmed SO101 dataset.
  - Explains runtime scripts, monitoring, checkpoints, and SO101 policy recording.
  - Explains the main engineering limitations and what would be needed for real robot online RL.

- `smolvla_rl_final_training_and_so101_report_20260316_zh.md`
  - Records the completed 2026-03-16 hybrid RL run.
  - Records the final `005000` checkpoint and the exact end-of-training status.
  - Records the current SO101 on-robot command, wrapper defaults, and the bool parsing fix for `clear_dataset_root`.

- `vla_smolvla_rl_ppt_knowledge_base_20260317_zh.md`
  - A generalized knowledge compendium for downstream PPT generation.
  - Covers VLA, SmolVLA architecture, flow matching, shared prefix representation, hybrid RL integration, PPO/SAC/DDPG comparisons, and deployment pitfalls without relying on local experiment paths.

- `smolvla_rl_complete_knowledge_for_ppt_20260317_zh.md`
  - A generalized, path-light, project-agnostic knowledge brief for downstream PPT generation.
  - Covers SmolVLA architecture, flow matching, hybrid RL, RL fundamentals, deployment logic, engineering pitfalls, and presentation structure.

- `reviews/`
  - Reviewer notes generated after the main document set was written.
