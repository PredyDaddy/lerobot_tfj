# pi_rtc

This directory is a structured copy of the document and script outputs produced under
`/data/tfj/lerobot_tfj/tfj_envs/pi_trt`, reorganized for the RTC-focused workflow.

Layout:

- `docs/`
  - copied document-style files only
  - includes top-level markdown docs and `docs/results/*` report files
  - keeps the original result directory names for traceability
- `scripts/`
  - copied executable/source scripts from the original `pi_trt/scripts`
  - excludes `__pycache__`
- `tests/`
  - copied local validation tests created for the RTC path

Filtering rules used for this copy:

- `docs/`: only `*.md`, `*.json`, `*.txt`
- `scripts/`: only `*.py`, `*.sh`
- `tests/`: only `*.py`, `*.sh`
- excluded caches and binary/runtime artifacts such as `__pycache__`, TensorRT engines,
  ONNX blobs, and other non-document artifacts

Source directory remains unchanged:

- `/data/tfj/lerobot_tfj/tfj_envs/pi_trt`

This copy exists to keep RTC-facing materials under a dedicated path without breaking the
existing runnable setup in `pi_trt`.
