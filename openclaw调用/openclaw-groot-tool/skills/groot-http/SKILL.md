---
name: groot-http
summary: Use local GROOT robot tools to start, monitor, and stop block-grasp tasks.
---

# GROOT HTTP Skill

Use this skill when the user asks the robot to execute a physical manipulation task with the local GROOT policy.

## Tools

- `groot_run`: start a new task. Always provide `task`. Prefer English task text unless the user explicitly wants Chinese.
- `groot_job_status`: inspect a running job and read recent logs.
- `groot_job_stop`: stop a running job when the user asks to cancel, reset, or halt execution.

## Workflow

1. If the user asks to start a real robot task, call `groot_run`.
2. Summarize the returned `job_id` and mention the task text actually sent.
3. If the user asks whether it is still running, call `groot_job_status`.
4. If the user asks to stop, call `groot_job_stop`.

## Guidance

- Prefer task prompts like `Pick up the block with the GROOT policy`.
- If the task is safety-sensitive or ambiguous, clarify before starting.
- Treat any returned `log_tail` as the primary debugging surface.
