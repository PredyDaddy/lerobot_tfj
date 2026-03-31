const PLUGIN_ID = "openclaw-groot-tool";

function getPluginConfig(api) {
  const config = api?.config?.plugins?.entries?.[PLUGIN_ID]?.config ?? {};
  const baseUrl = String(config.baseUrl ?? "http://127.0.0.1:8765").replace(/\/$/, "");
  const requestTimeoutMs = Math.max(Number(config.requestTimeoutMs ?? 180000), 180000);
  const defaultDisplayData = Boolean(config.defaultDisplayData ?? true);
  return { baseUrl, requestTimeoutMs, defaultDisplayData };
}

async function callJson(api, path, method = "GET", payload) {
  const cfg = getPluginConfig(api);
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), cfg.requestTimeoutMs);
  try {
    const res = await fetch(`${cfg.baseUrl}${path}`, {
      method,
      headers: { "Content-Type": "application/json" },
      body: payload ? JSON.stringify(payload) : undefined,
      signal: controller.signal,
    });

    const text = await res.text();
    let json;
    try {
      json = text ? JSON.parse(text) : {};
    } catch {
      json = { raw: text };
    }

    if (!res.ok) {
      throw new Error(`HTTP ${res.status}: ${JSON.stringify(json)}`);
    }

    return json;
  } finally {
    clearTimeout(timeout);
  }
}

function asText(payload) {
  return {
    content: [
      {
        type: "text",
        text: JSON.stringify(payload, null, 2),
      },
    ],
  };
}

function asCommandText(payload) {
  return {
    text: JSON.stringify(payload, null, 2),
  };
}

function usageText(text) {
  return asCommandText({ ok: false, usage: text });
}

const ARM_DIRECTION_ALIASES = new Map([
  ["up", "up"],
  ["上", "up"],
  ["向上", "up"],
  ["down", "down"],
  ["下", "down"],
  ["向下", "down"],
  ["left", "left"],
  ["左", "left"],
  ["向左", "left"],
  ["right", "right"],
  ["右", "right"],
  ["向右", "right"],
  ["forward", "forward"],
  ["front", "forward"],
  ["前", "forward"],
  ["向前", "forward"],
  ["back", "back"],
  ["backward", "back"],
  ["后", "back"],
  ["向后", "back"],
  ["open", "open"],
  ["打开", "open"],
  ["开", "open"],
  ["open_gripper", "open"],
  ["close", "close"],
  ["闭合", "close"],
  ["关闭", "close"],
  ["合上", "close"],
  ["关", "close"],
  ["close_gripper", "close"],
  ["home", "home"],
  ["归位", "home"],
  ["回家", "home"],
  ["复位", "home"],
]);

const ARM_GRIPPER_ALIASES = new Map([
  ["open", "open"],
  ["打开", "open"],
  ["开", "open"],
  ["close", "close"],
  ["闭合", "close"],
  ["关闭", "close"],
  ["hold", "hold"],
  ["保持", "hold"],
  ["stop", "hold"],
]);

const ARM_MOVE_DEFAULT_MM = {
  up: 3,
  down: 3,
  left: 5,
  right: 5,
  forward: 5,
  back: 5,
};

const ARM_MOVE_LIMIT_MM = {
  up: 10,
  down: 10,
  left: 20,
  right: 20,
  forward: 20,
  back: 20,
};

function normalizeCompactToken(value) {
  return String(value ?? "")
    .trim()
    .toLowerCase()
    .replace(/\s+/g, "");
}

function normalizeArmDirection(value) {
  const token = normalizeCompactToken(value);
  if (!token) {
    return null;
  }
  return ARM_DIRECTION_ALIASES.get(token) ?? null;
}

function normalizeArmGripper(value) {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }
  const token = normalizeCompactToken(value);
  if (!token) {
    return undefined;
  }
  return ARM_GRIPPER_ALIASES.get(token);
}

function parseFiniteNumber(value, fieldName) {
  if (value === undefined || value === null || value === "") {
    return undefined;
  }
  const number = Number(value);
  if (!Number.isFinite(number)) {
    throw new Error(`${fieldName} 必须是数字`);
  }
  return number;
}

function parseDistanceMmToken(value) {
  const token = String(value ?? "").trim().toLowerCase();
  if (!token) {
    return undefined;
  }
  const normalized = token
    .replace(/毫米/g, "mm")
    .replace(/厘米/g, "cm");
  const match = normalized.match(/^([-+]?\d+(?:\.\d+)?)(mm|cm)?$/);
  if (!match) {
    throw new Error("距离格式不正确，示例：0.5、5mm、0.5cm；裸数字默认按 cm 解释");
  }
  const number = Number(match[1]);
  if (!Number.isFinite(number)) {
    throw new Error("距离必须是数字");
  }
  return match[2] === "mm" ? number : number * 10;
}

function normalizeArmMovePayload(input) {
  if (!input || typeof input !== "object" || Array.isArray(input)) {
    throw new Error("arm_move 参数必须是对象");
  }

  const direction = normalizeArmDirection(input.direction ?? input.action ?? input.command ?? input.move);
  if (!direction) {
    throw new Error("direction 必须是 up/down/left/right/forward/back/open/close/home 或中文同义词");
  }

  const normalized = {
    direction,
  };

  if (input.exact !== undefined) {
    normalized.exact = Boolean(input.exact);
  }

  if (["open", "close", "home"].includes(direction)) {
    return normalized;
  }

  let distanceMm = parseFiniteNumber(input.distance_mm ?? input.mm, "distance_mm");
  if (distanceMm === undefined && input.cm !== undefined) {
    distanceMm = Number(input.cm) * 10;
  }
  if (distanceMm === undefined) {
    distanceMm = ARM_MOVE_DEFAULT_MM[direction];
  }
  if (!Number.isFinite(distanceMm) || distanceMm <= 0) {
    throw new Error("distance_mm 必须大于 0");
  }

  const maxDistanceMm = ARM_MOVE_LIMIT_MM[direction];
  if (distanceMm > maxDistanceMm) {
    throw new Error(`${direction} 单次最大步长为 ${maxDistanceMm}mm`);
  }

  normalized.distance_mm = Number(distanceMm.toFixed(3));
  normalized.cm = Number((distanceMm / 10).toFixed(4));
  return normalized;
}

function normalizeArmJogPayload(input) {
  if (!input || typeof input !== "object" || Array.isArray(input)) {
    throw new Error("arm_jog 参数必须是对象");
  }

  const dxMm = parseFiniteNumber(input.dx_mm ?? input.dx, "dx_mm") ?? 0;
  const dyMm = parseFiniteNumber(input.dy_mm ?? input.dy, "dy_mm") ?? 0;
  const dzMm = parseFiniteNumber(input.dz_mm ?? input.dz, "dz_mm") ?? 0;
  const gripper = normalizeArmGripper(input.gripper);

  if (Math.abs(dxMm) > 20) {
    throw new Error("dx_mm 绝对值不能超过 20");
  }
  if (Math.abs(dyMm) > 20) {
    throw new Error("dy_mm 绝对值不能超过 20");
  }
  if (Math.abs(dzMm) > 10) {
    throw new Error("dz_mm 绝对值不能超过 10");
  }
  if (dxMm === 0 && dyMm === 0 && dzMm === 0 && gripper === undefined) {
    throw new Error("至少要提供一个非零位移或 gripper");
  }
  if (input.gripper !== undefined && gripper === undefined) {
    throw new Error("gripper 仅支持 open/close/hold 或数字");
  }

  return {
    dx_mm: Number(dxMm.toFixed(3)),
    dy_mm: Number(dyMm.toFixed(3)),
    dz_mm: Number(dzMm.toFixed(3)),
    ...(gripper !== undefined ? { gripper } : {}),
  };
}

function parseArmMoveArgs(rawArgs) {
  const args = String(rawArgs ?? "").trim();
  const usage =
    '/arm_move <方向> [距离，裸数字默认 cm] 或 /arm_move {"direction":"up","cm":0.5}；方向支持 上/下/左/右/前/后/打开/闭合/home';

  if (!args) {
    return { ok: false, usage };
  }

  if (args.startsWith("{")) {
    try {
      const parsed = JSON.parse(args);
      return { ok: true, payload: normalizeArmMovePayload(parsed) };
    } catch (err) {
      return { ok: false, usage: `/arm_move 参数解析失败: ${err?.message ?? String(err)}` };
    }
  }

  const [rawDirection, rawDistance] = args.split(/\s+/, 2);
  try {
    const direction = normalizeArmDirection(rawDirection);
    if (!direction) {
      return { ok: false, usage };
    }
    const payload = { direction };
    if (rawDistance !== undefined && !["open", "close", "home"].includes(direction)) {
      payload.distance_mm = parseDistanceMmToken(rawDistance);
    }
    return { ok: true, payload: normalizeArmMovePayload(payload) };
  } catch (err) {
    return { ok: false, usage: `/arm_move 参数解析失败: ${err?.message ?? String(err)}` };
  }
}

function parseArmJogArgs(rawArgs) {
  const args = String(rawArgs ?? "").trim();
  const usage = '/arm_jog {"dx_mm":5,"dy_mm":0,"dz_mm":0,"gripper":"hold"}';
  if (!args) {
    return { ok: false, usage };
  }
  if (!args.startsWith("{")) {
    return { ok: false, usage };
  }
  try {
    const parsed = JSON.parse(args);
    return { ok: true, payload: normalizeArmJogPayload(parsed) };
  } catch (err) {
    return { ok: false, usage: `/arm_jog 参数解析失败: ${err?.message ?? String(err)}` };
  }
}

function formatArmStateLine(state) {
  if (!state || typeof state !== "object" || Array.isArray(state)) {
    return null;
  }
  const jointNames = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"];
  const jointTokens = jointNames
    .filter((name) => state[name] !== undefined)
    .map((name) => {
      const value = Number(state[name]);
      return Number.isFinite(value) ? `${name}=${value.toFixed(1)}` : null;
    })
    .filter(Boolean);
  return jointTokens.length > 0 ? `state=${jointTokens.join(" ")}` : null;
}

function formatTcpLine(payload) {
  const tcp = payload?.tcp_xyz ?? payload?.status?.tcp_xyz;
  if (!Array.isArray(tcp) || tcp.length < 3) {
    return null;
  }
  const values = tcp
    .slice(0, 3)
    .map((value) => Number(value))
    .filter((value) => Number.isFinite(value));
  if (values.length !== 3) {
    return null;
  }
  return `tcp_mm=${values.map((value) => (value * 1000).toFixed(1)).join(",")}`;
}

function asArmCommandReply(payload, fallbackText) {
  if (!payload?.ok) {
    return asCommandText(payload);
  }

  const lines = [];
  const message = typeof payload.message === "string" ? payload.message.trim() : "";
  lines.push(message || fallbackText);

  const stateLine = formatArmStateLine(payload.state ?? payload.arm_state ?? payload.status?.state);
  if (stateLine) {
    lines.push(stateLine);
  }

  const tcpLine = formatTcpLine(payload);
  if (tcpLine) {
    lines.push(tcpLine);
  }

  return {
    text: lines.join("\n"),
  };
}

function asSnapshotCommandReply(payload) {
  const snapshot = payload?.snapshot ?? {};
  const mediaUrl =
    typeof snapshot.media_url === "string" && snapshot.media_url
      ? snapshot.media_url
      : typeof snapshot.path === "string" && snapshot.path
        ? snapshot.path
        : undefined;

  if (!payload?.ok) {
    return asCommandText(payload);
  }

  const lines = [
    `camera=${snapshot.camera ?? "unknown"}`,
    `camera_index=${snapshot.camera_index ?? "unknown"}`,
    `saved=${snapshot.path ?? "unknown"}`,
    "下一步可以直接发：/groot_run 把那个方块抓起来",
  ];

  return {
    text: lines.join("\n"),
    ...(mediaUrl ? { mediaUrl } : {}),
  };
}

function asDescribeCommandReply(payload) {
  const snapshot = payload?.snapshot ?? {};
  const vision = payload?.vision ?? {};
  const mediaUrl =
    typeof snapshot.media_url === "string" && snapshot.media_url
      ? snapshot.media_url
      : typeof snapshot.path === "string" && snapshot.path
        ? snapshot.path
        : undefined;

  if (!payload?.ok) {
    return asCommandText(payload);
  }

  const lines = [
    String(vision.answer ?? "").trim() || "没有拿到视觉描述结果。",
    "",
    `vision_backend=${vision.backend ?? "unknown"}`,
    `vision_model=${vision.model ?? "unknown"}`,
    `camera=${snapshot.camera ?? "unknown"}`,
    `camera_index=${snapshot.camera_index ?? "unknown"}`,
    `saved=${snapshot.path ?? "unknown"}`,
    "下一步可以直接发：/groot_run 把那个方块抓起来",
  ];

  return {
    text: lines.join("\n"),
    ...(mediaUrl ? { mediaUrl } : {}),
  };
}

function parseRunArgs(rawArgs) {
  const args = String(rawArgs ?? "").trim();
  if (!args) {
    return { ok: false, usage: "/groot_run <task text | json payload>" };
  }

  if (args.startsWith("{")) {
    try {
      const parsed = JSON.parse(args);
      if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
        return { ok: false, usage: "/groot_run 参数 JSON 必须是对象，例如 {\"task\":\"pick up the red block\"}" };
      }
      return { ok: true, payload: parsed };
    } catch (err) {
      return { ok: false, usage: `/groot_run JSON 解析失败: ${err?.message ?? String(err)}` };
    }
  }

  return { ok: true, payload: { task: args } };
}

function parseSnapshotArgs(rawArgs) {
  const args = String(rawArgs ?? "").trim();
  if (!args) {
    return { ok: true, payload: {} };
  }

  if (args.startsWith("{")) {
    try {
      const parsed = JSON.parse(args);
      if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
        return { ok: false, usage: "/groot_snapshot 参数 JSON 必须是对象，例如 {\"camera\":\"top\"}" };
      }
      return { ok: true, payload: parsed };
    } catch (err) {
      return { ok: false, usage: `/groot_snapshot JSON 解析失败: ${err?.message ?? String(err)}` };
    }
  }

  return { ok: true, payload: { camera: args } };
}

function isCameraAliasToken(value) {
  const normalized = String(value ?? "").trim().toLowerCase();
  return [
    "top",
    "wrist",
    "顶部",
    "顶视",
    "顶视角",
    "俯视",
    "俯视角",
    "腕",
    "腕部",
    "手腕",
    "手腕相机",
    "夹爪",
    "夹爪相机",
  ].includes(normalized);
}

function parseDescribeArgs(rawArgs) {
  const args = String(rawArgs ?? "").trim();
  if (!args) {
    return { ok: true, payload: {} };
  }

  if (args.startsWith("{")) {
    try {
      const parsed = JSON.parse(args);
      if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
        return { ok: false, usage: "/groot_describe 参数 JSON 必须是对象，例如 {\"camera\":\"top\",\"prompt\":\"这个画面里有什么\"}" };
      }
      return { ok: true, payload: parsed };
    } catch (err) {
      return { ok: false, usage: `/groot_describe JSON 解析失败: ${err?.message ?? String(err)}` };
    }
  }

  const [firstToken, ...restTokens] = args.split(/\s+/);
  if (isCameraAliasToken(firstToken)) {
    return {
      ok: true,
      payload: {
        camera: firstToken,
        ...(restTokens.length > 0 ? { prompt: restTokens.join(" ") } : {}),
      },
    };
  }

  return { ok: true, payload: { prompt: args } };
}

function registerChatCommands(api) {
  if (typeof api.registerCommand !== "function") {
    return;
  }

  api.registerCommand(
    {
      name: "groot_run",
      description: "Run GROOT directly from chat command, bypassing LLM planning.",
      acceptsArgs: true,
      async handler(ctx) {
        const parsed = parseRunArgs(ctx.args);
        if (!parsed.ok) {
          return usageText(parsed.usage);
        }

        const cfg = getPluginConfig(api);
        const payload = {
          ...parsed.payload,
          display_data: parsed.payload.display_data ?? cfg.defaultDisplayData,
        };
        const result = await callJson(api, "/run", "POST", payload);
        return asCommandText(result);
      },
    }
  );

  api.registerCommand(
    {
      name: "groot_status",
      description: "Query job status by job id.",
      acceptsArgs: true,
      async handler(ctx) {
        const jobId = String(ctx.args ?? "").trim();
        if (!jobId) {
          return usageText("/groot_status <job_id>");
        }
        const result = await callJson(api, `/jobs/${encodeURIComponent(jobId)}`);
        return asCommandText(result);
      },
    }
  );

  api.registerCommand(
    {
      name: "groot_snapshot",
      description: "Capture one frame from the top or wrist camera and send it back to chat.",
      acceptsArgs: true,
      async handler(ctx) {
        const parsed = parseSnapshotArgs(ctx.args);
        if (!parsed.ok) {
          return usageText(parsed.usage);
        }
        const result = await callJson(api, "/snapshot", "POST", parsed.payload);
        return asSnapshotCommandReply(result);
      },
    }
  );

  api.registerCommand(
    {
      name: "groot_describe",
      description: "Capture a camera frame, ask a vision model, and send back the answer plus image.",
      acceptsArgs: true,
      async handler(ctx) {
        const parsed = parseDescribeArgs(ctx.args);
        if (!parsed.ok) {
          return usageText(parsed.usage);
        }
        const payload = {
          ...parsed.payload,
          vision_backend: parsed.payload.vision_backend ?? "auto",
          model: parsed.payload.model ?? "gpt-5.4",
          detail: parsed.payload.detail ?? "high",
          max_output_tokens: parsed.payload.max_output_tokens ?? 240,
        };
        const result = await callJson(api, "/describe", "POST", payload);
        return asDescribeCommandReply(result);
      },
    }
  );

  api.registerCommand(
    {
      name: "groot_stop",
      description: "Stop a running job by job id.",
      acceptsArgs: true,
      async handler(ctx) {
        const jobId = String(ctx.args ?? "").trim();
        if (!jobId) {
          return usageText("/groot_stop <job_id>");
        }
        const result = await callJson(api, `/jobs/${encodeURIComponent(jobId)}/stop`, "POST", {});
        return asCommandText(result);
      },
    }
  );

  api.registerCommand(
    {
      name: "arm_move",
      description: "Move the SO101 arm one safe step: 上/下/左/右/前/后/打开/闭合/home.",
      acceptsArgs: true,
      async handler(ctx) {
        const parsed = parseArmMoveArgs(ctx.args);
        if (!parsed.ok) {
          return usageText(parsed.usage);
        }
        const result = await callJson(api, "/arm/move", "POST", parsed.payload);
        return asArmCommandReply(result, "机械臂动作已发送。");
      },
    }
  );

  api.registerCommand(
    {
      name: "arm_jog",
      description: "Jog the SO101 arm in Cartesian millimeters via JSON payload.",
      acceptsArgs: true,
      async handler(ctx) {
        const parsed = parseArmJogArgs(ctx.args);
        if (!parsed.ok) {
          return usageText(parsed.usage);
        }
        const result = await callJson(api, "/arm/jog", "POST", parsed.payload);
        return asArmCommandReply(result, "机械臂 jog 指令已发送。");
      },
    }
  );
}

function runSchema() {
  return {
    type: "object",
    additionalProperties: false,
    properties: {
      backend: {
        type: "string",
        enum: ["groot", "smolvla", "pi05", "pi0.5", "act", "act_distill", "policy_record", "policy"],
        description: "Backend selector. Defaults to groot and routes through the guarded SO101 pick-place runner.",
      },
      job_id: {
        type: "string",
        description: "Optional caller-supplied job id. It must be unique; duplicate reuse is rejected to preserve session ownership.",
      },
      task: { type: "string", description: "Canonical natural-language task text. Legacy task-only requests remain valid." },
      task_text: { type: "string", description: "Compatibility alias for task." },
      intent_json: {
        oneOf: [
          { type: "string", description: "Structured intent encoded as a JSON string." },
          { type: "object", additionalProperties: true, description: "Structured intent object; server serializes it." },
        ],
      },
      intent: {
        oneOf: [
          { type: "string", description: "Compatibility alias for intent_json." },
          { type: "object", additionalProperties: true, description: "Compatibility alias for intent_json." },
        ],
      },
      task_intent_json: { type: "string", description: "Legacy alias for intent_json." },
      safety_profile: { type: "string", description: "Named safety profile passed through to the runner contract." },
      events_jsonl_path: { type: "string", description: "Canonical path for step events JSONL output." },
      events_path: { type: "string", description: "Legacy alias for events_jsonl_path." },
      clear_dataset_root: { type: "boolean", description: "When true, explicitly delete dataset_root before launching." },
      leader_port: { type: "string" },
      policy_path: { type: "string" },
      policy_device: { type: "string" },
      robot_port: { type: "string" },
      robot_id: { type: "string" },
      robot_calib_dir: { type: "string" },
      top_camera_index: { type: "integer" },
      wrist_camera_index: { type: "integer" },
      camera_width: { type: "integer" },
      camera_height: { type: "integer" },
      camera_fps: { type: "integer" },
      leader_id: { type: "string" },
      leader_calib_dir: { type: "string" },
      dataset_repo_id: { type: "string" },
      dataset_root: { type: "string" },
      num_episodes: { type: "integer", minimum: 1 },
      episode_time_s: { type: "number", minimum: 1 },
      reset_time_s: { type: "number", minimum: 0 },
      display_data: { type: "boolean" },
      python_bin: { type: "string" },
    },
  };
}

function snapshotSchema() {
  return {
    type: "object",
    additionalProperties: false,
    properties: {
      camera: {
        type: "string",
        description: "Camera alias. Supported values include top, wrist, 顶视, 手腕.",
      },
      camera_index: {
        type: "integer",
        description: "Explicit camera index override. Takes priority over camera alias.",
      },
      top_camera_index: { type: "integer" },
      wrist_camera_index: { type: "integer" },
      camera_width: { type: "integer" },
      camera_height: { type: "integer" },
      camera_fps: { type: "integer" },
      warmup_frames: { type: "integer", minimum: 0 },
      output_dir: { type: "string" },
      output_path: { type: "string" },
    },
  };
}

function describeSchema() {
  return {
    type: "object",
    additionalProperties: false,
    properties: {
      camera: {
        type: "string",
        description: "Camera alias. Supported values include top, wrist, 顶视, 手腕.",
      },
      camera_index: {
        type: "integer",
        description: "Explicit camera index override. Takes priority over camera alias.",
      },
      top_camera_index: { type: "integer" },
      wrist_camera_index: { type: "integer" },
      camera_width: { type: "integer" },
      camera_height: { type: "integer" },
      camera_fps: { type: "integer" },
      warmup_frames: { type: "integer", minimum: 0 },
      prompt: { type: "string", description: "Question or instruction for the vision model." },
      question: { type: "string", description: "Compatibility alias for prompt." },
      model: { type: "string", description: "Codex model override for /describe." },
      vision_backend: {
        type: "string",
        enum: ["auto", "heuristic", "codex", "openai"],
        description: "Vision backend override. Default is auto so the server can choose the most stable available fallback for the current frame.",
      },
      max_output_tokens: { type: "integer", minimum: 1 },
      detail: { type: "string", enum: ["low", "high", "auto"] },
      timeout_s: { type: "integer", minimum: 1 },
      output_dir: { type: "string" },
      output_path: { type: "string" },
    },
  };
}

function armMoveSchema() {
  return {
    type: "object",
    additionalProperties: false,
    properties: {
      direction: {
        type: "string",
        description: "Direction or action: up/down/left/right/forward/back/open/close/home.",
      },
      distance_mm: {
        type: "number",
        description: "Optional distance in millimeters. Defaults to a small safe step.",
      },
      cm: {
        type: "number",
        description: "Compatibility alias for distance in centimeters.",
      },
      exact: {
        type: "boolean",
        description: "Optional server hint to request exact Cartesian stepping when supported.",
      },
    },
    required: ["direction"],
  };
}

function armJogSchema() {
  return {
    type: "object",
    additionalProperties: false,
    properties: {
      dx_mm: { type: "number", description: "Cartesian X jog in millimeters." },
      dy_mm: { type: "number", description: "Cartesian Y jog in millimeters." },
      dz_mm: { type: "number", description: "Cartesian Z jog in millimeters." },
      gripper: {
        oneOf: [
          { type: "string", enum: ["open", "close", "hold"] },
          { type: "number" },
        ],
        description: "Optional gripper action or numeric override.",
      },
    },
  };
}

export const id = PLUGIN_ID;

export default function register(api) {
  registerChatCommands(api);

  api.registerTool(
    {
      name: "groot_run",
      description:
        "Start a guarded SO101 manipulation run on the local LeRobot server. Task-only payloads stay compatible; use backend, intent_json, safety_profile, and events_jsonl_path for the newer contract.",
      parameters: runSchema(),
      async execute(_id, params) {
        const cfg = getPluginConfig(api);
        const payload = {
          ...params,
          display_data: params.display_data ?? cfg.defaultDisplayData,
        };
        const result = await callJson(api, "/run", "POST", payload);
        return asText(result);
      },
    },
    { optional: true },
  );

  api.registerTool(
    {
      name: "groot_job_status",
      description: "Get the current status and recent logs for a GROOT run.",
      parameters: {
        type: "object",
        additionalProperties: false,
        properties: {
          job_id: { type: "string" },
        },
        required: ["job_id"],
      },
      async execute(_id, params) {
        const result = await callJson(api, `/jobs/${encodeURIComponent(params.job_id)}`);
        return asText(result);
      },
    },
    { optional: true },
  );

  api.registerTool(
    {
      name: "groot_job_stop",
      description: "Stop a running GROOT job by job id.",
      parameters: {
        type: "object",
        additionalProperties: false,
        properties: {
          job_id: { type: "string" },
        },
        required: ["job_id"],
      },
      async execute(_id, params) {
        const result = await callJson(api, `/jobs/${encodeURIComponent(params.job_id)}/stop`, "POST", {});
        return asText(result);
      },
    },
    { optional: true },
  );

  api.registerTool(
    {
      name: "groot_snapshot",
      description: "Capture a single frame from the SO101 top or wrist camera.",
      parameters: snapshotSchema(),
      async execute(_id, params) {
        const result = await callJson(api, "/snapshot", "POST", params);
        return asText(result);
      },
    },
    { optional: true },
  );

  api.registerTool(
    {
      name: "groot_describe",
      description: "Capture a camera frame and ask the local Codex vision backend what the robot sees.",
      parameters: describeSchema(),
      async execute(_id, params) {
        const result = await callJson(api, "/describe", "POST", params);
        return asText(result);
      },
    },
    { optional: true },
  );

  api.registerTool(
    {
      name: "arm_move",
      description: "Move the SO101 arm one safe step in Cartesian direction or trigger open/close/home.",
      parameters: armMoveSchema(),
      async execute(_id, params) {
        const payload = normalizeArmMovePayload(params);
        const result = await callJson(api, "/arm/move", "POST", payload);
        return asText(result);
      },
    },
    { optional: true },
  );

  api.registerTool(
    {
      name: "arm_jog",
      description: "Jog the SO101 arm by Cartesian millimeter deltas.",
      parameters: armJogSchema(),
      async execute(_id, params) {
        const payload = normalizeArmJogPayload(params);
        const result = await callJson(api, "/arm/jog", "POST", payload);
        return asText(result);
      },
    },
    { optional: true },
  );
}
