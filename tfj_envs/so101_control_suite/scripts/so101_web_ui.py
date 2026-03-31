#!/usr/bin/env python3
from __future__ import annotations

import argparse
import dataclasses
import json
import threading
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from so101_sdk import SO101SDK, SO101SDKConfig


HTML_PAGE = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>SO101 控制台</title>
  <style>
    :root {
      --bg: #f3f5f7;
      --panel: #ffffff;
      --text: #17212b;
      --sub: #4f6473;
      --line: #d9e1e8;
      --ok: #0f8f4f;
      --warn: #b95700;
      --bad: #c62828;
      --btn: #0d6efd;
      --btn-text: #ffffff;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: linear-gradient(160deg, #eef3f7 0%, #f8fbfd 100%);
      color: var(--text);
      font-family: "Noto Sans SC", "PingFang SC", "Microsoft YaHei", sans-serif;
    }
    .wrap {
      max-width: 1120px;
      margin: 24px auto 32px;
      padding: 0 16px;
      display: grid;
      gap: 14px;
      grid-template-columns: 1.15fr 0.85fr;
    }
    .card {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 14px;
      box-shadow: 0 2px 14px rgba(30, 52, 74, 0.05);
    }
    h1 { font-size: 22px; margin: 0 0 10px; }
    h2 { font-size: 16px; margin: 0 0 10px; }
    .row {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
      margin-bottom: 10px;
    }
    .field {
      display: flex;
      gap: 6px;
      align-items: center;
    }
    .field label { font-size: 13px; color: var(--sub); }
    input[type=text], input[type=number], select {
      padding: 7px 8px;
      border: 1px solid var(--line);
      border-radius: 8px;
      min-width: 96px;
      font-size: 14px;
      background: #fff;
    }
    button {
      border: 0;
      border-radius: 10px;
      padding: 8px 12px;
      font-size: 14px;
      cursor: pointer;
      background: var(--btn);
      color: var(--btn-text);
      transition: transform .04s ease, opacity .12s ease;
      min-width: 84px;
    }
    button:hover { opacity: 0.92; }
    button:active { transform: translateY(1px); }
    .ghost { background: #e9eef3; color: #1a2d3a; }
    .safe { background: #0f8f4f; }
    .warn { background: #d97706; }
    .danger { background: #c62828; }
    .status {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      font-weight: 600;
      padding: 7px 10px;
      border-radius: 9px;
      border: 1px solid var(--line);
      background: #f9fbfd;
    }
    .dot {
      width: 10px;
      height: 10px;
      border-radius: 50%;
      background: var(--bad);
    }
    .dot.ok { background: var(--ok); }
    .grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(80px, 1fr));
      gap: 8px;
      max-width: 420px;
    }
    .grid button { width: 100%; }
    .wide { min-width: 120px; }
    .state-list {
      display: grid;
      grid-template-columns: repeat(2, minmax(120px, 1fr));
      gap: 6px 12px;
      font-size: 14px;
    }
    .state-item {
      display: flex;
      justify-content: space-between;
      border-bottom: 1px dashed #e7edf3;
      padding-bottom: 4px;
    }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
    #log {
      width: 100%;
      height: 260px;
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 8px;
      overflow: auto;
      background: #f7fafc;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
      white-space: pre-wrap;
    }
    .small { font-size: 12px; color: var(--sub); }
    @media (max-width: 980px) {
      .wrap { grid-template-columns: 1fr; }
      .grid { max-width: 100%; }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h1>SO101 Web UI</h1>
      <div class="row">
        <span id="status" class="status"><span id="dot" class="dot"></span><span id="statusText">未连接</span></span>
        <span class="small">方向映射: <span id="frameMap" class="mono">-</span></span>
      </div>

      <div class="row">
        <div class="field"><label>串口</label><input id="robotPort" type="text" value="/dev/ttyACM0" /></div>
        <div class="field">
          <label>模式</label>
          <select id="controlMode">
            <option value="ik" selected>ik</option>
            <option value="joint">joint</option>
          </select>
        </div>
        <div class="field"><label><input id="dryRun" type="checkbox" /> dry-run</label></div>
        <button id="connectBtn" class="safe">连接</button>
        <button id="disconnectBtn" class="danger">断开</button>
      </div>
      <div class="row">
        <div class="field"><label>TCP X(mm)</label><input id="tcpXmm" type="number" value="0" step="1" /></div>
        <div class="field"><label>TCP Y(mm)</label><input id="tcpYmm" type="number" value="0" step="1" /></div>
        <div class="field"><label>TCP Z(mm)</label><input id="tcpZmm" type="number" value="0" step="1" /></div>
        <span class="small">按真实夹持点去修这个偏移，不是按腕部中心。</span>
      </div>

      <h2>位移控制</h2>
      <div class="row">
        <div class="field"><label>步长(cm)</label><input id="stepCm" type="number" value="1" step="0.1" min="0.1" /></div>
        <button id="homeBtn" class="warn">回 Home</button>
      </div>
      <div class="grid">
        <div></div><button data-dir="up">上移</button><div></div>
        <button data-dir="left">左移</button><button data-dir="down">下移</button><button data-dir="right">右移</button>
        <div></div><button data-dir="forward">前移</button><div></div>
        <div></div><button data-dir="back">后移</button><div></div>
      </div>

      <h2 style="margin-top:14px;">夹爪与校准</h2>
      <div class="row">
        <button id="openBtn" class="safe wide">开爪</button>
        <button id="closeBtn" class="danger wide">闭爪</button>
        <button id="calibrateBtn" class="ghost wide">方向校准</button>
        <button id="frameSaveBtn" class="ghost wide">保存方向映射</button>
      </div>

      <h2 style="margin-top:14px;">自定义命令</h2>
      <div class="row">
        <input id="rawCmd" type="text" style="flex:1; min-width:220px" placeholder="例如: up 1; open; obs" />
        <button id="rawCmdBtn" class="ghost">执行</button>
      </div>
    </div>

    <div class="card">
      <h2>当前关节状态</h2>
      <div id="stateBox" class="state-list mono"></div>
      <div class="row" style="margin-top:8px;">
        <span class="small">TCP xyz: <span id="tcpXyz" class="mono">-</span></span>
      </div>

      <h2 style="margin-top:14px;">日志</h2>
      <div id="log"></div>
      <div class="row" style="margin-top:8px;">
        <button id="refreshBtn" class="ghost">刷新状态</button>
        <span class="small">提示: 推荐先用 1cm 小步测试方向。</span>
      </div>
    </div>
  </div>

  <script>
    const stateBox = document.getElementById('stateBox');
    const logBox = document.getElementById('log');
    const statusText = document.getElementById('statusText');
    const dot = document.getElementById('dot');
    const frameMap = document.getElementById('frameMap');
    const tcpXyz = document.getElementById('tcpXyz');

    function renderState(state) {
      if (!state) {
        stateBox.innerHTML = '<span class="small">未连接</span>';
        return;
      }
      const rows = Object.entries(state).map(([k,v]) => (
        `<div class="state-item"><span>${k}</span><span>${Number(v).toFixed(2)}</span></div>`
      ));
      stateBox.innerHTML = rows.join('');
    }

    function renderLogs(lines) {
      if (!Array.isArray(lines)) return;
      logBox.textContent = lines.join('\\n');
      logBox.scrollTop = logBox.scrollHeight;
    }

    function updateStatus(data) {
      const connected = !!data.connected;
      statusText.textContent = connected ? '已连接' : '未连接';
      dot.className = connected ? 'dot ok' : 'dot';
      frameMap.textContent = data.frame_map || '-';
      tcpXyz.textContent = Array.isArray(data.tcp_xyz) ? data.tcp_xyz.map((v) => Number(v).toFixed(4)).join(', ') : '-';
      renderState(data.state || null);
      renderLogs(data.logs || []);
    }

    async function api(path, method='GET', payload=null) {
      const opts = { method, headers: {} };
      if (payload !== null) {
        opts.headers['Content-Type'] = 'application/json';
        opts.body = JSON.stringify(payload);
      }
      const resp = await fetch(path, opts);
      const data = await resp.json();
      if (!resp.ok || !data.ok) {
        throw new Error(data.error || `HTTP ${resp.status}`);
      }
      return data;
    }

    async function refresh() {
      try {
        const data = await api('/api/status');
        updateStatus(data);
      } catch (err) {
        statusText.textContent = '通信异常';
        dot.className = 'dot';
      }
    }

    document.getElementById('connectBtn').onclick = async () => {
      try {
        const payload = {
          robot_port: document.getElementById('robotPort').value.trim(),
          control_mode: document.getElementById('controlMode').value,
          dry_run: !!document.getElementById('dryRun').checked,
          tcp_offset_xyz: [
            Number(document.getElementById('tcpXmm').value) / 1000.0,
            Number(document.getElementById('tcpYmm').value) / 1000.0,
            Number(document.getElementById('tcpZmm').value) / 1000.0,
          ],
        };
        const data = await api('/api/connect', 'POST', payload);
        updateStatus(data);
      } catch (err) {
        alert('连接失败: ' + err.message);
      }
    };

    document.getElementById('disconnectBtn').onclick = async () => {
      try {
        const data = await api('/api/disconnect', 'POST', {});
        updateStatus(data);
      } catch (err) {
        alert('断开失败: ' + err.message);
      }
    };

    document.querySelectorAll('button[data-dir]').forEach((btn) => {
      btn.onclick = async () => {
        try {
          const cm = Number(document.getElementById('stepCm').value);
          const direction = btn.dataset.dir;
          const data = await api('/api/move', 'POST', { direction, cm });
          updateStatus(data);
        } catch (err) {
          alert('移动失败: ' + err.message);
        }
      };
    });

    document.getElementById('openBtn').onclick = async () => {
      try {
        const data = await api('/api/gripper', 'POST', { action: 'open' });
        updateStatus(data);
      } catch (err) {
        alert('开爪失败: ' + err.message);
      }
    };

    document.getElementById('closeBtn').onclick = async () => {
      try {
        const data = await api('/api/gripper', 'POST', { action: 'close' });
        updateStatus(data);
      } catch (err) {
        alert('闭爪失败: ' + err.message);
      }
    };

    document.getElementById('homeBtn').onclick = async () => {
      try {
        const data = await api('/api/home', 'POST', {});
        updateStatus(data);
      } catch (err) {
        alert('home失败: ' + err.message);
      }
    };

    document.getElementById('calibrateBtn').onclick = async () => {
      try {
        const cm = Number(document.getElementById('stepCm').value) || 1;
        const data = await api('/api/calibrate', 'POST', { probe_cm: cm });
        updateStatus(data);
      } catch (err) {
        alert('校准失败: ' + err.message);
      }
    };

    document.getElementById('frameSaveBtn').onclick = async () => {
      try {
        const data = await api('/api/frame', 'POST', { action: 'save' });
        updateStatus(data);
      } catch (err) {
        alert('保存方向映射失败: ' + err.message);
      }
    };

    document.getElementById('rawCmdBtn').onclick = async () => {
      try {
        const command = document.getElementById('rawCmd').value.trim();
        if (!command) return;
        const data = await api('/api/command', 'POST', { command });
        updateStatus(data);
      } catch (err) {
        alert('命令执行失败: ' + err.message);
      }
    };

    document.getElementById('refreshBtn').onclick = refresh;
    refresh();
    setInterval(refresh, 1200);
  </script>
</body>
</html>
"""


class SO101WebApp:
    def __init__(self, base_config: SO101SDKConfig, max_logs: int = 300):
        self._base_config = base_config
        self._sdk: SO101SDK | None = None
        self._lock = threading.RLock()
        self._logs: list[str] = []
        self._max_logs = max(50, int(max_logs))

    def _now(self) -> str:
        return datetime.now().strftime("%H:%M:%S")

    def _add_log(self, text: str) -> None:
        line = f"[{self._now()}] {text}"
        self._logs.append(line)
        if len(self._logs) > self._max_logs:
            self._logs = self._logs[-self._max_logs :]

    def _make_config(
        self,
        *,
        robot_port: str | None = None,
        control_mode: str | None = None,
        dry_run: bool | None = None,
        tcp_offset_xyz: tuple[float, float, float] | None = None,
    ) -> SO101SDKConfig:
        cfg = dataclasses.replace(self._base_config)
        if robot_port:
            cfg.robot_port = robot_port
        if control_mode in {"joint", "ik"}:
            cfg.control_mode = control_mode
        if dry_run is not None:
            cfg.dry_run = bool(dry_run)
        if tcp_offset_xyz is not None:
            cfg.tcp_offset_xyz = tcp_offset_xyz
        return cfg

    def _connected(self) -> bool:
        return self._sdk is not None and self._sdk.connected

    def _status_locked(self) -> dict[str, Any]:
        connected = self._connected()
        state: dict[str, float] | None = None
        frame_map: str | None = None
        tcp_xyz: tuple[float, float, float] | None = None
        if connected and self._sdk is not None:
            try:
                state = self._sdk.state()
            except Exception as exc:  # noqa: BLE001
                self._add_log(f"读取状态失败: {exc}")
            try:
                frame_map = self._sdk.frame_show()
            except Exception:
                frame_map = None
            try:
                tcp_xyz = self._sdk.tcp_xyz()
            except Exception:
                tcp_xyz = None
        return {
            "ok": True,
            "connected": connected,
            "state": state,
            "frame_map": frame_map,
            "tcp_xyz": list(tcp_xyz) if tcp_xyz is not None else None,
            "logs": list(self._logs),
        }

    def status(self) -> dict[str, Any]:
        with self._lock:
            return self._status_locked()

    def connect(self, payload: dict[str, Any]) -> dict[str, Any]:
        robot_port = str(payload.get("robot_port", "")).strip() or None
        control_mode = str(payload.get("control_mode", "")).strip() or None
        dry_run = payload.get("dry_run")
        raw_tcp_offset_xyz = payload.get("tcp_offset_xyz")
        tcp_offset_xyz: tuple[float, float, float] | None = None
        if raw_tcp_offset_xyz is not None:
            if not isinstance(raw_tcp_offset_xyz, (list, tuple)) or len(raw_tcp_offset_xyz) != 3:
                raise ValueError("tcp_offset_xyz must be a 3-element array.")
            tcp_offset_xyz = tuple(float(value) for value in raw_tcp_offset_xyz)
        with self._lock:
            if self._connected():
                self._add_log("连接请求: 已连接，忽略。")
                return self._status_locked()
            cfg = self._make_config(
                robot_port=robot_port,
                control_mode=control_mode,
                dry_run=dry_run,
                tcp_offset_xyz=tcp_offset_xyz,
            )
            self._sdk = SO101SDK(cfg)
            self._sdk.connect()
            self._add_log(
                f"已连接: port={cfg.robot_port}, mode={cfg.control_mode}, dry_run={'yes' if cfg.dry_run else 'no'}"
            )
            self._add_log(f"TCP offset (m): {cfg.tcp_offset_xyz}")
            return self._status_locked()

    def disconnect(self) -> dict[str, Any]:
        with self._lock:
            if self._sdk is not None:
                self._sdk.disconnect()
                self._add_log("已断开连接。")
                self._sdk = None
            return self._status_locked()

    def _require_sdk(self) -> SO101SDK:
        if not self._connected() or self._sdk is None:
            raise RuntimeError("机械臂未连接，请先点击“连接”。")
        return self._sdk

    def move(self, payload: dict[str, Any]) -> dict[str, Any]:
        direction = str(payload.get("direction", "")).strip().lower()
        cm = float(payload.get("cm", 1.0))
        if direction not in {"up", "down", "left", "right", "forward", "back"}:
            raise ValueError(f"非法方向: {direction}")
        if cm <= 0:
            raise ValueError("步长必须 > 0")
        with self._lock:
            sdk = self._require_sdk()
            sdk.move(direction, cm)
            self._add_log(f"移动: {direction} {cm:g}cm")
            return self._status_locked()

    def gripper(self, payload: dict[str, Any]) -> dict[str, Any]:
        action = str(payload.get("action", "")).strip().lower()
        with self._lock:
            sdk = self._require_sdk()
            if action == "open":
                sdk.open_gripper()
                self._add_log("夹爪: open")
            elif action == "close":
                sdk.close_gripper()
                self._add_log("夹爪: close")
            else:
                raise ValueError(f"非法夹爪动作: {action}")
            return self._status_locked()

    def home(self) -> dict[str, Any]:
        with self._lock:
            sdk = self._require_sdk()
            sdk.home()
            self._add_log("动作: home")
            return self._status_locked()

    def calibrate(self, payload: dict[str, Any]) -> dict[str, Any]:
        probe_cm = float(payload.get("probe_cm", 0.6))
        with self._lock:
            sdk = self._require_sdk()
            sdk.calibrate_directions(probe_cm=probe_cm)
            self._add_log(f"方向校准: probe={probe_cm:g}cm")
            return self._status_locked()

    def frame(self, payload: dict[str, Any]) -> dict[str, Any]:
        action = str(payload.get("action", "")).strip().lower()
        with self._lock:
            sdk = self._require_sdk()
            if action == "save":
                sdk.frame_save()
                self._add_log("方向映射: save")
            elif action == "load":
                sdk.frame_load()
                self._add_log("方向映射: load")
            elif action == "reset":
                sdk.frame_reset()
                self._add_log("方向映射: reset")
            else:
                raise ValueError(f"非法 frame action: {action}")
            return self._status_locked()

    def command(self, payload: dict[str, Any]) -> dict[str, Any]:
        command = str(payload.get("command", "")).strip()
        if not command:
            raise ValueError("command 不能为空")
        with self._lock:
            sdk = self._require_sdk()
            sdk.execute(command)
            self._add_log(f"命令: {command}")
            return self._status_locked()


class SO101RequestHandler(BaseHTTPRequestHandler):
    app: SO101WebApp | None = None

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        # Silence default HTTP logs.
        return

    def _send_json(self, payload: dict[str, Any], status: int = HTTPStatus.OK) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_html(self, html: str, status: int = HTTPStatus.OK) -> None:
        data = html.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _read_json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        if not raw:
            return {}
        return json.loads(raw.decode("utf-8"))

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/":
            self._send_html(HTML_PAGE)
            return
        if self.path == "/api/status":
            assert self.app is not None
            self._send_json(self.app.status())
            return
        self._send_json({"ok": False, "error": f"Not found: {self.path}"}, status=HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        assert self.app is not None
        try:
            payload = self._read_json_body()
            if self.path == "/api/connect":
                self._send_json(self.app.connect(payload))
                return
            if self.path == "/api/disconnect":
                self._send_json(self.app.disconnect())
                return
            if self.path == "/api/move":
                self._send_json(self.app.move(payload))
                return
            if self.path == "/api/gripper":
                self._send_json(self.app.gripper(payload))
                return
            if self.path == "/api/home":
                self._send_json(self.app.home())
                return
            if self.path == "/api/calibrate":
                self._send_json(self.app.calibrate(payload))
                return
            if self.path == "/api/frame":
                self._send_json(self.app.frame(payload))
                return
            if self.path == "/api/command":
                self._send_json(self.app.command(payload))
                return
            self._send_json({"ok": False, "error": f"Not found: {self.path}"}, status=HTTPStatus.NOT_FOUND)
        except Exception as exc:  # noqa: BLE001
            self._send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SO101 web UI server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)

    parser.add_argument("--robot-port", default="/dev/ttyACM0")
    parser.add_argument("--robot-id", default="my_so101")
    parser.add_argument(
        "--calibration-dir",
        default="/home/cqy/.cache/huggingface/lerobot/calibration/robots/so101_follower",
    )
    parser.add_argument("--control-mode", choices=("joint", "ik"), default="ik")
    parser.add_argument("--ik-solver", choices=("placo", "dls"), default="placo")
    parser.add_argument("--max-command-delta-deg", type=float, default=8.0)
    parser.add_argument("--max-relative-target-deg", type=float, default=8.0)
    parser.add_argument("--gripper-max-relative-target", type=float, default=40.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-keep", type=int, default=300)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    base_cfg = SO101SDKConfig(
        robot_port=args.robot_port,
        robot_id=args.robot_id,
        calibration_dir=args.calibration_dir,
        control_mode=args.control_mode,
        ik_solver=args.ik_solver,
        max_command_delta_deg=args.max_command_delta_deg,
        max_relative_target_deg=args.max_relative_target_deg,
        gripper_max_relative_target=args.gripper_max_relative_target,
        dry_run=args.dry_run,
    )
    app = SO101WebApp(base_cfg, max_logs=args.log_keep)
    SO101RequestHandler.app = app
    httpd = ThreadingHTTPServer((args.host, args.port), SO101RequestHandler)

    print(f"SO101 Web UI running: http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop server.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            app.disconnect()
        except Exception:
            pass
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
