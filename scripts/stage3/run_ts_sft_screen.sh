#!/usr/bin/env bash
set -euo pipefail

# TS-SFT screen 管理脚本
# 用法示例:
#   bash scripts/stage3/run_ts_sft_screen.sh start -- --task gsm8k
#   bash scripts/stage3/run_ts_sft_screen.sh status
#   bash scripts/stage3/run_ts_sft_screen.sh logs
#   bash scripts/stage3/run_ts_sft_screen.sh attach
#   bash scripts/stage3/run_ts_sft_screen.sh stop

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STATE_DIR="/tmp"
RUN_ID_FILE="$STATE_DIR/stage3_tssft_active_run_id.txt"
SESSION_FILE="$STATE_DIR/stage3_tssft_active_screen.txt"
LOG_FILE="$STATE_DIR/stage3_tssft_active_log.txt"

usage() {
  cat <<'EOF'
run_ts_sft_screen.sh <command> [args]

Commands:
  start [--run-id <id>] [-- <train_ts_sft.py args...>]
      启动 TS-SFT 训练到后台 screen 会话。

  status
      查看当前会话状态。

  logs
      持续查看当前日志。

  attach
      进入当前 screen 会话。

  stop
      停止当前会话。

  help
      显示帮助。
EOF
}

require_state_file() {
  local f="$1"
  if [[ ! -f "$f" ]]; then
    echo "State file not found: $f"
    exit 1
  fi
}

cmd_start() {
  local run_id=""
  local -a script_args=()

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --run-id)
        run_id="$2"
        shift 2
        ;;
      --)
        shift
        script_args=("$@")
        break
        ;;
      *)
        script_args+=("$1")
        shift
        ;;
    esac
  done

  if [[ -z "$run_id" ]]; then
    run_id="stage3_tssft_$(date +%Y%m%d_%H%M%S)"
  fi

  local session="tssft_${run_id#stage3_}"
  local log="logs/${run_id}.log"

  script_args+=(--run-id "$run_id")

  local script_args_escaped=""
  for arg in "${script_args[@]}"; do
    printf -v script_args_escaped "%s %q" "$script_args_escaped" "$arg"
  done

  cd "$ROOT_DIR"
  mkdir -p logs

  echo "$run_id" > "$RUN_ID_FILE"
  echo "$session" > "$SESSION_FILE"
  echo "$log" > "$LOG_FILE"

  screen -dmS "$session" bash -lc "cd '$ROOT_DIR' && /root/miniconda3/envs/cpo/bin/python scripts/stage3/train_ts_sft.py${script_args_escaped} 2>&1 | tee '$log'"

  echo "Started"
  echo "session=$session"
  echo "run_id=$run_id"
  echo "log=$log"
}

cmd_status() {
  require_state_file "$SESSION_FILE"
  require_state_file "$RUN_ID_FILE"
  require_state_file "$LOG_FILE"

  local session run_id log
  session="$(cat "$SESSION_FILE")"
  run_id="$(cat "$RUN_ID_FILE")"
  log="$(cat "$LOG_FILE")"

  echo "session=$session"
  echo "run_id=$run_id"
  echo "log=$log"

  if screen -ls | grep -q "[.]${session}[[:space:]]"; then
    echo "state=running"
  else
    echo "state=not-running"
  fi
}

cmd_logs() {
  require_state_file "$LOG_FILE"
  local log
  log="$(cat "$LOG_FILE")"
  cd "$ROOT_DIR"
  if [[ ! -f "$log" ]]; then
    echo "Log file not found: $log"
    exit 1
  fi
  tail -f "$log"
}

cmd_attach() {
  require_state_file "$SESSION_FILE"
  local session
  session="$(cat "$SESSION_FILE")"
  exec screen -r "$session"
}

cmd_stop() {
  require_state_file "$SESSION_FILE"
  local session
  session="$(cat "$SESSION_FILE")"
  screen -S "$session" -X quit || true
  echo "Stopped session: $session"
}

main() {
  local cmd="${1:-help}"
  shift || true
  case "$cmd" in
    start) cmd_start "$@" ;;
    status) cmd_status ;;
    logs) cmd_logs ;;
    attach) cmd_attach ;;
    stop) cmd_stop ;;
    help|-h|--help) usage ;;
    *)
      echo "Unknown command: $cmd"
      usage
      exit 1
      ;;
  esac
}

main "$@"
