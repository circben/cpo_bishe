#!/usr/bin/env bash
set -euo pipefail

# Ubuntu 22.04 bootstrap for cpo_project
# Default stack: Python 3.10 + PyTorch cu121 (works on CUDA 12.4 driver)

ENV_NAME="cpo"
PYTHON_VERSION="3.10"
TORCH_CHANNEL_URL="https://download.pytorch.org/whl/cu121"
PROJECT_DIR="${PROJECT_DIR:-$PWD}"
INSTALL_MINICONDA="${INSTALL_MINICONDA:-1}"
RUN_SMOKE_TESTS="${RUN_SMOKE_TESTS:-1}"

log() {
  echo "[bootstrap] $*"
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing command: $1"
    exit 1
  fi
}

install_system_packages() {
  log "Installing system packages"
  sudo apt-get update
  sudo apt-get install -y \
    git curl wget ca-certificates build-essential \
    pkg-config unzip tmux htop
}

install_miniconda_if_needed() {
  if command -v conda >/dev/null 2>&1; then
    log "Conda already exists: $(command -v conda)"
    return
  fi

  if [[ "$INSTALL_MINICONDA" != "1" ]]; then
    echo "Conda not found and INSTALL_MINICONDA=0"
    exit 1
  fi

  log "Installing Miniconda to $HOME/miniconda3"
  wget -qO "$HOME/miniconda.sh" "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
  bash "$HOME/miniconda.sh" -b -p "$HOME/miniconda3"
  rm -f "$HOME/miniconda.sh"

  # shellcheck disable=SC1091
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda init bash >/dev/null 2>&1 || true
}

activate_conda() {
  if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
  elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
  else
    echo "Cannot find conda.sh. Please install conda first."
    exit 1
  fi
}

create_or_update_env() {
  log "Creating/updating conda env: $ENV_NAME"
  if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    log "Env $ENV_NAME exists, reusing"
  else
    conda create -n "$ENV_NAME" "python=$PYTHON_VERSION" -y
  fi

  conda activate "$ENV_NAME"
  python -m pip install --upgrade pip
}

install_python_packages() {
  log "Installing PyTorch cu121"
  python -m pip install \
    torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url "$TORCH_CHANNEL_URL"

  log "Installing project dependencies"
  python -m pip install \
    transformers datasets accelerate peft trl bitsandbytes sentencepiece \
    pandas numpy scipy scikit-learn matplotlib seaborn networkx tqdm \
    pyyaml requests evaluate pytest hf-transfer
}

write_env_template() {
  local env_file="$PROJECT_DIR/.env.server.example"
  log "Writing env template: $env_file"
  cat > "$env_file" <<'EOF'
# Copy to .env.server and fill with your values
LLM_API_KEY=
LLM_PROVIDER=openai_compatible
LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_TIMEOUT=180
LLM_ENABLE_THINKING=false
LLM_MAX_RETRIES=4
LLM_RETRY_BACKOFF_SEC=1.0
EOF
}

run_smoke_tests() {
  if [[ "$RUN_SMOKE_TESTS" != "1" ]]; then
    log "Skipping smoke tests (RUN_SMOKE_TESTS=$RUN_SMOKE_TESTS)"
    return
  fi

  log "Running smoke tests"
  cd "$PROJECT_DIR"

  python - <<'PY'
import torch
print('torch_version=', torch.__version__)
print('cuda_available=', torch.cuda.is_available())
print('torch_cuda=', torch.version.cuda)
if torch.cuda.is_available():
    print('gpu=', torch.cuda.get_device_name(0))
PY

  python scripts/stage2/run_stage2_pipeline.py --print-config
  python scripts/stage3/train_dpo.py --help
}

main() {
  log "Project dir: $PROJECT_DIR"
  install_system_packages
  install_miniconda_if_needed
  activate_conda
  create_or_update_env
  install_python_packages
  write_env_template
  run_smoke_tests

  log "Done."
  log "Next:"
  log "1) Fill $PROJECT_DIR/.env.server.example (or your own env export)"
  log "2) conda activate $ENV_NAME"
  log "3) cd $PROJECT_DIR"
}

main "$@"
