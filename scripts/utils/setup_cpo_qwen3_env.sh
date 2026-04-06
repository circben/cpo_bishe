#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="cpo_qwen3"

source /root/miniconda3/etc/profile.d/conda.sh

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "[env] ${ENV_NAME} already exists, reusing"
else
  echo "[env] creating ${ENV_NAME}"
  conda create -y -n "${ENV_NAME}" python=3.10 pip
fi

conda activate "${ENV_NAME}"

echo "[env] installing torch/cu121"
pip install --upgrade pip
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

echo "[env] installing qwen3 + dpo stack"
pip install \
  transformers==4.51.3 \
  trl==0.11.4 \
  peft==0.18.1 \
  accelerate==1.13.0 \
  datasets==4.8.4 \
  bitsandbytes \
  sentencepiece \
  tiktoken \
  rich \
  pandas \
  scipy \
  scikit-learn \
  matplotlib \
  seaborn \
  evaluate \
  pytest

echo "[env] validating qwen3 + dpotrainer"
python - <<'PY'
from transformers import AutoConfig
from trl import DPOTrainer
import transformers, torch, trl
cfg = AutoConfig.from_pretrained('/root/autodl-tmp/llm/Qwen3-8B', local_files_only=True)
print('transformers', transformers.__version__)
print('torch', torch.__version__)
print('trl', trl.__version__)
print('model_type', cfg.model_type)
print('DPOTrainer', DPOTrainer.__name__)
PY

echo "[env] done: ${ENV_NAME}"
