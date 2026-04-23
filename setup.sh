#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"   # cd into egoself/

ENV_NAME=${ENV_NAME:-egoself}
CUDA=${CUDA:-cu121}
PYTHON_VERSION=${PYTHON_VERSION:-3.10}

conda create -n "$ENV_NAME" "python=$PYTHON_VERSION" -y

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
    --index-url "https://download.pytorch.org/whl/$CUDA"

pip install -r requirements/memory.txt
pip install -r requirements/event_graph.txt
pip install -r requirements/base.txt

pip install git+https://github.com/openai/CLIP.git

if [[ "${SKIP_NEO4J:-0}" != "1" ]]; then
  if ! docker ps -a --format '{{.Names}}' | grep -q '^neo4j-egoself$'; then
    docker run -d --name neo4j-egoself \
      -p 7474:7474 -p 7687:7687 \
      -e NEO4J_AUTH=neo4j/password \
      --env NEO4J_PLUGINS='["graph-data-science"]' \
      neo4j:latest
  fi
fi
