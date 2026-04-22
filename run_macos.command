#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi

source ".venv/bin/activate"
python -m pip install --upgrade pip
pip install -r requirements.txt

if [ ! -f ".env" ]; then
  cp ".env.example" ".env"
  echo "Created .env from template. Please set your vision API key in .env (see .env.example)."
fi

streamlit run ui/app.py
