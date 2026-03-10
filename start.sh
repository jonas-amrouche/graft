#!/bin/bash
echo "Starting Graft..."
echo "Make sure Ollama is running before continuing."
echo ""

# Create venv if it doesn't exist yet
if [ ! -d ".venv" ]; then
  echo "Creating virtual environment..."
  python3 -m venv .venv
fi

# Activate and install
source .venv/bin/activate
pip install -r requirements.txt --quiet

echo ""
echo "Opening http://localhost:8000"
xdg-open http://localhost:8000 2>/dev/null &

uvicorn main:app --host 0.0.0.0 --port 8000 --reload