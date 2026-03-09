@echo off
echo Starting Midlang...
echo Make sure Ollama is running before continuing.
echo.
pip install -r requirements.txt --quiet
echo.
echo Opening http://localhost:8000
start http://localhost:8000
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
