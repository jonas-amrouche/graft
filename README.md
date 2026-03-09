# Graft — Intent Compiler

## Setup

1. Make sure Ollama is running and you have pulled both models:
   ```
   ollama pull mistral
   ollama pull qwen2.5-coder:7b
   ```

2. Double-click `start.bat` (or run it in a terminal)
   - It installs dependencies automatically
   - Opens http://localhost:8000 in your browser

## How to use

1. Click **+ New** to create a project
2. Click **+ Root** to add your first intent node
3. Type your intent in the editor at the bottom left
4. Add child intents with **+ Child intent** to refine specific parts
5. Click **⚡ Generate Mid** — Mistral compiles your tree into a Mid document
6. Read the Mid document and verify it matches your intent
7. Click **▶ Generate Code** — Qwen2.5-Coder turns Mid into a working HTML app
8. Check the **Preview** tab to see it running

## Files

```
graft/
  main.py           — FastAPI server + Ollama pipeline
  index.html        — Full UI (intent tree + Mid viewer + preview)
  requirements.txt  — Python dependencies
  start.bat         — Windows startup script
  projects/         — Saved projects (JSON files)
```

## Models used

| Step | Model | Task |
|------|-------|------|
| Intent → Mid | mistral | Creative, English → structured prose |
| Mid → Code | qwen2.5-coder:7b | Translation, prose → HTML/CSS/JS |
