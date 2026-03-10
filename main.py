from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse, PlainTextResponse
from pydantic import BaseModel
from typing import Optional
import httpx
import json
import os
import re
from difflib import SequenceMatcher

app = FastAPI()

PROJECTS_DIR = "projects"
CONFIG_FILE  = "graft_config.json"

os.makedirs(PROJECTS_DIR, exist_ok=True)

# ── Default config ────────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "intent_compiler": {
        "source": "ollama",
        "ollama_model": "qwen2.5:3b",
        "ollama_url": "http://localhost:11434",
        "anthropic_model": "claude-sonnet-4-20250514",
        "anthropic_key": ""
    },
    "code_compiler": {
        "source": "ollama",
        "ollama_model": "qwen2.5-coder:3b",
        "ollama_url": "http://localhost:11434",
        "anthropic_model": "claude-sonnet-4-20250514",
        "anthropic_key": ""
    },
    "mid_verbosity": "moderate"   # "minimal" | "moderate" | "free"
}

def load_config() -> dict:
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE) as f:
                saved = json.load(f)
            cfg = json.loads(json.dumps(DEFAULT_CONFIG))
            for role in ["intent_compiler", "code_compiler"]:
                if role in saved:
                    cfg[role].update(saved[role])
            if "mid_verbosity" in saved:
                cfg["mid_verbosity"] = saved["mid_verbosity"]
            return cfg
        except Exception:
            pass
    return json.loads(json.dumps(DEFAULT_CONFIG))

def save_config(cfg: dict):
    with open(CONFIG_FILE, "w") as f:
        json.dump(cfg, f, indent=2)

# ── Data models ───────────────────────────────────────────────────────────────

class IntentNode(BaseModel):
    id: str
    label: str
    intent: str
    parent: Optional[str] = None
    children: list[str] = []
    owns: list[str] = []

class MidSection(BaseModel):
    id: str
    keyword: str
    name: str
    intent_tag: str
    owner: str
    body: str

class Project(BaseModel):
    name: str
    nodes: list[IntentNode]
    root_id: Optional[str] = None

class CompileNodeRequest(BaseModel):
    project_name: str
    node: IntentNode
    existing_sections: list[MidSection] = []

class CompileCodeRequest(BaseModel):
    mid: str

class SaveProjectRequest(BaseModel):
    project: Project

class SaveMidRequest(BaseModel):
    project_name: str
    sections: list[MidSection]

class SaveCodeRequest(BaseModel):
    project_name: str
    filename: str
    content: str

class OverlapCheckRequest(BaseModel):
    node_label: str
    node_intent: str
    existing_sections: list[MidSection]

class ConfigUpdate(BaseModel):
    intent_compiler: dict
    code_compiler: dict
    mid_verbosity: Optional[str] = None

# ── Mid file format ───────────────────────────────────────────────────────────

def sections_to_text(sections: list[MidSection]) -> str:
    parts = []
    for s in sections:
        parts.append("\n".join([
            f"{s.keyword}: {s.name}",
            f"OWNER: {s.owner}",
            f"INTENT: {s.intent_tag}",
            "---",
            s.body.strip(),
        ]))
    return "\n\n".join(parts)

def text_to_sections(text: str, owner_id: str = "") -> list[MidSection]:
    raw_sections = re.split(r'\n(?=(?:STATE|TYPE|STRUCTURE|WHEN):\s)', text.strip())
    result = []
    for i, raw in enumerate(raw_sections):
        raw = raw.strip()
        if not raw:
            continue
        lines = raw.split('\n')
        keyword = name = intent_tag = ''
        owner = owner_id
        body_lines = []
        in_body = False

        for line in lines:
            if in_body:
                body_lines.append(line)
                continue
            m = re.match(r'^(STATE|TYPE|STRUCTURE|WHEN):\s*(.*)', line, re.IGNORECASE)
            if m and not keyword:
                keyword = m.group(1).upper()
                name = m.group(2).strip()
            elif line.startswith('OWNER:'):
                owner = line[6:].strip() or owner_id
            elif line.startswith('INTENT:'):
                intent_tag = line[7:].strip()
            elif line.strip() == '---':
                in_body = True
            else:
                body_lines.append(line)

        body = '\n'.join(body_lines).strip()
        body = clean_body(body)

        if keyword and body and len(body) > 20:
            result.append(MidSection(
                id=f"s{i}", keyword=keyword, name=name,
                intent_tag=intent_tag, owner=owner, body=body
            ))
    return result

def load_mid_sections(project_name: str) -> list[MidSection]:
    safe = project_name.replace(" ", "_").lower()
    path = os.path.join(PROJECTS_DIR, f"{safe}.mid")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return text_to_sections(f.read())

def save_mid_sections(project_name: str, sections: list[MidSection]):
    safe = project_name.replace(" ", "_").lower()
    with open(os.path.join(PROJECTS_DIR, f"{safe}.mid"), "w", encoding="utf-8") as f:
        f.write(sections_to_text(sections))

# ── Body cleanup: strip narrative filler ─────────────────────────────────────
# Catches model artifacts like "Now the user can...", "At this point...", etc.

FILLER_PATTERNS = [
    r'(?i)^now[,\s]+(the\s+)?(user|app|system|interface)',
    r'(?i)^at this point[,\s]',
    r'(?i)^with this[,\s]+(the\s+)?(user|app|system)',
    r'(?i)^this (means|allows|enables|lets)',
    r'(?i)^the (user|app|system) (can now|is now able)',
    r'(?i)^in (summary|conclusion|short)',
    r'(?i)^overall[,\s]',
    r'(?i)^as a result[,\s]',
    r'(?i)^to summarize[,\s]',
]
_FILLER_RE = [re.compile(p) for p in FILLER_PATTERNS]

def clean_body(text: str) -> str:
    """Remove known filler/narrative sentences from model output."""
    paragraphs = re.split(r'\n{2,}', text.strip())
    cleaned = []
    for para in paragraphs:
        sentences = re.split(r'(?<=[.!?])\s+', para.strip())
        kept = []
        for sent in sentences:
            s = sent.strip()
            if not s:
                continue
            if any(p.match(s) for p in _FILLER_RE):
                continue
            kept.append(s)
        if kept:
            cleaned.append(' '.join(kept))
    return '\n\n'.join(cleaned)

# ── Overlap detection ─────────────────────────────────────────────────────────

def similarity(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def extract_keywords(text):
    stopwords = {'a','an','the','and','or','of','to','in','is','it','for','with','that','this','be','as','on','at','by','from','into'}
    return {w for w in re.findall(r'\b[a-z]{3,}\b', text.lower()) if w not in stopwords}

def detect_overlaps(node_label, node_intent, existing_sections):
    intent_words = ' '.join(node_intent.split()[:20])
    node_kws = extract_keywords(f"{node_label} {intent_words}")
    overlaps = []
    for sec in existing_sections:
        sec_kws = extract_keywords(f"{sec.name} {sec.intent_tag}")
        if not node_kws or not sec_kws:
            continue
        score = max(
            len(node_kws & sec_kws) / len(node_kws | sec_kws),
            similarity(node_label, sec.name),
            similarity(node_label, sec.intent_tag),
            similarity(intent_words, sec.intent_tag)
        )
        if score >= 0.35:
            overlaps.append({
                "section_id": sec.id, "section_keyword": sec.keyword,
                "section_name": sec.name, "section_intent": sec.intent_tag,
                "current_owner": sec.owner, "score": round(score, 2)
            })
    return sorted(overlaps, key=lambda x: x["score"], reverse=True)

# ── Verbosity instructions ────────────────────────────────────────────────────

VERBOSITY_HINTS = {
    "minimal": (
        "SCOPE: Be strictly minimal. Cover only what the intent explicitly states. "
        "Do not add features, edge cases, or sections not directly implied by the intent phrase. "
        "Prefer 1-2 sections total. Stop as soon as the intent is fully covered."
    ),
    "moderate": (
        "SCOPE: Cover what the intent states plus obvious implied requirements "
        "(e.g. if a form is described, include its submit behavior). "
        "Do not invent features the intent does not mention. Aim for 2-4 sections."
    ),
    "free": (
        # No scope injection — model decides
        ""
    ),
}

# ── Prompts ───────────────────────────────────────────────────────────────────

def node_to_mid_prompt(node: IntentNode, existing_names: list[str],
                        existing_summary: str = "", verbosity: str = "moderate") -> str:
    existing_hint = ""
    if existing_names:
        existing_hint = f"\nAlready defined — do NOT redefine or contradict these:\n" + "\n".join(f"  - {n}" for n in existing_names)
    context_hint = ""
    if existing_summary:
        context_hint = f"\nExisting project context (read-only, for reference only):\n{existing_summary}\n"

    verbosity_hint = VERBOSITY_HINTS.get(verbosity, VERBOSITY_HINTS["moderate"])
    verbosity_block = f"\n{verbosity_hint}\n" if verbosity_hint else ""

    return f"""You are a Mid Language compiler. Mid is the human-readable specification layer between intent and code.

WHAT MID IS:
Mid describes software in plain English. It is read by humans to verify correctness and by a code compiler to generate implementation. It must be precise enough to generate correct code but must never mention implementation details.

MID GUIDELINES — follow these strictly:

STATE sections:
- Declare what data exists in memory: variable names, types, initial values.
- Do NOT describe how data changes (that is WHEN). Do NOT describe layout (that is STRUCTURE).
- Use snake_case for all variable names. One STATE per logical data group.

TYPE sections:
- Describe the shape of one structured data entity (used when STATE holds a collection).
- List every field, its type, and its default value. Nothing else.
- Only write TYPE if STATE contains a collection of structured items.

STRUCTURE sections:
- Describe what the user sees: layout, elements, labels, visibility conditions.
- Reference STATE variable names to describe dynamic content.
- Do NOT describe what happens on interaction (that is WHEN). Do NOT mention CSS, colors, or libraries.

WHEN sections:
- Describe exactly one user action or system event from trigger to result.
- Must include: trigger, precondition (if any), every state change caused, visible feedback.
- One WHEN per distinct user action. Never combine two actions into one WHEN.

STYLE RULES — always follow:
- Write declaratively. Never use narrative phrases like "Now the user can...", "At this point...", "This means...", "With this in place...".
- No summaries, no conclusions, no meta-commentary. Just specification.
- Every sentence must describe a fact about the system, not a story about using it.

NEVER include in Mid: pixel values, color names, library names, CSS classes, API calls, SQL, file paths, or any implementation detail.
{verbosity_block}
OUTPUT FORMAT — one section at a time, exactly like this:

KEYWORD: section_name
INTENT: the intent phrase this section was compiled from
---
2 to 4 full English sentences. Plain language. No bullet points. No code.

EXAMPLE:

STATE: task_list
INTENT: the app stores a list of tasks
---
The application holds one central piece of data: task_list, an ordered collection of task items that starts empty when the page loads. Items are added and removed as the user interacts with the app.

TYPE: task
INTENT: each task has a title and a done flag
---
A task has a task_title, a short text string set at creation and never modified. It also has task_done, a boolean that starts as false and flips when the user marks the task complete.

WHEN: add_task
INTENT: pressing Add creates a task from the input field
---
This fires when the user activates the "Add" control. It requires current_input to be non-empty — if empty, nothing happens. A new task is created with task_title equal to current_input and task_done set to false, appended to task_list. current_input is then cleared.

---

Now compile this intent into Mid sections:
  Label: {node.label or 'Application'}
  Intent: {node.intent}
{context_hint}{existing_hint}

Write only sections directly introduced by this intent. Start with the first keyword line. No preamble. No summary at the end.
"""

def mid_to_code_prompt(mid: str) -> str:
    return f"""You are a code compiler. Read the Mid document below and produce a single complete working HTML file.

Rules:
- Output ONLY the HTML. Nothing before <!DOCTYPE html>. No markdown fences.
- All CSS inside a <style> tag in <head>.
- All JavaScript inside a <script> tag before </body>.
- Implement every WHEN behavior as a JavaScript function.
- Match the STRUCTURE sections exactly.
- Use the STATE and TYPE sections to define your data model.
- Dark background preferred unless Mid says otherwise.
- Mark each function with its Mid section name: // [graft:section_name]

Mid document:
{mid}

Write the complete HTML file now:
"""

# ── Streaming: Ollama ─────────────────────────────────────────────────────────

async def stream_ollama(cfg: dict, prompt: str):
    url = cfg.get("ollama_url", "http://localhost:11434")
    model = cfg.get("ollama_model", "qwen2.5:3b")
    full = []
    try:
        async with httpx.AsyncClient(timeout=600.0) as client:
            async with client.stream("POST", f"{url}/api/generate",
                json={"model": model, "prompt": prompt, "stream": True}) as resp:
                if resp.status_code != 200:
                    err = await resp.aread()
                    yield f"data: {json.dumps({'error': err.decode()})}\n\n"
                    return
                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    token = obj.get("response", "")
                    if token:
                        full.append(token)
                        yield f"data: {json.dumps({'token': token})}\n\n"
                    if obj.get("done"):
                        break
        yield f"data: {json.dumps({'done': True, 'full': ''.join(full).strip()})}\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"

# ── Streaming: Anthropic ──────────────────────────────────────────────────────

async def stream_anthropic(cfg: dict, prompt: str):
    api_key = cfg.get("anthropic_key", "")
    model   = cfg.get("anthropic_model", "claude-sonnet-4-20250514")
    if not api_key:
        yield f"data: {json.dumps({'error': 'No Anthropic API key configured. Add it in Settings.'})}\n\n"
        return
    full = []
    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            async with client.stream(
                "POST", "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": model,
                    "max_tokens": 4096,
                    "stream": True,
                    "messages": [{"role": "user", "content": prompt}]
                }
            ) as resp:
                if resp.status_code != 200:
                    err = await resp.aread()
                    yield f"data: {json.dumps({'error': f'Anthropic error {resp.status_code}: ' + err.decode()[:200]})}\n\n"
                    return
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    raw = line[5:].strip()
                    if raw == "[DONE]":
                        break
                    try:
                        obj = json.loads(raw)
                    except Exception:
                        continue
                    if obj.get("type") == "content_block_delta":
                        token = obj.get("delta", {}).get("text", "")
                        if token:
                            full.append(token)
                            yield f"data: {json.dumps({'token': token})}\n\n"
                    if obj.get("type") == "message_stop":
                        break
        yield f"data: {json.dumps({'done': True, 'full': ''.join(full).strip()})}\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"

# ── Unified stream dispatcher ─────────────────────────────────────────────────

def stream_model(role_cfg: dict, prompt: str):
    if role_cfg.get("source") == "anthropic":
        return stream_anthropic(role_cfg, prompt)
    return stream_ollama(role_cfg, prompt)

# ── Routes: config ────────────────────────────────────────────────────────────

@app.get("/config")
async def get_config():
    cfg = load_config()
    for role in ["intent_compiler", "code_compiler"]:
        key = cfg[role].get("anthropic_key", "")
        if key:
            cfg[role]["anthropic_key_set"] = True
            cfg[role]["anthropic_key"] = key[:8] + "…"
        else:
            cfg[role]["anthropic_key_set"] = False
    return cfg

@app.post("/config")
async def post_config(update: ConfigUpdate):
    cfg = load_config()
    for role in ["intent_compiler", "code_compiler"]:
        incoming = dict(getattr(update, role))
        if "…" in incoming.get("anthropic_key", ""):
            incoming.pop("anthropic_key", None)
        incoming.pop("anthropic_key_set", None)
        cfg[role].update(incoming)
    if update.mid_verbosity in ("minimal", "moderate", "free"):
        cfg["mid_verbosity"] = update.mid_verbosity
    save_config(cfg)
    return {"saved": True}

@app.get("/config/ollama/models")
async def list_ollama_models():
    cfg = load_config()
    urls = set([
        cfg["intent_compiler"].get("ollama_url", "http://localhost:11434"),
        cfg["code_compiler"].get("ollama_url", "http://localhost:11434"),
    ])
    models = []
    for url in urls:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.get(f"{url}/api/tags")
                if r.status_code == 200:
                    data = r.json()
                    for m in data.get("models", []):
                        name = m.get("name","")
                        if name and name not in models:
                            models.append(name)
        except Exception:
            pass
    return {"models": sorted(models)}

# ── Routes: compile ───────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return FileResponse("index.html", headers={
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache"
    })

@app.post("/check/overlaps")
async def check_overlaps(req: OverlapCheckRequest):
    return {"overlaps": detect_overlaps(req.node_label, req.node_intent, req.existing_sections)}

@app.post("/compile/mid")
async def compile_mid(req: CompileNodeRequest):
    cfg = load_config()
    role_cfg = cfg["intent_compiler"]
    verbosity = cfg.get("mid_verbosity", "moderate")

    existing_names = [
        f"{s.keyword}:{s.name}"
        for s in req.existing_sections
        if s.owner != req.node.id
    ]

    existing_summary = ""
    if req.existing_sections:
        lines = []
        for s in req.existing_sections:
            if s.owner != req.node.id:
                lines.append(f"  {s.keyword}:{s.name} — {s.intent_tag}")
        if lines:
            existing_summary = "\n".join(lines)

    prompt = node_to_mid_prompt(req.node, existing_names, existing_summary, verbosity)

    async def generate():
        full_parts = []
        async for chunk in stream_model(role_cfg, prompt):
            yield chunk
            try:
                obj = json.loads(chunk[5:].strip())
                if obj.get("token"):
                    full_parts.append(obj["token"])
                if obj.get("done"):
                    full_text = obj.get("full", "".join(full_parts)).strip()
                    new_sections = text_to_sections(full_text, owner_id=req.node.id)

                    if not new_sections:
                        repair_notice = json.dumps({"token": "\n\n[No valid sections found — attempting repair...]\n\n"})
                        yield f"data: {repair_notice}\n\n"

                        repair_prompt = (
                            "Write a Mid document section describing this application feature.\n\n"
                            f"Feature: {req.node.intent}\n\n"
                            "Use exactly this format and nothing else:\n\n"
                            "STATE: data_name\n"
                            f"INTENT: {req.node.intent}\n"
                            "---\n"
                            "Write 3 sentences here describing what data exists, what it contains, and how it starts.\n\n"
                            "Then if the feature involves user actions, add:\n\n"
                            "WHEN: action_name\n"
                            f"INTENT: {req.node.intent}\n"
                            "---\n"
                            "Write 3 sentences describing the trigger, what changes, and the result.\n\n"
                            "Start with STATE: now."
                        )
                        repair_parts = []
                        async for rc in stream_model(role_cfg, repair_prompt):
                            yield rc
                            try:
                                ro = json.loads(rc[5:].strip())
                                if ro.get("token"):
                                    repair_parts.append(ro["token"])
                                if ro.get("done"):
                                    new_sections = text_to_sections("".join(repair_parts).strip(), owner_id=req.node.id)
                            except Exception:
                                pass

                        if not new_sections:
                            err_msg = json.dumps({"error": "Could not generate valid Mid sections after repair. Try rephrasing your intent as a single clear sentence."})
                            yield f"data: {err_msg}\n\n"
                            return

                    kept = [s for s in req.existing_sections if s.owner != req.node.id]
                    merged = kept + new_sections
                    for i, s in enumerate(merged):
                        s.id = f"s{i}"
                    save_mid_sections(req.project_name, merged)
                    yield f"data: {json.dumps({'saved': True, 'sections': [s.dict() for s in merged]})}\n\n"
            except Exception:
                pass

    return StreamingResponse(generate(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

@app.post("/compile/code")
async def compile_code(req: CompileCodeRequest):
    cfg = load_config()
    role_cfg = cfg["code_compiler"]
    prompt = mid_to_code_prompt(req.mid)
    return StreamingResponse(stream_model(role_cfg, prompt), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

# ── Routes: Mid persistence ───────────────────────────────────────────────────

@app.get("/projects/{name}/mid")
async def get_mid(name: str):
    secs = load_mid_sections(name)
    return {"sections": [s.dict() for s in secs], "text": sections_to_text(secs) if secs else ""}

@app.post("/projects/mid")
async def post_mid(req: SaveMidRequest):
    save_mid_sections(req.project_name, req.sections)
    return {"saved": True}

# ── Routes: code persistence ──────────────────────────────────────────────────

@app.post("/projects/code")
async def save_code(req: SaveCodeRequest):
    safe = req.project_name.replace(" ","_").lower()
    code_dir = os.path.join(PROJECTS_DIR, safe + "_output")
    os.makedirs(code_dir, exist_ok=True)
    fname = re.sub(r'[^\w.\-]', '_', req.filename)
    with open(os.path.join(code_dir, fname), "w", encoding="utf-8") as f:
        f.write(req.content)
    return {"saved": fname}

@app.get("/projects/{name}/code")
async def list_code_files(name: str):
    safe = name.replace(" ","_").lower()
    code_dir = os.path.join(PROJECTS_DIR, safe + "_output")
    if not os.path.exists(code_dir):
        return {"files": []}
    files = [{"name": f, "size": os.path.getsize(os.path.join(code_dir, f))}
             for f in os.listdir(code_dir)]
    return {"files": files}

# ── Routes: project persistence ───────────────────────────────────────────────

@app.get("/projects/{name}_output/{filename}")
async def serve_output_file(name: str, filename: str):
    safe = name.replace(" ","_").lower()
    path = os.path.join(PROJECTS_DIR, f"{safe}_output", filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found")
    with open(path, encoding="utf-8") as f:
        content = f.read()
    media = "text/html" if filename.endswith(".html") else "text/plain"
    return PlainTextResponse(content, media_type=media)

@app.get("/projects")
async def list_projects():
    files = [f.replace(".json","") for f in os.listdir(PROJECTS_DIR) if f.endswith(".json")]
    return {"projects": sorted(files)}

@app.get("/projects/{name}")
async def load_project(name: str):
    path = os.path.join(PROJECTS_DIR, f"{name}.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Project not found")
    with open(path) as f:
        return json.load(f)

@app.post("/projects")
async def save_project(req: SaveProjectRequest):
    safe = req.project.name.replace(" ","_").lower()
    with open(os.path.join(PROJECTS_DIR, f"{safe}.json"), "w") as f:
        json.dump(req.project.dict(), f, indent=2)
    return {"saved": safe}

@app.delete("/projects/{name}")
async def delete_project(name: str):
    for ext in [".json", ".mid"]:
        p = os.path.join(PROJECTS_DIR, f"{name}{ext}")
        if os.path.exists(p):
            os.remove(p)
    return {"deleted": name}