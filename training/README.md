# Mid LoRA — Training & Deployment Guide

## What this is
A LoRA adapter for `Qwen2.5-7B-Instruct` that specialises it as a Mid compiler.
Fixes: keyword confusion, global-part dumping, schema language in TYPE bodies.

## Dataset
- 25 hand-authored examples (train: 21, val: 4)
- Tags: simple, medium, medium_complex, auth, edge, targeted, targeted_rewrite

---

## Setup

```bash
# In your graft venv
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2/ --upgrade --force-reinstall
pip install unsloth trl datasets
python -c "import unsloth; print('ok')"
```

---

## Training

```bash
cd ~/git/graft/lora_mid   # or wherever you copied these files
python train.py
```

**Expected time on RX 6600 XT:** ~45–90 minutes for 4 epochs over 21 examples.
Watch the training loss — it should drop from ~2.0 to ~0.3–0.5 by the end.
If it goes below 0.1 you are likely overfitting — stop early or reduce epochs.

**VRAM usage:** ~6–7GB with 4-bit quantisation. Should fit comfortably.

---

## Convert to GGUF for Ollama

After training, `mid_lora_adapter/` contains the LoRA weights.
You need to merge them into the base model and convert to GGUF.

```bash
# 1. Install llama.cpp (if not already)
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && make

# 2. Merge LoRA into base (run from lora_mid/)
python merge_and_export.py   # see below

# 3. Convert merged model to GGUF
python llama.cpp/convert_hf_to_gguf.py ./mid_merged_model \
    --outfile mid_specialist_q4.gguf \
    --outtype q4_k_m

# 4. Load into Ollama
ollama create mid-specialist -f Modelfile
```

---

## merge_and_export.py

Create this file in lora_mid/:

```python
from unsloth import FastLanguageModel
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name    = "./mid_lora_adapter",
    max_seq_length= 2048,
    dtype         = torch.float16,
    load_in_4bit  = True,
)
model = model.merge_and_unload()
model.save_pretrained("./mid_merged_model")
tokenizer.save_pretrained("./mid_merged_model")
print("Merged and saved to ./mid_merged_model")
```

---

## Modelfile for Ollama

```
FROM ./mid_specialist_q4.gguf
PARAMETER temperature 0.3
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.1
SYSTEM ""
```

Note: temperature 0.3 — lower than default because Mid output should be
deterministic and well-structured, not creative.

---

## Wire into Graft

In `graft_config.json`, change the intent_compiler:

```json
{
  "intent_compiler": {
    "source": "ollama",
    "ollama_model": "mid-specialist",
    "ollama_url": "http://localhost:11434"
  }
}
```

---

## If you want more training data

Run `generate_dataset.py` — it has all 25 examples inline.
To add more: append `ex("prompt", """output""", tag="tag")` calls before
the print statement at the bottom, then re-run the script.

The most impactful additions would be:
- More targeted recompilation examples (the model struggles most here)
- Examples with 3-4 part apps (teaches splitting)
- Examples where the user gives a vague prompt (teaches not inventing features)

---

## Troubleshooting

**`NotImplementedError: AMD ROCm GPU but no HIP accelerator`**
→ Reinstall PyTorch ROCm wheels (see Setup above)

**OOM during training**
→ Set `load_in_4bit = True` (already default), reduce `MAX_SEQ_LEN` to 1024

**Loss not decreasing**
→ Check that `format_example` is producing valid chat-template output
→ Print `ds["train"][0]["text"]` and verify structure

**Model still writes `STATE: WHEN` after training**
→ Dataset too small — add 10 more targeted examples showing this error corrected
→ Increase EPOCHS to 6
