"""
Mid LoRA training — bf16, ROCm stable
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig

print(f"Torch: {torch.__version__}")
print(f"GPU:   {torch.cuda.get_device_name(0)}")
print(f"VRAM:  {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"bf16 supported: {torch.cuda.is_bf16_supported()}")

BASE_MODEL  = "Qwen/Qwen2.5-3B-Instruct"
OUTPUT_DIR  = "./mid_lora_adapter"
MAX_SEQ_LEN = 768
EPOCHS      = 5
BATCH_SIZE  = 1
GRAD_ACCUM  = 8
LR          = 2e-4

print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

print("Loading model in bf16...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,  # bf16 — more stable than fp16, no scaler needed
    device_map="auto",
    trust_remote_code=True,
)
model.config.use_cache = False
model.gradient_checkpointing_enable()
model.enable_input_require_grads()

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    bias="none",
)
model = get_peft_model(model, lora_config, autocast_adapter_dtype=False)
model.print_trainable_parameters()

def format_example(example):
    messages = [
        {"role": "system",    "content": example["instruction"]},
        {"role": "user",      "content": example["input"]},
        {"role": "assistant", "content": example["output"]},
    ]
    return {"text": tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )}

print("\nLoading dataset...")
ds = load_dataset("json", data_files={
    "train":      "train.jsonl",
    "validation": "val.jsonl",
})
ds = ds.map(format_example)
print(f"Train: {len(ds['train'])}  Val: {len(ds['validation'])}")

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=ds["train"],
    eval_dataset=ds["validation"],
    args=SFTConfig(
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LEN,
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        gradient_checkpointing=True,
        warmup_steps=10,
        learning_rate=LR,
        fp16=False,
        bf16=True,   # stable mixed precision, no scaler
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        optim="adamw_torch",
        dataloader_pin_memory=False,
        seed=42,
        report_to="none",
    ),
)

print("\nStarting training...")
trainer.train()

print(f"\nSaving adapter to {OUTPUT_DIR}...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("Done.")