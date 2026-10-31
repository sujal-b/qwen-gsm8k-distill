"""
Core Essential Snippets for Qwen3.5-0.8B GSM8K Distillation Showcase
"""

import torch
from dataclasses import dataclass
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel,
)

SYSTEM_PROMPT = (
    "You are a meticulous math tutor. Solve the problem step by step, "
    "ending with '#### <answer>'."
)

# ==============================================================================
# Snippet 1: Prompt Templating & Label Masking (-100)
# ==============================================================================
def encode_example(ex: dict, tokenizer, max_seq_len: int = 768):
    """
    Constructs chat template with system prompt and user problem, 
    then masks the prompt tokens with -100 so CrossEntropyLoss is 
    calculated strictly on the assistant's reasoning and answer.
    """
    question = str(ex.get("question", "")).strip()
    answer = str(ex.get("teacher_solution", "")).strip()
    if not question or not answer:
        return None

    prompt_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Problem: {question}\n\nSolve it step by step."},
    ]

    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    full_text = prompt_text + answer + (tokenizer.eos_token or "")

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(
        full_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_seq_len,
    )["input_ids"]

    prompt_len = len(prompt_ids)
    if prompt_len >= len(full_ids):
        return None

    # Mask prompt with -100 so loss is computed solely on teacher response
    labels = [-100] * prompt_len + full_ids[prompt_len:]

    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
        "length": len(full_ids),
    }


# ==============================================================================
# Snippet 2: Dynamic Collator with Tensor-Core Alignment
# ==============================================================================
@dataclass
class CausalLMCollator:
    pad_token_id: int
    pad_to_multiple_of: int = 8  # Hardware optimization for NVIDIA Tensor Cores

    def __call__(self, features):
        max_len = max(len(f["input_ids"]) for f in features)
        if self.pad_to_multiple_of and (rem := max_len % self.pad_to_multiple_of):
            max_len += self.pad_to_multiple_of - rem

        input_ids, attention_mask, labels = [], [], []
        for f in features:
            pad_len = max_len - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [self.pad_token_id] * pad_len)
            attention_mask.append(f["attention_mask"] + [0] * pad_len)
            labels.append(f["labels"] + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


# ==============================================================================
# Snippet 3: 4-Bit NormalFloat Quantization & LoRA Adapter Setup
# ==============================================================================
def setup_qlora_model(model_name: str = "Qwen/Qwen3.5-0.8B"):
    # 4-bit NormalFloat4 with double quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    # Gradient checkpointing reduces memory during backward pass
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.config.use_cache = False

    # LoRA applied across all linear layers
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    return get_peft_model(model, lora_config)


# ==============================================================================
# Snippet 4: Training Arguments (Batching, Optimization & LR Schedule)
# ==============================================================================
def get_training_args(output_dir: str = "qlora_stage1", lr: float = 1e-4, epochs: float = 1.0):
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,   # Effective batch: 2 * 4 * 2 GPUs = 16
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        optim="paged_adamw_8bit",        # 8-bit optimizer states paged to CPU when needed
        fp16=True,
        gradient_checkpointing=True,
        group_by_length=True,            # Minimizes padding within batches
        logging_steps=5,
        save_steps=100,
        save_total_limit=3,
    )


# ==============================================================================
# Snippet 5: Zero-Overhead Adapter Fusion (Merge and Unload)
# ==============================================================================
def merge_and_export(base_model_name: str, adapter_path: str, output_path: str):
    """
    Fuses low-rank matrices W = W_base + (alpha/r)*B*A to eliminate inference latency.
    """
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base_model, adapter_path, device_map="cpu")
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(output_path, safe_serialization=True)
    
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    tokenizer.save_pretrained(output_path)
    print(f"Model successfully merged into: {output_path}")
