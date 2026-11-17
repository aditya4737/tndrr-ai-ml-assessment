import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import (
    LoraConfig,
    prepare_model_for_kbit_training,
    get_peft_model
)

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

with open("dataset/train.json") as f:
    train_data = json.load(f)

with open("dataset/validation.json") as f:
    validation_data = json.load(f)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    dtype=torch.float16
)

model.config.use_cache = False
model.config.pad_token_id = tokenizer.pad_token_id

model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

SYSTEM_PROMPT = (
    "You are an assistant that converts meeting requests "
    "into structured JSON. Return only valid JSON with "
    "keys: action, person, date, time. "
    "Do not include explanations or markdown."
)

def tokenize_example(example):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": example["input"]}
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    full_text = prompt + example["output"] + tokenizer.eos_token

    prompt_ids = tokenizer(
        prompt, add_special_tokens=False
    )["input_ids"]

    full_ids = tokenizer(
        full_text,
        add_special_tokens=False,
        truncation=True,
        max_length=256
    )["input_ids"]

    labels = full_ids.copy()
    labels[:min(len(prompt_ids), len(labels))] = (
        [-100] * min(len(prompt_ids), len(labels))
    )

    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels
    }

train_dataset = Dataset.from_list([
    tokenize_example(x) for x in train_data
])

validation_dataset = Dataset.from_list([
    tokenize_example(x) for x in validation_data
])

args = TrainingArguments(
    output_dir="./qwen-lora-results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    warmup_steps=20,
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    fp16=True,
    optim="adamw_torch",
    lr_scheduler_type="cosine",
    weight_decay=0.01,
    report_to="none",
    save_total_limit=2,
    gradient_checkpointing=True,
    seed=42
)

collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    padding=True,
    label_pad_token_id=-100
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    data_collator=collator
)

trainer.train()

trainer.model.save_pretrained("./adapter")
tokenizer.save_pretrained("./adapter")

with open("./training_logs.json", "w") as f:
    json.dump(trainer.state.log_history, f, indent=2)

print("Training completed.")
