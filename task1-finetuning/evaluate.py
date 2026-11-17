import json
import torch
from tqdm import tqdm
from peft import PeftModel
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

SYSTEM_PROMPT = (
    "You are an assistant that converts meeting requests "
    "into structured JSON. Return only valid JSON with "
    "keys: action, person, date, time. "
    "Do not include explanations or markdown."
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=config,
    device_map="auto",
    dtype=torch.float16
)

model = PeftModel.from_pretrained(
    base_model, "./adapter"
)
model.eval()

def generate_response(user_input):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_input}
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        prompt, return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )

    new_tokens = output[0][inputs["input_ids"].shape[1]:]

    return tokenizer.decode(
        new_tokens, skip_special_tokens=True
    ).strip()

with open("dataset/test.json") as f:
    examples = json.load(f)

predictions = []
valid = 0
correct = 0

for example in tqdm(examples):
    prediction = generate_response(example["input"])
    predictions.append(prediction)

    try:
        parsed = json.loads(prediction)
        valid += 1

        if parsed == json.loads(example["output"]):
            correct += 1

    except json.JSONDecodeError:
        pass

results = {
    "total": len(examples),
    "valid_json": valid,
    "valid_json_rate": valid / len(examples) * 100,
    "exact_matches": correct,
    "exact_match_accuracy": correct / len(examples) * 100
}

print(results)

with open("evaluation_results.json", "w") as f:
    json.dump(results, f, indent=2)

with open("predictions.json", "w") as f:
    json.dump(predictions, f, indent=2)
