import torch
from peft import PeftModel
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained("./adapter")

config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

base = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=config,
    device_map="auto",
    dtype=torch.float16
)

model = PeftModel.from_pretrained(base, "./adapter")
model.eval()

system = (
    "You are an assistant that converts meeting requests "
    "into structured JSON. Return only valid JSON with "
    "keys: action, person, date, time. "
    "Do not include explanations or markdown."
)

while True:
    user_input = input("\nMeeting request (or exit): ")

    if user_input.lower() == "exit":
        break

    messages = [
        {"role": "system", "content": system},
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

    result = output[0][inputs["input_ids"].shape[1]:]

    print(tokenizer.decode(
        result, skip_special_tokens=True
    ))
