
import time
import json
import csv
import threading
import subprocess
import requests
import psutil
import os
from datetime import datetime

MODELS = {
    "Q4_K_M": "qwen2.5:0.5b",
    "Q8_0": "qwen2.5:0.5b-instruct-q8_0"
}

PROMPTS = [
    "Explain machine learning in simple terms.",
    "What is the difference between AI and machine learning?",
    "Explain overfitting and how to prevent it.",
    "What is a neural network?",
    "Explain supervised and unsupervised learning.",
    "What is gradient descent?",
    "Explain the bias-variance tradeoff.",
    "What is natural language processing?",
    "Explain how transformers work at a high level.",
    "What are the advantages and disadvantages of deep learning?"
]

API_URL = "http://localhost:11434/api/generate"
NUM_PREDICT = 150

process = psutil.Process(os.getpid())
monitoring = False
samples = []
lock = threading.Lock()


def get_gpu_memory():
    try:
        result = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits"
            ],
            text=True,
            stderr=subprocess.DEVNULL
        )
        return int(result.strip().splitlines()[0])
    except Exception:
        return None


def monitor_memory():
    while monitoring:
        ram = psutil.virtual_memory().used / (1024 ** 2)
        gpu = get_gpu_memory()

        with lock:
            samples.append({
                "ram_mb": ram,
                "gpu_mb": gpu
            })

        time.sleep(0.25)


def run_prompt(model, prompt):
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": {
            "temperature": 0,
            "seed": 42,
            "num_predict": NUM_PREDICT,
            "num_ctx": 4096
        }
    }

    start = time.perf_counter()
    token_count = 0
    response_text = ""
    first_token_time = None

    with requests.post(
        API_URL,
        json=payload,
        stream=True,
        timeout=300
    ) as response:
        response.raise_for_status()

        for line in response.iter_lines():
            if not line:
                continue

            data = json.loads(line)

            if data.get("response"):
                if first_token_time is None:
                    first_token_time = time.perf_counter() - start

                response_text += data["response"]

            if data.get("done"):
                token_count = data.get("eval_count", 0)
                total_duration = data.get("total_duration", 0) / 1e9
                eval_duration = data.get("eval_duration", 0) / 1e9
                break

    elapsed = time.perf_counter() - start
    tokens_per_second = (
        token_count / eval_duration if eval_duration > 0 else 0
    )

    return {
        "tokens": token_count,
        "elapsed_seconds": round(elapsed, 3),
        "generation_seconds": round(eval_duration, 3),
        "tokens_per_second": round(tokens_per_second, 2),
        "first_token_seconds": round(first_token_time or 0, 3),
        "response": response_text
    }


def main():
    all_results = []

    for config, model in MODELS.items():
        print(f"\n{'=' * 60}")
        print(f"Testing {config}: {model}")
        print(f"{'=' * 60}")

        print("Loading model and warming up...")
        run_prompt(model, "Say hello in one sentence.")

        global monitoring, samples
        samples = []
        monitoring = True

        monitor = threading.Thread(
            target=monitor_memory,
            daemon=True
        )
        monitor.start()

        config_results = []

        for index, prompt in enumerate(PROMPTS, start=1):
            print(f"[{index}/{len(PROMPTS)}] {prompt}")

            result = run_prompt(model, prompt)

            row = {
                "configuration": config,
                "model": model,
                "prompt_number": index,
                "prompt": prompt,
                "tokens": result["tokens"],
                "elapsed_seconds": result["elapsed_seconds"],
                "generation_seconds": result["generation_seconds"],
                "tokens_per_second": result["tokens_per_second"],
                "first_token_seconds": result["first_token_seconds"],
                "response": result["response"]
            }

            config_results.append(row)
            all_results.append(row)

            print(
                f"  {result['tokens_per_second']} tokens/sec | "
                f"{result['tokens']} tokens"
            )

        monitoring = False
        monitor.join(timeout=2)

        with lock:
            config_samples = list(samples)

        ram_values = [x["ram_mb"] for x in config_samples]
        gpu_values = [
            x["gpu_mb"] for x in config_samples
            if x["gpu_mb"] is not None
        ]

        avg_speed = sum(
            x["tokens_per_second"] for x in config_results
        ) / len(config_results)

        print(f"\n{config} results:")
        print(f"Average generation speed: {avg_speed:.2f} tokens/sec")
        print(
            f"Peak system RAM: "
            f"{max(ram_values):.1f} MB"
            if ram_values else "RAM data unavailable"
        )
        print(
            f"Peak total GPU memory: "
            f"{max(gpu_values)} MB"
            if gpu_values else "GPU data unavailable"
        )

        with open(
            f"{config}_memory_samples.json",
            "w",
            encoding="utf-8"
        ) as file:
            json.dump(config_samples, file, indent=2)

    with open(
        "benchmark_results.csv",
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=all_results[0].keys()
        )
        writer.writeheader()
        writer.writerows(all_results)

    print("\nBenchmark completed.")
    print("Saved: benchmark_results.csv")
    print("Saved: Q4_K_M_memory_samples.json")
    print("Saved: Q8_0_memory_samples.json")


if __name__ == "__main__":
    main()
