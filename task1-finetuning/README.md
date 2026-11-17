# Task 1 — QLoRA Fine-tuning

## Objective
Fine-tune Qwen2.5-0.5B-Instruct to convert natural-language
meeting requests into structured JSON.

## Model
Qwen/Qwen2.5-0.5B-Instruct

## Method
QLoRA with 4-bit NF4 quantization.

## LoRA configuration
- Rank: 16
- Alpha: 32
- Dropout: 0.05
- Target modules: q_proj, k_proj, v_proj, o_proj,
  gate_proj, up_proj, down_proj
- Learning rate: 0.0002
- Epochs: 3
- Training examples: 500
- Validation examples: 100
- Held-out test examples: 100

## Experimental results

| Evaluation | Base | Fine-tuned |
|---|---:|---:|
| JSON validity | 100% | 100% |
| Exact match | 0% | 100% |

Additional evaluations:
- Challenge set: 20/20 (100%)
- Hard test set: 6/10 (60%)

## Failure analysis

1. "quarter past nine" predicted 09:00 instead of 09:15.
2. 7:15 AM predicted 15:15 instead of 07:15.
3. 8:45 PM predicted 09:45 instead of 20:45.
4. 10:30 AM predicted 10:30:00 instead of 10:30.

## Training performance
- Runtime: 213.17 seconds
- Trainable parameters: 8,798,208
- Total parameters: 502,830,976
- Final logged validation loss: approximately 0.0000437

## Limitations
The primary datasets were synthetically generated using a small
number of templates. The high held-out accuracy therefore does
not establish broad real-world generalisation. The hard test
exposed weaknesses in time expression interpretation and
format consistency.

## Running

Install dependencies:

    pip install -r requirements.txt

Train:

    python train.py

Evaluate:

    python evaluate.py

Interactive inference:

    python inference.py

Training requires a compatible CUDA GPU.

## Directory structure

- dataset/: training, validation and test examples
- results/: recorded experimental results and predictions
- adapter/: trained LoRA adapter
- train.py: training pipeline
- evaluate.py: evaluation pipeline
- inference.py: interactive inference
