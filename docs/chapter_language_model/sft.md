# Supervised Fine-Tuning

<a href="https://colab.research.google.com/github/junwei-lu/ai4med/blob/main/codes/nlp/llm_model2pretrain2ft.ipynb#scrollTo=nlp_sec5_sft" target="_blank" style="display: inline-flex; align-items: center; gap: 6px; padding: 6px 12px; background: linear-gradient(135deg, #1565c0 0%, #42a5f5 100%); color: white; border-radius: 6px; text-decoration: none; font-size: 0.85em; font-weight: 600;">▶ Try in Colab</a>

This lecture covers the topic on supervised fine-tuning (SFT) through a minimal, reliable workflow to fine-tune an open LLM using Hugging Face tools.

You will learn to:
- Understand the SFT loss function and how it differs from raw pre-training
- Install the right libraries and pick a manageable model
- Prepare a beginner-friendly dataset and template
- Set up a supervised fine-tuning trainer and run training



![SFT Pipeline](ft.assets/sft_plot.png)

## The SFT Loss Function

### Starting point: NTP loss

Recall from the [Next-Token Prediction](ntp.md) tutorial that pre-training minimizes:

$$
\mathcal{L}_{\text{NTP}}(\theta) = -\frac{1}{T} \sum_{t=1}^{T} \log P_\theta(x_t \mid x_1, \ldots, x_{t-1})
$$

over all tokens in the corpus. **Every** token—whether a system prompt, user question, or assistant answer—contributes equally to the loss.

### SFT: only learn from the response

In supervised fine-tuning, the training data consists of **input-output pairs**:

$$
\mathcal{D} = \{(x^{(i)}, y^{(i)})\}_{i=1}^{N}
$$

where $x^{(i)}$ is the **prompt** (system instruction + user turn) and $ y^{(i)}$ is the **target response** (assistant turn). The goal is to teach the model to produce $ y $given$ x $.

The SFT loss is the NTP loss computed **only on the response tokens**, with the prompt tokens masked out:

$$
\boxed{
\mathcal{L}_{\text{SFT}}(\theta) = -\frac{1}{N} \sum_{i=1}^{N} \frac{1}{|y^{(i)}|} \sum_{t=1}^{|y^{(i)}|} \log P_\theta\!\left(y_t^{(i)} \;\middle|\; x^{(i)}, y_1^{(i)}, \ldots, y_{t-1}^{(i)}\right)
}
$$

**Why mask the prompt?**  
- Prompt tokens are given as context, not generated—penalizing the model for not predicting them would confuse training
- Masking focuses capacity on learning the *style* and *content* of the target response
- It also allows much longer prompts without inflating the loss denominator

<!-- 
### Masking in practice

The `SFTTrainer` implements this via a label mask: positions corresponding to the prompt are set to `-100`, and PyTorch's `cross_entropy` ignores positions with label `-100`.

```
Full sequence:  [SYS] You are a clinical assistant.  [USER] What is the eGFR?  [ASST] 45 mL/min
                ↑________________________ prompt _____________________↑  ↑____ response ____↑
Label mask:         -100    -100    ...      -100       -100   -100  ...   45    mL   /   min
Loss computed:      ✗       ✗                ✗          ✗      ✗           ✓    ✓    ✓    ✓
``` -->

### SFT as maximum likelihood estimation

Maximizing the log-likelihood of responses is equivalent to maximum likelihood estimation (MLE) of the conditional distribution:

$$
\theta^* = \arg\max_\theta \sum_{i=1}^{N} \log P_\theta(y^{(i)} \mid x^{(i)})
$$

Each response $y^{(i)}$ factorizes autoregressively, so:

$$
\log P_\theta(y^{(i)} \mid x^{(i)}) = \sum_{t=1}^{|y^{(i)}|} \log P_\theta\!\left(y_t^{(i)} \mid x^{(i)}, y_{<t}^{(i)}\right)
$$

which is exactly the inner sum in the SFT loss.

## SFT Training Guidelines

Next, we will cover how to implement the SFT in practice.

### Choose a model

For a first end-to-end SFT run, it is usually better to choose a **small instruct model** that you can fine-tune in the simplest possible way. Here we use `Qwen/Qwen2.5-1.5B-Instruct` and load it **without LoRA or quantization** so the training recipe stays close to the underlying math.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Qwen/Qwen2.5-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
```



## Instruction Dataset

Instruction data is what turns a base language model from a **text continuer** into a **task-following assistant**. Pre-training teaches general language patterns, but SFT teaches the model what kinds of prompts it should respond to, what output format to follow, and what style of answer is desired.

In practice, several instruction-data templates are common:

- **Alpaca-style triples**: `instruction`, optional `input`, and `output` ([Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca))
- **Chat-style messages**: a list of `{role, content}` turns, compatible with Hugging Face [chat templates](https://huggingface.co/docs/transformers/chat_templating)
- **ShareGPT-style conversations**: multi-turn chat records, often converted into the same `messages` schema before training

For this lecture, we use the chat-style `messages` template because it maps directly to modern instruct models and to Hugging Face tooling.

Chat-style messages with a system instruction, a user prompt, and an assistant answer. This mirrors how you would prepare clinical data (e.g., MedCalc-style questions with ground-truth answers), but here we keep a simple schema for clarity.

Data format (JSONL), each line is one record corresponding to one $(x^{(i)}, y^{(i)})$ pair:

```json
{"messages": [
  {"role": "system",    "content": "You are a clinical calculator assistant."},
  {"role": "user",      "content": "Patient Note: ...\nQuestion: ...\nAnswer:"},
  {"role": "assistant", "content": "95 mL/min"}
]}
```

The **prompt** $x^{(i)}$ consists of the `system` + `user` turns. The **response** $ y^{(i)}$ is the `assistant` turn. Loss is computed only on `assistant` tokens.

Create a tiny demo dataset programmatically (replace with your real data later):

```python
import json

# Minimal demo dataset in OpenAI "messages" format
# Each record = one (x, y) pair; loss is computed on assistant turn only
train_records = [
    {
        "messages": [
            {"role": "system",    "content": "You are a clinical calculator assistant."},
            {"role": "user",      "content": "Patient Note: 16-year-old female with severe hypertension...\nQuestion: Compute creatinine clearance (Cockcroft-Gault).\nAnswer:"},
            {"role": "assistant", "content": "95 mL/min"}  # ← SFT loss computed here
        ]
    },
    {
        "messages": [
            {"role": "system",    "content": "You are a clinical calculator assistant."},
            {"role": "user",      "content": "Patient Note: BMI example.\nQuestion: Height 1.75m, Weight 70kg.\nAnswer:"},
            {"role": "assistant", "content": "22.86"}  # ← SFT loss computed here
        ]
    }
]

# Write as JSONL (one JSON object per line), which HF datasets can read efficiently
with open("train_dataset.json", "w") as f:
    for r in train_records:
        f.write(json.dumps(r) + "\n")
```

**Why this template?**
- A consistent structure simplifies tokenization and training
- The final answer is clear and easy to evaluate later



## Training Prompt Examples

You should find out the proper SFT tasks for your own model and dataset. Here we provide some ideas you may use for SFT.

The prompt $x^{(i)}$ fed to the model during SFT can take many shapes depending on your task. The only requirement is that the format is **consistent across all training examples** so the model can learn the pattern. Below are representative templates for common biomedical and general NLP tasks.

---

### 1 · Closed-Form Q&A

The most common pattern: a short factual question with a single correct answer.

```json
{"messages": [
  {"role": "system",    "content": "You are a medical knowledge assistant. Answer concisely."},
  {"role": "user",      "content": "What is the first-line treatment for type 2 diabetes?"},
  {"role": "assistant", "content": "Metformin, combined with lifestyle modification (diet and exercise)."}
]}
```

Key design choices:

- The system prompt sets the persona and answer style ("concisely")
- The user turn contains *only* the question—no scaffolding text
- The assistant answer is the single ground-truth string the model must learn to reproduce

---

### 2 · Clinical Calculator / Numeric Reasoning

For tasks where the answer is a number derived from patient data, include all necessary values in the note.

```json
{"messages": [
  {"role": "system",    "content": "You are a clinical calculator. Return only the numeric result with units."},
  {"role": "user",      "content": "Patient: 65-year-old male, weight 80 kg, serum creatinine 1.2 mg/dL.\nCompute creatinine clearance using the Cockcroft-Gault equation.\nAnswer:"},
  {"role": "assistant", "content": "64 mL/min"}
]}
```

The trailing `Answer:` in the user turn is a **prompt cue**—it primes the model to emit the answer token immediately, reducing the chance it starts with preamble text.

---

### 3 · Multi-Choice / Classification

Frame the choices explicitly in the prompt so the model learns to pick exactly one option.

```json
{"messages": [
  {"role": "system",    "content": "You are a clinical decision support assistant. Choose the single best answer."},
  {"role": "user",      "content": "A 45-year-old woman presents with pleuritic chest pain, dyspnea, and a positive D-dimer. Which diagnosis is most likely?\nA. Pneumonia\nB. Pulmonary embolism\nC. Myocardial infarction\nD. Pneumothorax\nAnswer:"},
  {"role": "assistant", "content": "B. Pulmonary embolism"}
]}
```

Restrict the assistant turn to the letter + label (not an explanation) if your goal is accurate classification. Add reasoning only if chain-of-thought is required.

---

### 4 · Summarization

Teach the model to compress long clinical text into a structured summary.

```json
{"messages": [
  {"role": "system",    "content": "You are a clinical documentation assistant. Summarize the discharge note in 2–3 sentences covering diagnosis, treatment, and follow-up."},
  {"role": "user",      "content": "Discharge Note:\nThe patient is a 72-year-old male admitted for acute decompensated heart failure with bilateral lower extremity edema and shortness of breath on exertion. He was treated with IV furosemide with marked improvement in fluid status. Echocardiography showed an ejection fraction of 35%. He was discharged on oral furosemide 40 mg daily, lisinopril 10 mg daily, and carvedilol 6.25 mg twice daily with close outpatient follow-up in 1 week.\n\nSummary:"},
  {"role": "assistant", "content": "A 72-year-old male was admitted for acute decompensated heart failure (EF 35%) and responded well to IV diuresis. He was discharged on oral furosemide, lisinopril, and carvedilol. Follow-up is scheduled in 1 week."}
]}
```

---

### 5 · Information / Entity Extraction

Instruct the model to extract structured fields from unstructured text.

```json
{"messages": [
  {"role": "system",    "content": "Extract the requested clinical fields from the note. Return valid JSON only."},
  {"role": "user",      "content": "Note: Patient is a 58-year-old female with a history of hypertension and type 2 diabetes, currently on metformin 1000 mg BID and amlodipine 5 mg daily. She presents with a 3-day history of productive cough and fever (38.4°C).\n\nExtract: {\"age\": ..., \"sex\": ..., \"conditions\": [...], \"medications\": [...], \"presenting_symptoms\": [...]}"},
  {"role": "assistant", "content": "{\"age\": 58, \"sex\": \"female\", \"conditions\": [\"hypertension\", \"type 2 diabetes\"], \"medications\": [\"metformin 1000 mg BID\", \"amlodipine 5 mg daily\"], \"presenting_symptoms\": [\"productive cough\", \"fever\"]}"}
]}
```

Enforcing JSON output in the assistant turn makes the results machine-readable and easy to evaluate.

---

### 6 · Chain-of-Thought (Step-by-Step) Reasoning

For complex reasoning tasks, the assistant turn should include the reasoning steps, not just the final answer.

```json
{"messages": [
  {"role": "system",    "content": "You are a clinical reasoning assistant. Think step by step before giving your final answer."},
  {"role": "user",      "content": "A patient with a serum sodium of 118 mEq/L is started on hypertonic saline. What is the maximum safe correction rate, and why?\nAnswer:"},
  {"role": "assistant", "content": "Step 1: Identify the risk. Rapid correction of chronic hyponatremia can cause osmotic demyelination syndrome (ODS).\nStep 2: Apply the guideline limit. The maximum safe rate is 8–10 mEq/L per 24 hours (some guidelines cite 10–12 mEq/L for the first 24 h).\nStep 3: Conclusion. Correct at no more than 8 mEq/L/24 h in this patient to avoid ODS; check serum sodium every 2–4 hours."}
]}
```

!!! tip "Consistency is key"
    Mix-and-match formats cause the model to learn *when* to reason vs. when to answer directly—which is powerful—but only if the distinction is intentional and consistently labelled in your training data. If all tasks are the same type, use a single unified template.

---

### Prompt Design Checklist

| Element | Recommendation |
|---|---|
| System prompt | One sentence: role + output style |
| User turn | Task description + all necessary inputs |
| Trailing cue | End with `Answer:` or `Output:` to cue generation |
| Assistant turn | Ground-truth response only—no meta-commentary |
| Format consistency | Same template for every example in the dataset |
| Output type | Match the assistant turn to evaluation metric (string, JSON, letter, etc.) |


<!-- 
## Verifying the Masking Manually

Before training, it is instructive to verify that the trainer correctly masks prompt tokens. Here is a minimal check:

```python
from transformers import AutoTokenizer
from trl import setup_chat_format
import torch

model_id = "meta-llama/Meta-Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_id)

messages = [
    {"role": "system",    "content": "You are a clinical calculator assistant."},
    {"role": "user",      "content": "Compute creatinine clearance.\nAnswer:"},
    {"role": "assistant", "content": "95 mL/min"},
]

# Apply chat template; return tensors as token ids
full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
full_ids = tokenizer(full_text, return_tensors="pt").input_ids[0]

# Identify where the assistant turn starts
prompt_text = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
prompt_ids = tokenizer(prompt_text, return_tensors="pt").input_ids[0]
prompt_len = len(prompt_ids)

# Build labels: -100 for prompt positions, actual ids for response positions
labels = torch.full_like(full_ids, -100)
labels[prompt_len:] = full_ids[prompt_len:]

print("Prompt tokens (masked):", full_ids[:prompt_len])
print("Response tokens (loss):", full_ids[prompt_len:])
print("Labels:", labels)
# -100 positions are ignored; loss is only on response tokens
``` -->



## Hugging Face SFT Training Guidelines

Hugging Face [TRL](https://huggingface.co/docs/trl/index) is the main high-level library for post-training LLMs. It provides specialized trainers such as `SFTTrainer`, `DPOTrainer`, and `GRPOTrainer`, so you can focus on data format and objective design instead of rewriting training loops.

We use TRL's `SFTTrainer` for simplicity. Under the hood it:

1. Applies the chat template to flatten messages into a single string
2. Tokenizes the full sequence
3. Builds the label mask (sets prompt tokens to `-100`)
4. Computes the SFT loss: $\mathcal{L}_{\text{SFT}} = -\frac{1}{|y|}\sum_{t} \log P_\theta(y_t \mid x, y_{<t})$

```python
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer, setup_chat_format

# Ensure chat formatting and special tokens are set (adds tokens and a chat template)
model, tokenizer = setup_chat_format(model, tokenizer)

# Load the JSONL dataset from disk
train_ds = load_dataset("json", data_files="train_dataset.json", split="train")

# Basic training config; start small for quick feedback, then scale
args = TrainingArguments(
    output_dir="llama3-8b-basics-sft",   # where checkpoints/logs are saved
    num_train_epochs=1,                   # try 1 epoch first to verify pipeline
    per_device_train_batch_size=1,        # small batch to fit in memory
    gradient_accumulation_steps=8,        # effective batch size = 1 × 8
    gradient_checkpointing=True,          # trade compute for lower memory
    learning_rate=2e-5,                   # smaller LR is typical for full-model SFT
    bf16=True,                            # use bfloat16 on supported GPUs (e.g., A100/4090)
    tf32=True,                            # faster matmul on Ampere+
    logging_steps=10,                     # log every N steps
    save_strategy="epoch",                # save at end of epoch
    report_to="none",                     # set to "tensorboard" if you want TB logs
)

# SFTTrainer handles message formatting and prompt masking
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_ds,
    args=args,
    max_seq_length=2048,       # truncate/pack sequences to this length
    packing=True,              # pack multiple short samples in one sequence
    dataset_kwargs={"add_special_tokens": False, "append_concat_token": False},
)

# Run training; loss logged = SFT loss averaged over response tokens
trainer.train()
trainer.save_model()
```

**Tips for biomedical data:**
- Keep prompts short and focused: patient note + question; end with "Answer:"
- If answers are numeric (e.g., mL/min), use a consistent unit and precision
- Start with a few hundred curated examples, then scale up



### Quick inference

```python
def generate(prompt):
    # Tokenize the prompt and move tensors to the model's device (CPU/GPU)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    # Disable gradients for faster, memory-efficient inference
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=100)
    # Convert token ids back to text, skipping special tokens
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example: ask for a numeric clinical answer, consistent with training template
user_q = (
    "Patient Note: 16-year-old female with severe hypertension...\n"
    "Question: Compute creatinine clearance (Cockcroft-Gault).\n"
    "Answer:"
)
print(generate(f"You are a clinical calculator assistant.\n\n{user_q}"))
```



## What to try next

- Add evaluation: compare model outputs against ground truth answers
- Expand dataset with more clinical calculators (e.g., BMI, eGFR)
- Use curriculum: start with simple tasks, then harder ones
- Consider GRPO (see [RL Fine-Tuning](grpo.md)) if you want to optimize non-differentiable rewards


## References

- Ouyang et al., [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)
- Hugging Face, [TRL documentation](https://huggingface.co/docs/trl/index)
- Hugging Face, [Chat templates](https://huggingface.co/docs/transformers/chat_templating)
