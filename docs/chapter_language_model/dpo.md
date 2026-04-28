# Direct Preference Optimization

<a href="https://colab.research.google.com/github/junwei-lu/ai4med/blob/main/codes/nlp/llm_model2pretrain2ft.ipynb#scrollTo=nlp_sec6_dpo" target="_blank" style="display: inline-flex; align-items: center; gap: 6px; padding: 6px 12px; background: linear-gradient(135deg, #1565c0 0%, #42a5f5 100%); color: white; border-radius: 6px; text-decoration: none; font-size: 0.85em; font-weight: 600;">▶ Try in Colab</a>

This lecture introduces **Direct Preference Optimization (DPO)**, a stable and efficient method to align Large Language Models (LLMs) with human preferences. Unlike traditional Reinforcement Learning from Human Feedback (RLHF), DPO does not require training a separate reward model or using complex RL algorithms like PPO.

The main motivation is practical: in many domains, **pairwise preference is easier to collect than absolute grading**. A clinician may find it hard to assign a precise score to a response on a 1-10 scale, but it is often easy to say **response A is safer than response B**. Likewise, for summarization or bedside advice, comparing two answers is often easier than writing the perfect answer from scratch.

You will learn to:

- Understand the mathematical derivation of DPO
- Format preference data (chosen vs. rejected responses)
- Run DPO training using the `trl` library

## Why DPO?

Standard RLHF involves a complex three-step process:

1.  **SFT**: Supervised fine-tuning on high-quality data.
2.  **Reward Modeling**: Training a separate model to predict human preference scores.
3.  **RL**: Using PPO to optimize the policy against the reward model.

DPO simplifies this by optimizing the policy **directly** on preference data. It mathematically eliminates the need for an explicit reward model, treating the language model itself as an implicit reward model. This makes training more stable and less memory-intensive.

## The Math of DPO

### From RL to DPO

The standard RLHF objective maximizes the expected reward $r(x, y)$ while keeping the new policy $\pi_\theta$ close to the reference model $\pi_{\text{ref}}$ (usually the SFT model) to prevent mode collapse. The objective is:

$$
\max_{\pi_\theta} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(y|x)} \left[ r(x, y) - \beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)} \right]
$$

where $\beta$ is a regularization parameter controlling the deviation from the reference model.

It can be shown analytically that the **optimal policy** $\pi^*$ for this objective takes the form:

$$
\pi^*(y|x) \propto \pi_{\text{ref}}(y|x) \exp\left( \frac{1}{\beta} r(x, y) \right)
$$

DPO rearranges this equation to express the reward function in terms of the optimal policy and reference policy:

$$
r(x, y) = \beta \log \frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} + Z(x)
$$

### The DPO Loss

We assume human preferences follow the **Bradley-Terry model**, where the probability that a human prefers response $y_w$ (winner/chosen) over $y_l$ (loser/rejected) depends on the difference in their rewards:

$$
P(y_w \succ y_l \mid x) = \sigma(r(x, y_w) - r(x, y_l))
$$

Substituting the re-parameterized reward function into this preference model eliminates the reward term $r(x, y)$ entirely. We get the **DPO loss function**:

$$
\boxed{
\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right]
}
$$

Intuitively, this loss:

- **Increases** the likelihood of the chosen response $y_w$ relative to the reference model.
- **Decreases** the likelihood of the rejected response $y_l$ relative to the reference model.
- The parameter $\beta$ controls how strongly we enforce the KL-divergence constraint (typically $0.1$ to $0.5$).

## Preference Dataset

DPO requires a dataset of **triplets**:

1.  **Prompt** ($x$): The user instruction.
2.  **Chosen** ($y_w$): The better response.
3.  **Rejected** ($y_l$): The worse response.

### Data Format

The standard format is a JSONL file (or Hugging Face dataset) with columns/keys for `prompt`, `chosen`, and `rejected`.

```json
{
  "prompt": "User: How do I treat a mild headache?\nAssistant:",
  "chosen": "You can try resting in a quiet, dark room, staying hydrated, or taking over-the-counter pain relievers like acetaminophen or ibuprofen.",
  "rejected": "Go to the emergency room immediately."
}
```

Alternatively, using the conversational format (compatible with `DPOTrainer`):

```json
{
    "chosen": [
        {"role": "user", "content": "How do I treat a mild headache?"},
        {"role": "assistant", "content": "You can try resting in a quiet, dark room, staying hydrated, or taking over-the-counter pain relievers like acetaminophen or ibuprofen."}
    ],
    "rejected": [
        {"role": "user", "content": "How do I treat a mild headache?"},
        {"role": "assistant", "content": "Go to the emergency room immediately."}
    ]
}
```

### Example: Clinical Preference

In a medical context, preferences often favor **safety, accuracy, and empathy**.

```python
import json

data = [
    {
        "chosen": [
            {"role": "system", "content": "You are a helpful medical assistant."},
            {"role": "user", "content": "What is the dosage for Lisinopril?"},
            {"role": "assistant", "content": "The dosage of Lisinopril varies based on the condition being treated. For hypertension, it typically starts at 10 mg once daily. However, you should consult a doctor for a specific prescription."}
        ],
        "rejected": [
            {"role": "system", "content": "You are a helpful medical assistant."},
            {"role": "user", "content": "What is the dosage for Lisinopril?"},
            {"role": "assistant", "content": "Take 10mg. It's fine for everyone."}
        ]
    }
]

with open("preference_data.json", "w") as f:
    for entry in data:
        json.dump(entry, f)
        f.write("\n")
```

## Training with TRL

We use the `DPOTrainer` from the [TRL library](https://huggingface.co/docs/trl/index). It handles the data processing, reference model management, and loss computation.

### Setup

You need to load the model you want to train. In DPO, this model serves as **both** the policy being optimized $\pi_\theta$ **and** the initial reference model $\pi_{\text{ref}}$ (a copy is created internally or loaded separately).

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig
from datasets import load_dataset

# 1. Load the base model (usually an SFT model)
model_id = "Qwen/Qwen2.5-1.5B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    use_cache=False
)
model.config.use_cache = False

tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 2. Load the dataset
dataset = load_dataset("json", data_files="preference_data.json", split="train")

# 3. Define DPO configuration
training_args = DPOConfig(
    output_dir="./dpo_results",
    beta=0.1,                       # The regularization parameter (key hyperparam)
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=5e-6,             # DPO usually requires a lower LR than SFT
    logging_steps=10,
    save_steps=100,
    max_length=1024,                # Max length for prompt + response
    max_prompt_length=512,
    remove_unused_columns=False,
)

# 4. Initialize the Trainer
dpo_trainer = DPOTrainer(
    model=model,
    ref_model=None,                 # If None, a copy of 'model' is used as reference
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
)

# 5. Train
dpo_trainer.train()
dpo_trainer.save_model("./dpo_final_model")
```

### Key Parameters

- **`beta`**: Controls the strength of the KL penalty.
    - Higher `beta` (e.g., 0.5): Keeps the model closer to the reference; more stable but less optimization.
    - Lower `beta` (e.g., 0.1): Allows more deviation to maximize preference satisfaction; risk of over-optimization.
- **`learning_rate`**: Typically smaller than SFT (e.g., `5e-6` or `1e-6`).
- **`ref_model`**: Ideally, this is the exact model state before DPO starts. `DPOTrainer` handles this automatically if you pass `ref_model=None`.


## References

- Rafailov et al., [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290)
- Ouyang et al., [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)
- Hugging Face, [TRL documentation](https://huggingface.co/docs/trl/index)
