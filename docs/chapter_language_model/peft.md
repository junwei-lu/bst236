# Parameter-Efficient Fine-Tuning (PEFT)

<a href="https://colab.research.google.com/github/junwei-lu/ai4med/blob/main/codes/nlp/llm_model2pretrain2ft.ipynb#scrollTo=nlp_sec9_peft" target="_blank" style="display: inline-flex; align-items: center; gap: 6px; padding: 6px 12px; background: linear-gradient(135deg, #1565c0 0%, #42a5f5 100%); color: white; border-radius: 6px; text-decoration: none; font-size: 0.85em; font-weight: 600;">▶ Try in Colab</a>

Large Language Models (LLMs) are expensive to fine-tune end-to-end. Parameter-Efficient Fine-Tuning (PEFT) adapts a pre-trained model by training a small number of additional parameters while keeping the original weights frozen. This lecture focuses on two beginner-friendly PEFT techniques widely used in practice:

- LoRA (Low-Rank Adaptation)
- Quantization (8-bit and 4-bit) and QLoRA (LoRA on a quantized base)

These methods enable fine-tuning models like Llama 3 on a single consumer GPU with limited memory. The examples below use the Hugging Face ecosystem: `transformers`, `datasets`, `peft`, `bitsandbytes`, and `trl`.

## Why PEFT?

- Reduce memory and compute costs by training fewer parameters
- Maintain strong performance by reusing powerful base models
- Enable domain adaptation on modest hardware

Typical use cases in biostatistics and biomedical research:

- Adapting a general LLM to clinical language or calculators
- Enforcing structured outputs (e.g., report templates)
- Reducing hallucinations via supervised examples


## Quantization Before Adapters

Before introducing LoRA, it helps to separate two ideas that are often mixed together:

1. **Quantization** reduces the memory footprint of the **frozen base model**
2. **PEFT adapters** decide **which trainable parameters** you update

Quantization is not the same thing as PEFT, but the two are often combined. In practice, the most common options in the Hugging Face + PEFT workflow are:

| Option | Typical tool | When to use it | Main trade-off |
|--|--|--|--|
| bf16 / fp16 weights | native `transformers` loading | Model already fits in memory | Simplest and most stable |
| 8-bit quantization | `bitsandbytes` (`load_in_8bit=True`) | Moderate memory pressure | Good stability, modest compression |
| 4-bit quantization | `bitsandbytes` (`load_in_4bit=True`) | Severe memory pressure | Best compression, more care needed |
| GPTQ / AWQ style quantization | specialized inference stacks | Mostly inference deployment | Fast inference, less common for training |


![Quantization](./ft.assets/quant.png)

### How to choose

- Choose **full precision / mixed precision** if the model fits and you want the simplest debugging experience
- Choose **8-bit + LoRA** when memory is somewhat tight but you still want a conservative setup
- Choose **4-bit + LoRA (QLoRA)** when the model would otherwise not fit on your GPU
- Prefer **`bfloat16` compute** on newer GPUs; fall back to `float16` on older hardware
- If you add new special tokens or resize embeddings, keep an eye on embedding layers because they may need higher precision

### Minimal loading patterns

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_id = "meta-llama/Meta-Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 8-bit base model
bnb_8bit = BitsAndBytesConfig(load_in_8bit=True)

# 4-bit base model for QLoRA
bnb_4bit = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
```




## LoRA: The Math

### Why low-rank?

A pre-trained Transformer contains large weight matrices $W_0 \in \mathbb{R}^{d \times k}$ (e.g., $d = k = 4096$ for Llama 3). Fine-tuning all parameters requires storing and updating $d \times k \approx 16.8$ million numbers *per layer*.

The key insight of LoRA ([Hu et al., 2021](https://arxiv.org/abs/2106.09685)) is that **the weight updates during fine-tuning have low intrinsic rank**—most of the useful signal lives in a small subspace of parameter space.

### The low-rank decomposition

Instead of updating $W_0$ directly, LoRA adds a **low-rank perturbation**:

$$
W = W_0 + \Delta W = W_0 + B A
$$

where:

| Matrix | Shape | Role |
|--|-||
| $W_0$ | $d \times k$ | **Frozen** pre-trained weight |
| $A$ | $r \times k$ | Trainable; initialized from $\mathcal{N}(0, \sigma^2)$ |
| $B$ | $d \times r$ | Trainable; initialized to **zero** (so $\Delta W = 0$ at start) |
| $r$ | scalar | Rank, $r \ll \min(d, k)$ (e.g., 8–64) |

![LoRA adapter diagram](./ft.assets/lora_diagram.png)

*Figure: The frozen weight $W_0$ passes through unchanged. The adapter path computes $BAx$ with far fewer parameters than $W_0x$.*

### Scaled forward pass

The full forward pass through an adapted layer is:

$$
h = W_0 x + \frac{\alpha}{r} B A x
$$

where $\alpha$ is a scaling hyperparameter (`lora_alpha` in the code). The factor $\alpha / r$ stabilizes training: increasing rank $r$without adjusting$\alpha$ would otherwise amplify the adapter's contribution.

**In practice** $B$ is initialized to zero, so at the start of training $h = W_0 x$—the adapter starts as an identity perturbation and gradually learns the task-specific correction.

### Parameter savings

For a single weight matrix:

| Method | Trainable params |
|--|--|
| Full fine-tuning | $d \times k$ |
| LoRA rank $r$ | $r \times (d + k)$ |

Example: $d = k = 4096$, $r = 16$ → LoRA trains $16 \times 8192 = 131{,}072$ vs. $4096^2 = 16{,}777{,}216$ parameters. That is a **128× reduction** in trainable parameters for this layer.

Across all layers of Llama 3 8B, LoRA (rank 16, all-linear) typically trains ~1–2% of total parameters.

### Why does $B$ start at zero?

If both $A$and$B$were random at initialization, the perturbation$\Delta W = BA$would immediately shift the model away from its well-optimized starting point. Initializing$B = 0$ ensures:

$$
\Delta W \big|_{t=0} = B_0 A_0 = \mathbf{0}
$$

so the fine-tuning starts from the pre-trained model's behavior and only diverges as the task signal accumulates.



## PEFT Package


The [PEFT (Parameter-Efficient Fine-Tuning)](https://github.com/huggingface/peft) package, developed by Hugging Face, provides simple interfaces for applying parameter-efficient methods such as LoRA to large language models. It integrates seamlessly with common model libraries.

The LoRA math maps directly to the `LoraConfig` parameters:

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,               # rank r — capacity of adapter; higher = more expressive
    lora_alpha=32,      # α scaling factor; effective scale = α/r = 2.0
    lora_dropout=0.05,  # dropout on the adapter path for regularization
    bias="none",        # do not train bias terms
    target_modules="all-linear",   # apply ΔW = BA to every linear layer
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# → trainable params: ~20M || all params: ~8B || trainable%: ~0.24%
```

To see the actual adapter matrices inside the model:

```python
# Inspect one LoRA adapter layer
for name, module in model.named_modules():
    if hasattr(module, "lora_A"):
        print(f"Layer: {name}")
        print(f"  A shape (r × k): {module.lora_A['default'].weight.shape}")
        print(f"  B shape (d × r): {module.lora_B['default'].weight.shape}")
        break
```

The forward pass in PEFT's source mirrors the math exactly:

```python
# Conceptual pseudocode matching h = W₀x + (α/r) * B * A * x
result = F.linear(x, self.weight)                      # W₀ x   (frozen)
lora_out = self.lora_B(self.lora_A(self.lora_dropout(x)))  # B(A(x))
result += lora_out * (self.lora_alpha / self.r)        # + (α/r) B A x
```



## Quantization: 8-bit vs 4-bit

Quantization stores model weights in lower precision to reduce memory.

- **8-bit (int8):** Good trade-off of speed and stability; widely used for inference and fine-tuning with LoRA.
- **4-bit (int4):** Maximum compression; combined with LoRA → **QLoRA** for efficient fine-tuning on very limited VRAM.

Loading a causal LM (e.g., Llama 3 8B) with quantization:

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

model_id = "meta-llama/Meta-Llama-3-8B"

# Choose either 8-bit OR 4-bit config
bnb_8bit = BitsAndBytesConfig(load_in_8bit=True)
bnb_4bit = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,   # second quantization for extra compression
    bnb_4bit_quant_type="nf4",        # NormalFloat4: best distribution for LLM weights
    bnb_4bit_compute_dtype=torch.bfloat16,  # compute in bf16 even though weights are 4-bit
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    quantization_config=bnb_4bit,  # or bnb_8bit
    torch_dtype=torch.bfloat16,
)
```

Tips:
- If you encounter numerical instability on older GPUs, try `torch.float16` compute dtype.
- For chat-tuned models, ensure the tokenizer has correct special tokens and chat template.



## QLoRA: LoRA on a 4-bit Base

QLoRA combines quantization and LoRA: the **base model stays in 4-bit** (frozen), while the **LoRA adapter matrices $A$and$B$ are trained in bfloat16**. This is possible because:

1. The quantized weights are **dequantized on-the-fly** during the forward pass to bf16 for matrix multiplication
2. Gradients only flow through the adapter $BA$, never through $W_0$

The math remains identical; only the storage format of $W_0$ changes:

$$
h = \text{dequant}(W_0^{4\text{bit}}) \cdot x + \frac{\alpha}{r} B A x
$$

Practical tips:
- Use `nf4` quant type and bf16 compute where possible
- Enable gradient checkpointing to trade compute for memory
- Consider training embeddings and `lm_head` if you add special tokens/chat templates

Example with `trl.SFTTrainer`:

```python
import torch
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer, setup_chat_format
from peft import LoraConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

model_id = "meta-llama/Meta-Llama-3-8B"

# 4-bit base model — W₀ stored in 4-bit, dequantized to bf16 for compute
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
)

# Ensure chat format if training on conversations
model, tokenizer = setup_chat_format(model, tokenizer)

# LoRA adapters: r=16, α=16 → effective scale α/r = 1.0
peft_config = LoraConfig(
    r=16,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    target_modules="all-linear",
    task_type="CAUSAL_LM",
)

# Example: load a small jsonl dataset in OpenAI messages format
train_data = load_dataset("json", data_files="train_dataset.json", split="train")

args = TrainingArguments(
    output_dir="llama3-8b-qlora-demo",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    learning_rate=2e-4,
    logging_steps=10,
    save_strategy="epoch",
    bf16=True,
    tf32=True,
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_data,
    args=args,
    peft_config=peft_config,
    max_seq_length=2048,
    packing=True,
    dataset_kwargs={"add_special_tokens": False, "append_concat_token": False},
)

trainer.train()
trainer.save_model()
```



## Merging Adapters  

QLoRA saves only adapter weights. For simple deployment without PEFT, you can merge adapters into the base model on CPU and save a standalone checkpoint:

$$
W_{\text{merged}} = W_0 + \frac{\alpha}{r} B A
$$

```python
from peft import AutoPeftModelForCausalLM

peft_dir = "llama3-8b-qlora-demo"
peft_model = AutoPeftModelForCausalLM.from_pretrained(peft_dir, torch_dtype=torch.float16, low_cpu_mem_usage=True)
merged = peft_model.merge_and_unload()   # computes W₀ + (α/r)BA for each layer
merged.save_pretrained(peft_dir, safe_serialization=True, max_shard_size="2GB")
```


## Combining PEFT with `trl`

Once the base model is loaded, the practical recipe is straightforward:

- **LoRA** = full-precision or bf16 base model + `LoraConfig`
- **8-bit LoRA** = 8-bit base model + `LoraConfig`
- **QLoRA** = 4-bit base model + `LoraConfig`

The trainer class changes with the learning objective, but the PEFT pattern stays almost the same.

### Shared adapter config

```python
from peft import LoraConfig

peft_config = LoraConfig(
    r=16,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    target_modules="all-linear",
    task_type="CAUSAL_LM",
)
```

### SFT + LoRA / QLoRA

```python
from trl import SFTTrainer

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_data,
    args=args,
    peft_config=peft_config,
)
```

### DPO + LoRA / QLoRA

```python
from trl import DPOTrainer

trainer = DPOTrainer(
    model=model,
    ref_model=None,
    tokenizer=tokenizer,
    train_dataset=preference_data,
    args=dpo_args,
    peft_config=peft_config,
)
```

### GRPO + LoRA / QLoRA

```python
from trl import GRPOTrainer

trainer = GRPOTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_data,
    reward_funcs=reward_funcs,
    args=grpo_args,
    peft_config=peft_config,
)
```

### Practical rules of thumb

| Goal | Recommended setup |
|--|--|
| Small model fits comfortably | Full fine-tuning or LoRA |
| Medium model, limited VRAM | 8-bit LoRA |
| Large model, tight VRAM | QLoRA |
| Fastest iteration for teaching/demo | LoRA on a smaller model |
| Lowest memory footprint | 4-bit base + LoRA |

If you are unsure, a good default is: **start with LoRA on a small model, then move to QLoRA only when memory becomes the bottleneck**.



Tips for setting up LoRA:

| Setting | Guidance |
||-|
| `r` (rank) | Start with 16; increase to 64–128 for harder tasks |
| `lora_alpha` | Set equal to `r` (scale = 1) or 2× `r` (scale = 2) |
| `lora_dropout` | 0.05–0.1 for regularization |
| Quant bits | 4-bit for 8B+ models on 24GB VRAM; 8-bit for extra stability |
| `packing` | `True` for short examples; boosts throughput |


## References

- Hu et al., [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- Dettmers et al., [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- Hugging Face, [PEFT documentation](https://huggingface.co/docs/peft/index)
- Hugging Face, [TRL documentation](https://huggingface.co/docs/trl/index)
