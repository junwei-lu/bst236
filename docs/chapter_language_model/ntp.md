# Next-Token Prediction

<a href="https://colab.research.google.com/github/junwei-lu/ai4med/blob/main/codes/nlp/llm_model2pretrain2ft.ipynb#scrollTo=nlp_sec4_ntp" target="_blank" style="display: inline-flex; align-items: center; gap: 6px; padding: 6px 12px; background: linear-gradient(135deg, #1565c0 0%, #42a5f5 100%); color: white; border-radius: 6px; text-decoration: none; font-size: 0.85em; font-weight: 600;">▶ Try in Colab</a>

Before fine-tuning, it is essential to understand what a language model actually learns during pre-training. The core objective is **next-token prediction (NTP)**: given a sequence of tokens, predict the next one. Everything from GPT-2 to Llama 3 is trained with this single principle.



## The Probability Model

A language model assigns a probability to every possible sequence of tokens $x_1, x_2, \ldots, x_T$. Using the **chain rule of probability**, any joint distribution can be factored autoregressively:

$$
P(x_1, x_2, \ldots, x_T) = \prod_{t=1}^{T} P(x_t \mid x_1, x_2, \ldots, x_{t-1})
$$

A neural language model with parameters $\theta$ approximates each conditional:

$$
P_\theta(x_t \mid x_1, \ldots, x_{t-1})
$$

This is a **categorical distribution** over the vocabulary $\mathcal{V}$(e.g., 32,000 tokens for Llama). The model outputs a vector of logits $\mathbf{z}_t \in \mathbb{R}^{|\mathcal{V}|}$, which are converted to probabilities via softmax:

$$
P_\theta(x_t = v \mid x_{<t}) = \frac{\exp(z_{t,v})}{\sum_{v' \in \mathcal{V}} \exp(z_{t,v'})}
$$

**Autoregressive generation:** At inference time, the model generates token by token—each new token is appended to the context and fed back in to predict the next:

![Autoregressive token generation](./ft.assets/ntp_autoregress.gif)



## The Training Objective: Cross-Entropy Loss

Given a training corpus of documents, each document is treated as a sequence of tokens. The model is trained to **maximize the log-likelihood** of observed tokens, which is equivalent to **minimizing the cross-entropy loss**.

For a single document of length $T$:

$$
\mathcal{L}(\theta) = -\frac{1}{T} \sum_{t=1}^{T} \log P_\theta(x_t \mid x_1, \ldots, x_{t-1})
$$

Each term $-\log P_\theta(x_t \mid x_{<t})$ measures how surprised the model is when it sees the actual next token $x_t$. A perfect model would assign probability 1 to the correct token, giving a loss of 0.

**Over a dataset** $\mathcal{D} = \{d^{(1)}, d^{(2)}, \ldots, d^{(N)}\}$ of $N$ documents:

$$
\mathcal{L}(\theta) = -\frac{1}{|\mathcal{D}|} \sum_{d \in \mathcal{D}} \frac{1}{|d|} \sum_{t=1}^{|d|} \log P_\theta(x_t^{(d)} \mid x_1^{(d)}, \ldots, x_{t-1}^{(d)})
$$

### Connection to Perplexity

**Perplexity (PPL)** is the standard evaluation metric for language models and is directly tied to the NTP loss:

$$
\text{PPL} = \exp\!\left(\mathcal{L}(\theta)\right) = \exp\!\left(-\frac{1}{T}\sum_{t=1}^{T} \log P_\theta(x_t \mid x_{<t})\right)
$$

Intuitively, a perplexity of $k $ means the model is "as confused as if choosing uniformly among$k$ options" at each step. Lower is better.

![NTP Pipeline](ft.assets/nft_plot.png)


## Contrast: Masked Encoder Training

Autoregressive next-token prediction is the standard objective for **decoder-only** models such as GPT and Llama. By contrast, **encoder-only** models such as BERT are usually trained with **masked language modeling (MLM)**: randomly hide a subset of tokens, then predict the missing tokens from both left and right context.

If $\mathcal{M}$ is the set of masked positions, the MLM loss is:

$$
\mathcal{L}_{\text{MLM}}(\theta) = -\frac{1}{|\mathcal{M}|} \sum_{t \in \mathcal{M}} \log P_\theta(x_t \mid x_{\setminus \mathcal{M}})
$$

The key difference is architectural:

- **Causal NTP** uses a triangular mask and only looks left, so it supports text generation naturally
- **Masked encoder training** uses bidirectional context, so it learns strong contextual representations for classification, retrieval, and token labeling
- Encoder models are excellent feature extractors, but they are not usually used as standalone autoregressive generators

So when you hear "masked encoder training," think **representation learning with missing-token reconstruction**, not step-by-step text generation.


## Why Does This Work?

Training on next-token prediction on large corpora forces the model to:

1. **Learn syntax and grammar** — token sequences must be grammatically plausible
2. **Learn factual knowledge** — predicting "The capital of France is ___" requires knowing "Paris"
3. **Learn reasoning patterns** — math or logic examples appear in text and must be predicted correctly
4. **Learn long-range dependencies** — the Transformer's attention lets each prediction attend to all prior tokens

This is why a model pre-trained purely on NTP can then be fine-tuned for specific tasks with relatively few examples.



## Training Next-Token Prediction Loss from Scratch

Let us implement the NTP loss manually to build intuition before using high-level trainers. If you are beginner, we suggest you skip this part and directly use the `transformer` package in the [next part](#training-ntp-with-hugging-face-transformers).

### Minimal example with pure PyTorch

```python
import torch
import torch.nn.functional as F

#  Toy example 
# Suppose we have a tiny vocabulary of 5 tokens and a sequence of length 4
# tokens: [2, 0, 3, 1]  → input = [2, 0, 3], target = [0, 3, 1]

vocab_size = 5
seq_len = 3  # we predict 3 positions

# Simulated logits from the model (shape: [seq_len, vocab_size])
torch.manual_seed(42)
logits = torch.randn(seq_len, vocab_size)

# The correct next tokens for each position
targets = torch.tensor([0, 3, 1])  # shape: [seq_len]

# Cross-entropy loss: equivalent to -log P(correct token)
# F.cross_entropy applies log-softmax internally
loss_per_token = F.cross_entropy(logits, targets, reduction="none")
print("Per-token losses:", loss_per_token)

loss = loss_per_token.mean()
print(f"Average NTP loss: {loss.item():.4f}")
print(f"Perplexity: {torch.exp(loss).item():.2f}")
```

Expected output (deterministic with `torch.manual_seed(42)`):
```
Per-token losses: tensor([1.3644, 2.3091, 1.8469])
Average NTP loss: 1.8401
Perplexity: 6.30
```

### Shift-by-one: input vs. target in practice

The key implementation detail: **the target at position $t$ is the input at position$t+1$**. This is done by shifting the token sequence by one.

```python
import torch
import torch.nn.functional as F

def ntp_loss(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """
    Compute causal language modeling (NTP) loss.

    Args:
        logits:    Model output, shape [batch, seq_len, vocab_size]
        input_ids: Token IDs,     shape [batch, seq_len]

    Returns:
        Scalar loss (mean cross-entropy over all non-padding positions)
    """
    # Shift: predict position t+1 using logits at position t
    # logits at positions 0..T-2 should predict tokens at positions 1..T-1
    shift_logits = logits[:, :-1, :].contiguous()   # [B, T-1, V]
    shift_labels = input_ids[:, 1:].contiguous()     # [B, T-1]

    # Flatten batch and time dimensions for cross_entropy
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),  # [B*(T-1), V]
        shift_labels.view(-1),                          # [B*(T-1)]
    )
    return loss

#  Demo with a batch of 2 sequences of length 6 
torch.manual_seed(0)
B, T, V = 2, 6, 32000  # batch, seq_len, vocab_size

dummy_logits = torch.randn(B, T, V)
dummy_input_ids = torch.randint(0, V, (B, T))

loss = ntp_loss(dummy_logits, dummy_input_ids)
print(f"NTP loss: {loss.item():.4f}")         # ~log(32000) ≈ 10.37 for random init
print(f"Perplexity: {torch.exp(loss).item():.1f}")
```



## What the Gradient Does

During backpropagation, the gradient of the loss with respect to the logit $z_{t,v}$ is:

$$
\frac{\partial \mathcal{L}}{\partial z_{t,v}} = P_\theta(x_t = v \mid x_{<t}) - \mathbf{1}[v = x_t]
$$

This means:

- For the **correct token** $v = x_t$: the gradient is $P_\theta - 1$, which is **negative** → the logit is pushed **up**
- For all **other tokens**: the gradient is $P_\theta > 0$, which is **positive** → those logits are pushed **down**

The model learns by repeatedly increasing the probability of observed tokens and decreasing the probability of unobserved tokens.


## Training NTP with Hugging Face Transformers

The manual PyTorch code above builds intuition, but in practice we use the Hugging Face `Trainer` API. This section walks through a complete pipeline: prepare the data, configure the model, train, and generate text.

### Step 1: Install dependencies

```bash
pip install transformers datasets torch
```

### Step 2: Load a pre-trained model and tokenizer

We start from a pre-trained GPT-2 checkpoint. Even when the goal is continued pre-training on domain-specific text, initializing from an existing checkpoint is much cheaper than training from scratch.

```python
import torch
from transformers import AutoTokenizer, GPT2LMHeadModel

model_name = "gpt2"  # 124M parameters
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

tokenizer.pad_token = tokenizer.eos_token

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Vocab size: {tokenizer.vocab_size}")
```

??? note "Training from scratch"
    To train a randomly initialized model instead (as done in the workshop notebook), replace `from_pretrained` with a fresh config:

    ```python
    from transformers import GPT2Config, GPT2LMHeadModel

    config = GPT2Config(
        vocab_size=len(tokenizer),
        n_positions=256,
        n_embd=256,
        n_layer=4,
        n_head=4,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    model = GPT2LMHeadModel(config)
    ```

### Step 3: Prepare the dataset

NTP training requires **long, contiguous chunks** of tokens. The standard recipe is:

1. Tokenize every document
2. Concatenate all token IDs into one long stream
3. Slice the stream into fixed-length blocks

```python
from datasets import load_dataset

dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

BLOCK_SIZE = 256

def tokenize_and_chunk(examples):
    tokenized = tokenizer(examples["text"], truncation=False)
    all_ids = []
    for ids in tokenized["input_ids"]:
        all_ids.extend(ids)
    chunks = [
        all_ids[i : i + BLOCK_SIZE]
        for i in range(0, len(all_ids) - BLOCK_SIZE, BLOCK_SIZE)
    ]
    return {"input_ids": chunks}

lm_dataset = dataset.map(
    tokenize_and_chunk,
    batched=True,
    remove_columns=dataset.column_names,
    batch_size=1000,
)
lm_dataset.set_format("torch")

print(f"Training chunks: {len(lm_dataset)} (each {BLOCK_SIZE} tokens)")
```

**Why concatenate-then-chunk?** Documents vary in length. If we padded each document to the block size individually, most tokens in a batch would be padding—wasting computation. Concatenating documents into a continuous stream and slicing into equal-length blocks keeps every token meaningful.

### Step 4: Data collator

`DataCollatorForLanguageModeling` with `mlm=False` handles the shift-by-one logic: it copies `input_ids` into `labels` so the model's internal loss function can compare position $t$ logits against position $t+1$ tokens.

```python
from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # causal LM, not masked LM
)
```

### Step 5: Configure training

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="gpt2-ntp",
    max_steps=500,
    per_device_train_batch_size=8,
    learning_rate=5e-4,
    warmup_steps=100,
    logging_steps=10,
    fp16=torch.cuda.is_available(),
    save_strategy="no",
    report_to="none",
    seed=42,
)
```

| Argument | Purpose |
|---|---|
| `max_steps` | Total gradient updates. Use `num_train_epochs` instead for full-epoch training. |
| `learning_rate` | Peak LR after warmup. 5e-4 is typical for small-scale continued pre-training. |
| `warmup_steps` | Linear warmup avoids large early updates that destabilize training. |
| `fp16` | Mixed-precision training — roughly 2x speed on modern GPUs with no quality loss. |

### Step 6: Train

```python
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_dataset,
    data_collator=data_collator,
)

trainer.train()
```

The training loss should drop quickly in the first 100 steps (the model learns basic token co-occurrence patterns) and then decrease more slowly as it captures longer-range dependencies.

### Step 7: Generate text

After training, test the model with autoregressive generation:

```python
def generate(prompt, max_new_tokens=100, temperature=0.7):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(
        output[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )

prompts = [
    "The patient presented with",
    "Recent studies have shown that",
    "In this paper, we propose",
]

for p in prompts:
    print(f"Prompt: '{p}'")
    print(f"Output: {generate(p)}\n")
```

### Step 8: Save and reload

```python
model.save_pretrained("gpt2-ntp")
tokenizer.save_pretrained("gpt2-ntp")

reloaded = GPT2LMHeadModel.from_pretrained("gpt2-ntp").to(device)
```

The saved model can later serve as the starting point for [supervised fine-tuning](sft.md) or [parameter-efficient fine-tuning](peft.md).


## References

- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- Radford et al., [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- Devlin et al., [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
