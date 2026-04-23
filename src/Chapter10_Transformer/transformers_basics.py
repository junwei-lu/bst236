#%%
import torch
from transformers import AutoTokenizer, AutoModel
#%%
name = "distilbert/distilbert-base-cased"
# name = "user/name" when loading from
# Load pre-trained tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(name)
model = AutoModel.from_pretrained(name)
print(tokenizer)
print(model)
#%% Sample input
text = "Hello, how are you?"

# Get tokens as strings
tokens = tokenizer.tokenize(text)
print(tokens)

#%% # If you want to see token IDs and then convert back to tokens
token_ids = tokenizer.encode(text)
print(token_ids)
tokens_with_special = tokenizer.convert_ids_to_tokens(token_ids)
print(tokens_with_special)


#%%
# Tokenization and obtaining token embeddings
inputs = tokenizer(text, return_tensors='pt') # return_tensors='pt' returns a PyTorch tensor
def print_encoding(model_inputs, indent=4):
    indent_str = " " * indent
    print("{")
    for k, v in model_inputs.items():
        print(indent_str + k + ":")
        print(indent_str + indent_str + str(v))
    print("}")

print_encoding(inputs)

#%% Get Input Embeddings
input_ids = inputs["input_ids"]
word_embeddings = model.get_input_embeddings()(input_ids)
print(word_embeddings.shape)
# Get positional embeddings
position_ids = torch.arange(0, input_ids.size(1)).unsqueeze(0)
position_embeddings = model.distilbert.embeddings.position_embeddings
position_embeddings_output = position_embeddings(position_ids)

print(position_embeddings_output.shape)
#%%
outputs = model(**inputs)

# Extract token embeddings
token_embeddings = outputs.last_hidden_state  # Shape: [batch_size, seq_len, hidden_size]
print(token_embeddings.shape)


#%% You can pass multiple strings into the tokenizer and pad them as you need

texts = ["Hello, how are you?", 
         "I'm fine, thank you! And you?",
         "I'm not fine."
         ]
model_inputs = tokenizer(texts, padding=True, return_tensors='pt')

#%%
print(f"Pad token: {tokenizer.pad_token} | Pad token id: {tokenizer.pad_token_id}")
# Print input ids
print(model_inputs['input_ids'])
# Print attention mask
print(model_inputs['attention_mask'])
# Print tokens
print(tokenizer.batch_decode(model_inputs.input_ids))

#%%
model = AutoModel.from_pretrained(name)
# Option 1: Pass the model inputs directly
model_outputs = model(**model_inputs)
# Option 2: Pass the input ids and attention mask separately
model_outputs = model(input_ids=model_inputs.input_ids, attention_mask=model_inputs.attention_mask)
print(model_outputs)
token_embeddings = model_outputs.last_hidden_state
print(token_embeddings.shape)

#%%
# Accessing the model configuration
configuration = model.config
print(configuration)

#%% Take a look at the model's parameters
for name, param in model.named_parameters():
    print(name, param.shape)

#%% Get hidden states
model = AutoModel.from_pretrained("distilbert-base-cased", output_attentions=True, output_hidden_states=True)
model.eval()
input_str = "Hello, how are you?"
model_inputs = tokenizer(input_str, return_tensors="pt")
with torch.no_grad():
    model_output = model(**model_inputs)

print(f"Hidden state size (per layer):  {model_output.hidden_states[0].shape}")
print(f"Attention head size (per layer): {model_output.attentions[0].shape}")     # (layer, head_number, query_word_idx, key_word_idxs)
# Attention is softmax(K^T * Q / sqrt(d_k))


# %% Visualize attention heads
tokens = tokenizer.convert_ids_to_tokens(model_inputs.input_ids[0])
print(tokens)

import matplotlib.pyplot as plt
n_layers = len(model_output.attentions)
n_heads = len(model_output.attentions[0][0])
fig, axes = plt.subplots(6, 12)
fig.set_size_inches(18.5*2, 10.5*2)
for layer in range(n_layers):
    for i in range(n_heads):
        axes[layer, i].imshow(model_output.attentions[layer][0, i])
        axes[layer][i].set_xticks(list(range(8)))
        axes[layer][i].set_xticklabels(labels=tokens, rotation="vertical")
        axes[layer][i].set_yticks(list(range(8)))
        axes[layer][i].set_yticklabels(labels=tokens)

        if layer == 5:
            axes[layer, i].set(xlabel=f"head={i}")
        if i == 0:
            axes[layer, i].set(ylabel=f"layer={layer}")

plt.subplots_adjust(wspace=0.3)
plt.show()

# %% Training a model
# load dataset
from datasets import load_dataset, DatasetDict

# DataLoader(zip(list1, list2))
dataset_name = "stanfordnlp/imdb"
imdb_dataset = load_dataset(dataset_name)
# Just take the first 50 tokens for speed/running on cpu
def truncate(example):
    return {
        'text': " ".join(example['text'].split()[:50]),
        'label': example['label']
    }
imdb_dataset

#%%
# Take 128 random examples for train and 32 validation
small_imdb_dataset = DatasetDict(
    train=imdb_dataset['train'].shuffle(seed=1111).select(range(128)).map(truncate),
    val=imdb_dataset['train'].shuffle(seed=1111).select(range(128, 160)).map(truncate),
)
small_imdb_dataset

#%%
small_imdb_dataset['train'][:10]

# %%# Prepare the dataset - this tokenizes the dataset in batches of 16 examples.
small_tokenized_dataset = small_imdb_dataset.map(
    lambda example: tokenizer(example['text'], padding=True, truncation=True), # https://huggingface.co/docs/transformers/pad_truncation
    batched=True,
    batch_size=16
)
# It truncates any input text that exceeds the model’s maximum token length (e.g., 512 for BERT).

small_tokenized_dataset = small_tokenized_dataset.remove_columns(["text"])
small_tokenized_dataset = small_tokenized_dataset.rename_column("label", "labels")
small_tokenized_dataset.set_format("torch")

# %%
small_tokenized_dataset['train'][0:2]

#%% Training Loop: Model for sequence classification
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from transformers import DistilBertForSequenceClassification, DistilBertConfig
from tqdm.notebook import tqdm


# Initializing a DistilBERT configuration
configuration = DistilBertConfig()
configuration.num_labels=2
# Initializing a model (with random weights) from the configuration
model = DistilBertForSequenceClassification(configuration)

#%% Check the model's output
input_str = small_imdb_dataset['train'][0]['text']
label = small_imdb_dataset['train'][0]['label']
input_str

model_inputs = tokenizer(input_str, return_tensors="pt")
model_outputs = model(**model_inputs)
print(f"Distribution over labels: {torch.softmax(model_outputs.logits, dim=1)}")

#%% Compute the loss
label = torch.tensor([label])
loss = torch.nn.functional.cross_entropy(model_outputs.logits, label)
loss.backward()

# %% Now we can create a DataLoader
from torch.utils.data import DataLoader

train_dataloader = DataLoader(small_tokenized_dataset['train'], batch_size=16)
eval_dataloader = DataLoader(small_tokenized_dataset['val'], batch_size=16)

#%%
from tqdm import tqdm
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

num_epochs = 1
num_training_steps = len(train_dataloader)
optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

best_val_loss = float("inf")
progress_bar = tqdm(range(num_training_steps))
for epoch in range(num_epochs):
    # training
    model.train()
    for batch_i, batch in enumerate(train_dataloader):

        # batch = ([text1, text2], [0, 1])

        output = model(**batch)

        optimizer.zero_grad()
        output.loss.backward()
        optimizer.step()
        lr_scheduler.step()
        progress_bar.update(1)

    # validation
    model.eval()
    for batch_i, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            output = model(**batch)
        loss += output.loss

    avg_val_loss = loss / len(eval_dataloader)
    print(f"Validation loss: {avg_val_loss}")
    if avg_val_loss < best_val_loss:
        print("Saving checkpoint!")
        best_val_loss = avg_val_loss
        # torch.save({
        #     'epoch': epoch,
        #     'model_state_dict': model.state_dict(),
        #     'optimizer_state_dict': optimizer.state_dict(),
        #     'val_loss': best_val_loss,
        #     },
        #     f"checkpoints/epoch_{epoch}.pt"
        # )
# %% 
# Hugging Face Trainer
imdb_dataset = load_dataset("stanfordnlp/imdb")

small_imdb_dataset = DatasetDict(
    train=imdb_dataset['train'].shuffle(seed=1111).select(range(128)).map(truncate),
    val=imdb_dataset['train'].shuffle(seed=1111).select(range(128, 160)).map(truncate),
)

small_tokenized_dataset = small_imdb_dataset.map(
    lambda example: tokenizer(example['text'], truncation=True),
    batched=True,
    batch_size=16
)

#%% 
from transformers import TrainingArguments, Trainer

model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-cased', num_labels=2)

arguments = TrainingArguments(
    output_dir="sample_hf_trainer",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    load_best_model_at_end=True,
    seed=224
)


def compute_metrics(eval_pred):
    """Called at the end of validation. Gives accuracy"""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    # calculates the accuracy
    return {"accuracy": np.mean(predictions == labels)}


trainer = Trainer(
    model=model,
    args=arguments,
    train_dataset=small_tokenized_dataset['train'],
    eval_dataset=small_tokenized_dataset['val'], # change to test when you do your final evaluation!
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)
# %% Fine-tuning
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-cased', num_labels=2)
print(model)
for param in model.distilbert.parameters():
    param.requires_grad = False  # freeze base encoder


#%% Generation

from transformers import AutoModelForCausalLM

gpt2_tokenizer = AutoTokenizer.from_pretrained('gpt2')

gpt2 = AutoModelForCausalLM.from_pretrained('distilgpt2')
# gpt2.config.pad_token_id = gpt2.config.eos_token_id  # Prevents warning during decoding
# %%
prompt = "Once upon a time"

tokenized_prompt = gpt2_tokenizer(prompt, return_tensors="pt")

for i in range(10):
    output = gpt2.generate(**tokenized_prompt,
                  max_length=50,
                  do_sample=True,
                  top_p=0.9)

    print(f"{i + 1}) {gpt2_tokenizer.batch_decode(output)[0]}")
# %%
# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V2.5")
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-V2.5")
prompt = "Once upon a time"

tokenized_prompt = tokenizer(prompt, return_tensors="pt")

for i in range(10):
    output = model.generate(**tokenized_prompt,
                  max_length=50,
                  do_sample=True,
                  top_p=0.9)

    print(f"{i + 1}) {gpt2_tokenizer.batch_decode(output)[0]}")



# %% Pipeline
from transformers import pipeline

classifier = pipeline("sentiment-analysis")

classifier("I've been waiting for a Hugging Face course my whole life.")

#%%
generator = pipeline("text-generation")

result = generator("Hello, I've been")

# %% Use the DistilBERT model for sentiment analysis

classifier = pipeline("sentiment-analysis", model="")
