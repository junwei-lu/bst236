# Hugging Face and Transformers Package

In this lecture, we'll explore [Hugging Face's Transformers library](https://huggingface.co/docs/transformers/index)—a powerful Python package for working with state-of-the-art NLP (Natural Language Processing) models. You can refer to the [hugging face course](https://huggingface.co/learn/llm-course) for more details.

## Hugging Face 🤗 Website

[Hugging Face 🤗](https://huggingface.co/) is often referred to as the "GitHub for AI models" - a central hub where researchers and developers can share, discover, and collaborate on machine learning models, datasets, and applications. The platform hosts thousands of pre-trained models that anyone can download and use.


- [**Model Repository**](https://huggingface.co/models): Access thousands of pre-trained models for various tasks including language modeling, computer vision, audio, and more. Each model has detailed documentation, including their capabilities, limitations, and intended uses.

- [**Datasets**](https://huggingface.co/datasets): A collection of public datasets for training and evaluating models.

- [**Spaces**](https://huggingface.co/spaces): Interactive web applications to demonstrate AI capabilities without requiring any setup.




The Hugging Face 🤗 ecosystem significantly lowers the barrier to entry for working with advanced AI models, making cutting-edge NLP accessible to developers of all skill levels.

## Transformers Package

Hugging Face 🤗 provides a powerful Python package called [Transformers](https://huggingface.co/docs/transformers/index) that simplifies working with many different NLP models. You can refer to the [official documentation](https://huggingface.co/docs/transformers/index) for more details of API usage.

To use the Transformers package, you need to install the following packages:

```bash
pip install torch transformers datasets accelerate
```

In order to use the pre-trained models, you need to check the [model hub](https://huggingface.co/models) to find the model you want to use. The model string is in the format of `{model_owner}/{model_name}`.  We will use the `AutoTokenizer` and `AutoModel` classes to load the tokenizer and model.

Below is an example of how to load the tokenizer and model for [DistilBERT](https://huggingface.co/distilbert-base-uncased), a lightweight and efficient transformer model:

```python
from transformers import AutoTokenizer, AutoModel

model_name = "distilbert/distilbert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name) # Load the tokenizer
model = AutoModel.from_pretrained(model_name) # Load the model
print(tokenizer)
print(model)
```



### Tokenization

Once you have loaded the tokenizer, you can use the `tokenize` method to convert raw text into tokens.

```python
text = "Hello, how are you?"
tokens = tokenizer.tokenize(text)

print("Tokens:", tokens)
# Example output: ['Hello', ',', 'how', 'are', 'you', '?']
```

Models don't directly understand text—they use numerical representations called token IDs.

```python
# Convert tokens to token IDs
token_ids = tokenizer.encode(text)
print("Token IDs:", token_ids)

# Convert token IDs back to tokens (includes special tokens such as [CLS], [SEP])
tokens_with_special = tokenizer.convert_ids_to_tokens(token_ids)
print("Tokens with special characters:", tokens_with_special)
# Example output: ['[CLS]', 'Hello', ',', 'how', 'are', 'you', '?', '[SEP]']
```


Models require input tensors. Here, we tokenize and convert our input text to PyTorch tensors:

```python
inputs = tokenizer(text, return_tensors='pt')  # 'pt' stands for PyTorch tensors
print(inputs)
# {'input_ids': 
#    tensor([[ 101, 8667,  117, 1293, 1132, 1128,  136,  102]]), 
#  'attention_mask': 
#    tensor([[1, 1, 1, 1, 1, 1, 1, 1]])
# }
```

The `inputs` dictionary contains keys:

- `input_ids`: The token IDs of the input text with dimensions `(batch_size, sequence_length)`.
- `attention_mask`: A binary mask indicating which tokens are real (1) and which are padding (0) with dimensions `(batch_size, sequence_length)`.

You can then pass the `inputs` dictionary to the model using two different methods:

```python
# Option 1: Pass the entire inputs dictionary
outputs = model(**inputs)
# Option 2: Pass the input IDs and attention mask separately
outputs = model(inputs['input_ids'], attention_mask=inputs['attention_mask'])
```



**Batch of sentences**

You can also pass batch of sentences to the model.
Tokenizing multiple sentences at once can require padding to ensure consistent input lengths:

```python
texts = [
    "Hello, how are you?",
    "I'm fine, thank you! And you?",
    "I'm not fine."
]

# Pad sentences to match the length of the longest sentence
model_inputs = tokenizer(texts, padding=True, return_tensors='pt')

print(f"Pad token: {tokenizer.pad_token} | Pad token id: {tokenizer.pad_token_id}")
# Print input ids
print(model_inputs['input_ids'])
# Print attention mask
print(model_inputs['attention_mask'])
# Pad token: [PAD] | Pad token id: 0
# tensor([[101,8667,117,1293,1132,1128,136,102,0,0,0,0,0],
#         [101,146,112,182,2503,117,6243,1128,106,1262,1128,136,102],
#         [101,146,112,182,1136,2503,119,102,0,0,0,0,0]])
# tensor([[1,1,1,1,1,1,1,1,0,0,0,0,0],
#         [1,1,1,1,1,1,1,1,1,1,1,1,1],
#         [1,1,1,1,1,1,1,1,0,0,0,0,0]])
```

---

### Model Parameters

Obtain embeddings for multiple sentences and explore model configuration:

```python
model_outputs = model(**model_inputs)
token_embeddings = model_outputs.last_hidden_state

print("Token embeddings shape (multiple sentences):", token_embeddings.shape)

# Inspect model configuration (details like number of layers, hidden sizes)
print("Model configuration:", model.config)
```

To get the middle layer of the model, you can set `output_hidden_states=True` when initializing the model. Then the model output will have `hidden_states` and `attentions` attributes.

```python
model = AutoModel.from_pretrained("distilbert-base-cased", output_attentions=True, output_hidden_states=True)
model.eval()
input_str = "Hello, how are you?"
model_inputs = tokenizer(input_str, return_tensors="pt")
with torch.no_grad():
    model_output = model(**model_inputs)

print(f"Hidden state size (per layer):  {model_output.hidden_states[0].shape}")
print(f"Attention head size (per layer): {model_output.attentions[0].shape}")     # (layer, head_number, query_word_idx, key_word_idxs)
# Attention is softmax(K^T * Q / sqrt(d_k))
```


You can visualize the attention scores from different layers of different heads using the following code:

```python
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
```

![Attention Scores](./tf.assets/attentions_scores.png)


### Loading Datasets
Similar to the [torchvision](../chapter_neural_networks/computer_vision.md#dataset-cifar-10), Hugging Face 🤗 provides a `datasets` package that allows you to load and preprocess datasets. You can refer to the [datasets documentation](https://huggingface.co/docs/datasets/index) for more details.

You can find many datasets in the [datasets hub](https://huggingface.co/datasets). Here we will use the [imdb](https://huggingface.co/datasets/imdb) dataset for sentiment analysis with the reviews as the text and the binary labels {0, 1} as the negative and positive targets.

```python
# load dataset
from datasets import load_dataset, DatasetDict
# DataLoader(zip(list1, list2))
dataset_name = "stanfordnlp/imdb"
imdb_dataset = load_dataset(dataset_name)
```

We take a subset of the dataset for demonstration.

```python
# Just take the first 50 tokens for speed/running on cpu
def truncate(example):
    return {
        'text': " ".join(example['text'].split()[:50]),
        'label': example['label']
    }
# Take 128 random examples for train and 32 validation
small_imdb_dataset = DatasetDict(
    train=imdb_dataset['train'].shuffle(seed=1111).select(range(128)).map(truncate),
    val=imdb_dataset['train'].shuffle(seed=1111).select(range(128, 160)).map(truncate),
)    
```
We then use the tokenizer to tokenize the text and convert the text to tokens.

```python
small_tokenized_dataset = small_imdb_dataset.map(
    lambda example: tokenizer(example['text'], padding=True, truncation=True), # It truncates any input text that exceeds the model’s maximum token length
    batched=True,
    batch_size=16
)

small_tokenized_dataset = small_tokenized_dataset.remove_columns(["text"])
small_tokenized_dataset = small_tokenized_dataset.rename_column("label", "labels")
small_tokenized_dataset.set_format("torch")
```

Now we can use the `DataLoader` to load the dataset.

```python
# %% Now we can create a DataLoader as usual
from torch.utils.data import DataLoader
train_dataloader = DataLoader(small_tokenized_dataset['train'], batch_size=16)
eval_dataloader = DataLoader(small_tokenized_dataset['val'], batch_size=16)
```

The remaining part is regular training loop as the previous chapter. For models in hugging face, we can directly get the loss by `model(**input).loss`.

```python
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
```

## Hugging Face Trainer

Hugging Face 🤗 provides a `Trainer` class that simplifies the training loop. You can refer to the [Trainer documentation](https://huggingface.co/docs/transformers/main/en/main_classes/trainer) for more details. With the `Trainer`, we can write the above training loop as follows:

```python
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

trainer.train()
```

### Fine-tuning a model

Similar to the [previous chapter](../chapter_neural_networks/fine_tuning.md), we can fine-tune a model for a specific task. Here we will fine-tune the model from hugging face by freezing the base encoder.

```python
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-cased', num_labels=2)
print(model)
for param in model.distilbert.parameters():
    param.requires_grad = False  # freeze base encoder
# Continue training the model
```  

### Model Inference

Although pretrained transformers generally share similar architectures, they require task-specific "heads" - additional layers of weights that need training for particular tasks like sequence classification or question answering. Hugging Face simplifies this by providing specialized model classes that automatically configure the appropriate architecture. For example, [DistilBERT](https://huggingface.co/docs/transformers/model_doc/distilbert) has the following models for different tasks:

- `DistilBertModel`: The base model for all the tasks.
- `DistilBertForMaskedLM`: For masked language modeling.
- `DistilBertForSequenceClassification`: For sequence classification.
- `DistilBertForMultipleChoice`: For multiple choice tasks.
- `DistilBertForTokenClassification`: For token classification tasks.
- `DistilBertForQuestionAnswering`: For question answering tasks.

You can also load different tasks models by `AutoModelFor*`:

- `AutoModelForSequenceClassification`: For sequence classification.
- `AutoModelForMaskedLM`: For masked language modeling.
- `AutoModelForMultipleChoice`: For multiple choice tasks.
- `AutoModelForTokenClassification`: For token classification tasks.
- `AutoModelForQuestionAnswering`: For question answering tasks.

Below is an example of how to load the model for sequence classification using two different methods.

```python
from transformers import AutoModelForSequenceClassification, DistilBertForSequenceClassification, DistilBertModel
print('Loading base model')
base_model = DistilBertModel.from_pretrained('distilbert-base-cased')
# Method 1
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-cased', num_labels=2)
# Method 2
model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-cased', num_labels=2)
```




As we mentioned in the [previous lecture](transformer.md#choosing-transformer-architecture), BERT is a encoder model which is proper for classification tasks.

### Language Generation

Hugging Face also has the encoder models like [GPT-2](https://huggingface.co/docs/transformers/model_doc/gpt2).

We can use the `AutoModelForCausalLM` to load the GPT-2 model and generate text.

```python

from transformers import AutoModelForCausalLM
gpt2 = AutoModelForCausalLM.from_pretrained('distilgpt2')

prompt = "Once upon a time"
tokenized_prompt = gpt2_tokenizer(prompt, return_tensors="pt")

for i in range(10):
    output = gpt2.generate(**tokenized_prompt,
                  max_length=50,
                  do_sample=True,
                  top_p=0.9)

    print(f"{i + 1}) {gpt2_tokenizer.batch_decode(output)[0]}")
```





