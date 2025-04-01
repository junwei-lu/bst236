# Deep Learning with GPU

Training deep learning models can be computationally intensive. Leveraging Graphics Processing Units (GPUs) can significantly accelerate this process. This guide introduces the basics of GPU usage in PyTorch, including the differences between CPUs and GPUs, how to move data and models between devices, and strategies for multi-GPU training.

## CPU versus GPU

Central Processing Units (CPUs) are designed for general-purpose computing tasks. They excel at handling a few complex threads simultaneously.​

Graphics Processing Units (GPUs) are specialized hardware designed to handle parallel tasks efficiently. They are particularly well-suited for operations like matrix multiplications, which are common in deep learning.​
By utilizing GPUs for deep learning, training times can be reduced significantly compared to using CPUs alone.​


## PyTorch with GPU 

Before leveraging GPUs in PyTorch, ensure that your environment is set up correctly.
We have discussed how to set up GPU cores in the class cluster in the [GPU Matrix](./gpu_matrix.md#set-up-gpu-core-in-the-class-cluster) chapter. In general, you can use the command `nvidia-smi` to check the GPU status.

```bash
[jul924@ip-10-37-33-243 ~]$ salloc --partition=gpu --cpus-per-task=1 --mem=30G --time=01:00:00 srun --pty bash
salloc: Granted job allocation 38087
[jul924@gpu-dy-gpu-cr-7 ~]$ nvidia-smi
Mon Mar  3 19:16:23 2025       
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.216.01             Driver Version: 535.216.01   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA L4                      On  | 00000000:31:00.0 Off |                    0 |
| N/A   28C    P8              16W /  72W |      0MiB / 23034MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|  No running processes found                                                           |
+---------------------------------------------------------------------------------------+
```

In PyTorch, we can check the device type by using the `torch.cuda.is_available()` function. 

```python
import torch

# Check if CUDA is available
if torch.cuda.is_available():
    print("CUDA is available. You can use GPU acceleration.")
else:
    print("CUDA is not available. Using CPU.")
```

**Moving data and models between devices**

To move data and models between devices, we can use the `to()` method. If you train your model on the GPU, you need to move both the training data and the model parameters to the GPU.

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
x = torch.randn(3, 3)
x = x.to(device)
model = MyModel()
model.to(device)
```

**Best practices for `.to(device)` in PyTorch Training**

- Remember that move data among CPU and GPU is time-consuming. 

- Keep model and data on the same GPU device if the GPU memory is enough.

- Don’t move the entire dataset to the GPU at once. If the dataset is large, it won’t fit into GPU memory. Instead, move each batch of data to the GPU inside the training loop.

- Use `pin_memory=True` in the `DataLoader`. In PyTorch's `DataLoader`, setting `pin_memory=True` ensures that the data loaded by the `DataLoader` resides in pinned (page-locked) memory. Pinned memory cannot be swapped to disk by the operating system, allowing faster data transfer to the GPU. Normally, data in pageable memory requires an extra step during transfer to the GPU: it must first be copied to an intermediate pinned memory buffer before being sent to the GPU. By directly loading data into pinned memory, this intermediate step is eliminated, resulting in faster and more efficient data transfers. You may need to set `pin_memory=False` when your system has limited RAM, as pinned memory consumes more of it. Excessive use can lead to system instability.

```python
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, pin_memory=True)
```

- Use `non_blocking=True` in the `to()` method if using GPU. When moving data from the CPU to the GPU using the `.to()` method, setting `non_blocking=True` allows the data transfer to occur asynchronously. This means the CPU can continue executing subsequent operations without waiting for the data transfer to complete. Asynchronous data transfers enable the overlap of data transfer and computation, leading to better utilization of both CPU and GPU resources. This overlap reduces idle times and can significantly speed up the training process.​ Notice that you need to set `pin_memory=True` in the `DataLoader` to make `non_blocking=True` effective. If the source tensor is not in pinned memory, the asynchronous transfer may be even slower.

```python
data = data.to(device, non_blocking=True)
```


We summarize the best practices for `.to(device)` above in the following checklist.

| Checklist | When to Use |
|-----------|-------------|
| `.to(device)` on model | Once after model creation |
| `.to(device)` on data | Inside the training loop, batch-by-batch |
| `pin_memory=True` in `DataLoader` | When using GPU, to speed up transfer |
| `non_blocking=True` in `.to()` | For faster async transfer with pinned memory |



Here we provide an example code for training a model on GPU.

```python
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim

# 1. Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 2. Define transforms and dataset (on disk)
transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
# 3. Dataloader - use pin_memory to speed up CPU -> GPU transfer
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, pin_memory=True)
# 4. Define model and move to GPU
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28*28, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)
model.to(device)
# 5. Optimizer and loss
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()
# 6. Training loop - move each batch to GPU
for epoch in range(5):
    model.train()
    for batch_idx, (data, targets) in enumerate(train_loader):
        # Move data to the same device as model
        data = data.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        # Forward
        outputs = model(data)
        loss = criterion(outputs, targets)
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f"Epoch {epoch} Batch {batch_idx} Loss {loss.item():.4f}")
```

## Multi-GPU Training

For larger models or datasets, utilizing multiple GPUs can further accelerate training. PyTorch offers two main approaches:​

1. `DataParallel`: This method first put the data on the main GPU and then split the data across multiple GPUs, which is not efficient. It's straightforward but may not be the most efficient for all scenarios.
2. `DistributedDataParallel`: DDP is more scalable and efficient for multi-GPU training, especially across multiple nodes. It requires more setup but offers better performance.

```python
import torch
# Wrap your model with DataParallel
model = torch.nn.DataParallel(model)
# Move model to the device
model.to(device)
```

```python
from torch.nn.parallel import DistributedDataParallel as DDP
# Initialize the process group
torch.distributed.init_process_group(backend='nccl')
# Wrap your model with DistributedDataParallel
model = DDP(model)
```



