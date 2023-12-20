# Fewshot Sampling Utility

This utility provides a simple way to perform fewshot sampling on either a JSON file or a PyTorch dataset. Fewshot sampling involves selecting a small percentage of the total data for training.
## Usage

The utility supports two types of data: JSON files and PyTorch datasets. It automatically detects the type of data provided and performs appropriate sampling.
## Examples
### For JSON File
To sample a JSON file, use the following:
```python
from datasets.fewshot_sampling_utility.fewshot_sampling import fewshot_sample

file_path = 'data.json'  # Replace with your JSON file path
sample_percentage = 0.1  # Replace with your desired fewshot sample percentage
sampled_file_path = fewshot_sample(file_path, sample_percentage)
# print("Sampled JSON File:", sampled_file_path)
```

### For PyTorch Dataset
For PyTorch datasets, use the utility as follows:

```python
from datasets.fewshot_sampling_utility.fewshot_sampling import fewshot_sample

import torchvision
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader

data_transform = torchvision.transforms.Compose([...]) # Replace with your desired data transformation
datadir = "MobileFM/code/datasets/cifar100"
train_dataset = CIFAR100(datadir, train=True, download=True, transform=data_transform)
sample_percentage = 0.1  # Replace with your desired fewshot sample percentage
sampled_dataset = fewshot_sample(train_dataset, sample_percentage)
# then the 'sampled_dataset' can be used by DataLoader.
```