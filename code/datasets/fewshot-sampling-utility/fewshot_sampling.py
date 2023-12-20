import json
import random
import pandas as pd
from torch.utils.data import DataLoader, Subset
import torch

def fewshot_sample(data, sample_percentage):
    """
    Samples the given data by 'sample_percentage'.
    The function automatically detects the type of the data (JSON file or dataset object) and applies appropriate sampling.

    :param data: JSON file path or dataset object.
    :param sample_percentage: The percentage of the data to sample.
    :return: Sampled data.
    """
    if isinstance(data, str) and data.endswith('.json'):
        return sample_json(data, sample_percentage)
    elif isinstance(data, torch.utils.data.Dataset):
        return sample_dataset(data, sample_percentage)
    else:
        raise ValueError("Unsupported data type. Please provide a JSON file path or a dataset object.")

def sample_json(file_path, sample_percentage):
    """
    Samples a JSON file by the given percentage.

    :param file_path: Path to the JSON file.
    :param sample_percentage: The percentage of the data to sample.
    :return: Json file with sampled data.
    """
    random.seed(43)
    with open(file_path, 'r') as f:
        data = json.load(f)
    selected_count = int(len(data) * sample_percentage)
    selected_data = random.sample(data, selected_count)
    output_file = file_path.replace('.json', f'_fewshot{sample_percentage}.json')
    with open(output_file, 'w') as f:
        json.dump(selected_data, f, indent=4)
    return output_file

def sample_dataset(dataset, sample_percentage):
    """
    Samples a dataset by the given percentage.

    :param dataset: dataset from pytorch.
    :param sample_percentage: The percentage of the data to sample.
    :return: Sampled dataset.
    """
    random.seed(43)
    train_size = int(sample_percentage * len(dataset))
    train_indices = random.sample(range(len(dataset)), train_size)
    fewshot_train_dataset = Subset(dataset, train_indices)
    return fewshot_train_dataset