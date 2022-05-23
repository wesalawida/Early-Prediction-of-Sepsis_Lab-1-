import torch
from torch.utils.data import Dataset


class SepsisDataset(Dataset):

    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence, label, name = self.sequences[idx]
        return torch.Tensor(sequence.to_numpy()), torch.tensor(label), name
