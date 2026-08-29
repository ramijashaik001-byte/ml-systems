import random
from typing import List, Tuple

class Dataset:
    def __init__(self, features: List[list], labels: List[list]):
        self.features = features
        self.labels = labels

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[list, list]:
        return self.features[idx], self.labels[idx]

class DataLoader:
    def __init__(self, dataset: Dataset, batch_size: int = 32, shuffle: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = list(range(len(dataset)))

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.indices)
        self.curr = 0
        return self

    def __next__(self) -> Tuple[List[list], List[list]]:
        if self.curr >= len(self.dataset):
            raise StopIteration
        
        batch_indices = self.indices[self.curr : self.curr + self.batch_size]
        self.curr += self.batch_size
        
        features, labels = [], []
        for idx in batch_indices:
            f, l = self.dataset[idx]
            features.append(f)
            labels.append(l)
        return features, labels
