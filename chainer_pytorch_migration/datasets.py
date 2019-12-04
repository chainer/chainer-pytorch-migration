import torch


class TransformDataset(torch.utils.data.Dataset):

    def __init__(self, dataset, transform):
        self._dataset = dataset
        self._transform = transform

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, i):
        in_data = self._dataset[i]
        return self._transform(in_data)
