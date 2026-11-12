import os
from torch.utils.data import Dataset, DataLoader
import torch


def collate_fn(batch):
    # batch: List[(waveform, mag)]
    waveforms, mags = zip(*batch)
    waveforms = torch.stack(waveforms, dim=0)
    mags = torch.stack(mags, dim=0)
    return waveforms, mags


class SeisDataset(Dataset):
    def __init__(self, dataset):
        """
        Args:
            dataset["waveform"]: tensor of shape (N, 3, 3000)
            dataset["mag"]: numpy array of shape (N, 1)
            dataset["meta"]: list of meta dicts
        """
        self.waveforms, self.mags, self.metas = dataset["waveform"], dataset["mag"], dataset["meta"]

    def __len__(self):
        return len(self.waveforms)

    def __getitem__(self, idx):
        waveform = self.waveforms[idx]  # shape: (3, 3000)
        mag = self.mags[idx]  # shape: (1,)

        return waveform, mag


def make_dataset(dataset_path, dataset_type="train"):
    """
    Optimized version: load train/val/test splits separately into memory, stored as arrays
    """
    try:
        print(f"Loading dataset cache: /root/autodl-tmp/{dataset_path}/{dataset_type}")
        """Load the dataset of the specified type"""
        file_path = os.path.join(f"/root/autodl-tmp/{dataset_path}", f"{dataset_type}_dataset.pt")
        data = torch.load(file_path, map_location="cpu", weights_only=False)
        dataset = SeisDataset(data)
        return dataset
    except FileNotFoundError as e:
        print(f"[ERROR] Dataset file not found: /dataset/{dataset_type}")
        raise e
    except Exception as e:
        print("[ERROR] Unknown error occurred while loading the dataset")
        raise e


def make_dataloaders(dataset='5s', batch_size=256, num_workers=4, seed=42):
    """Optimized version of dataloader creation"""
    # Check the cache first; if no cache exists, then filter/split and load the dataset
    # Load the datasets
    train_dataset = make_dataset(dataset, "train")
    val_dataset = make_dataset(dataset, "val")

    # Create DataLoaders - tune num_workers
    # For in-memory data, num_workers=0 is sometimes faster because it avoids inter-process communication
    optimal_workers = 0  # Adjust based on the number of CPU cores

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=optimal_workers, pin_memory=True,
                              collate_fn=collate_fn, persistent_workers=True if optimal_workers > 0 else False)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=optimal_workers, pin_memory=True,
                            collate_fn=collate_fn, persistent_workers=True if optimal_workers > 0 else False)

    return train_loader, val_loader





