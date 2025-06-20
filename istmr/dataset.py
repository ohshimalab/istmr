import os

import pandas as pd
import pytorch_lightning as pl
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class LoRADataset(Dataset):
    def __init__(
        self,
        root_dir,
        image_paths,
        labels,
        num_classes,
        transform=None,
    ):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        for image_path, label in zip(image_paths, labels):
            self.samples.append((os.path.join(root_dir, image_path), label))
        self.num_classes = num_classes

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label_onehot = torch.zeros(self.num_classes, dtype=torch.float)
        label_onehot[label] = 1.0
        return (
            img,
            label_onehot,
            torch.IntTensor(list(range(self.num_classes))),
        )


class LoRADataModule(pl.LightningDataModule):
    def __init__(
        self,
        root_dir,
        df_data,
        train_transform,
        test_transform,
        batch_size,
        num_workers,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.df_data = df_data
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.batch_size = batch_size
        self.num_workers = num_workers

    def _get_dataset_from_df(self, df, is_train) -> LoRADataset:
        return LoRADataset(
            root_dir=self.root_dir,
            image_paths=df["generated_image_path"].values.tolist(),
            labels=df["label"].values.tolist(),
            num_classes=self.num_classes,
            transform=self.train_transform if is_train else self.test_transform,
        )

    def setup(self, stage=None):
        self.df_data_train = self.df_data[self.df_data["split_type"].isin(["train", "val"])]

        self.df_data_test = self.df_data[self.df_data["split_type"] == "test"]

        self.num_classes = len(self.df_data["label"].unique().tolist())

        self.train_ds = self._get_dataset_from_df(self.df_data_train, is_train=True)
        self.test_ds = self._get_dataset_from_df(self.df_data_test, is_train=False)

    def _create_dataloader(self, ds, shuffle=False):
        return DataLoader(
            ds,
            self.batch_size,
            shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def train_dataloader(self):
        return self._create_dataloader(self.train_ds, True)

    def test_dataloader(self):
        return self._create_dataloader(self.test_ds, False)
