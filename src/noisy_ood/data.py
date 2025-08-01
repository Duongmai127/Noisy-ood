import pandas as pd
import pytorch_lightning as pl
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class ChestDataset(Dataset):
    def __init__(self, df_path, transform=None):
        self.df_path = df_path
        self.transform = transform
        self.dataframe = pd.read_csv(self.df_path)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image_path = row['image_path']
        image = Image.open(image_path).convert('L')
        label = row['class_idx']

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(label).float().unsqueeze(0)

        return image, label

class ChestXrayDataModule(pl.LightningDataModule):
    def __init__(self, train_df, val_df_list, test_df_list, transform_dict, batch_size=32, workers=4, val_multi_noise: bool=False):
        super().__init__()
        self.save_hyperparameters(ignore=['transform_dict'])

        self.train_df_path = train_df
        self.val_df_list = val_df_list
        self.test_df_list = test_df_list
        self.train_transform = transform_dict['train_transform']
        self.test_transform = transform_dict['test_transform']
        self.val_transform_list = transform_dict['val_transform_list']
        self.batch_size = batch_size
        self.num_workers = workers
        self.val_multi_noise = val_multi_noise
    
    def setup(self, stage: str):
        if stage == 'fit':
            self.train_dataset = ChestDataset(self.train_df_path, transform=self.train_transform)
            self.val_dataset_list = [ChestDataset(val_df_path, transform=val_transform) for val_df_path, val_transform in zip(self.val_df_list, self.val_transform_list)]
        
        if stage == 'test':
            self.test_dataset_list = [ChestDataset(test_df_path, transform=self.test_transform) for test_df_path in self.test_df_list]
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return [DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True) for val_dataset in self.val_dataset_list]

    def test_dataloader(self):
        return [DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True) for test_dataset in self.test_dataset_list]
        return [DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True) for test_dataset in self.test_dataset_list]
