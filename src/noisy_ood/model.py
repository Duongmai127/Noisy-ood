from collections import defaultdict

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryF1Score,
    BinaryRecall,
    BinarySpecificity,
)
from torchvision import models


# --- Model Architecture (ResNet50 Fine-tuning) ---
class ResNet50FineTune(nn.Module):
    def __init__(self, num_classes=1):
        super(ResNet50FineTune, self).__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(resnet.children())[:-1])

        for name, param in self.features.named_parameters():
            if "bn" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(2048, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(64),
            nn.Linear(64, num_classes),
        )
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Model {self.__class__.__name__}: Total trainable parameters: {total_params / 1000:.1f}K")

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
class ResNet50LightningModule(pl.LightningModule):
    def __init__(self, model, val_idx_to_name, test_idx_to_name, learning_rate=10e-4,  patience=5):
        super().__init__()
        # call this to save learning rate to checkpoint
        # now accessible at self.hparams
        self.save_hyperparameters(ignore=['model'])
        self.model = model

        # define metrics
        self.metrics = MetricCollection([BinaryAUROC(),BinaryAccuracy(),BinaryF1Score(),BinaryRecall(), BinarySpecificity()])

        self.train_metrics = self.metrics.clone(prefix='')
        self.test_metrics = torch.nn.ModuleList()
        self.val_metrics = torch.nn.ModuleList()
        
        self.val_idx_to_name = val_idx_to_name
        self.test_idx_to_name = test_idx_to_name

    def setup(self, stage: str):
        if stage == 'fit':
            num_val_dataloaders = len(self.trainer.datamodule.val_dataloader())
            self.val_metrics = torch.nn.ModuleList([self.metrics.clone('') for i in range(num_val_dataloaders)])
        
        if stage == 'test':
            num_test_dataloaders = len(self.trainer.datamodule.test_dataloader())
            self.test_metrics = torch.nn.ModuleList([self.metrics.clone('') for i in range(num_test_dataloaders)])

    def _shared_step(self, batch):
        # compute loss
        imgs, labels = batch
        labels = labels.float() # Convert labels to float form, (batch_size, 1)
        logits = self.model(imgs)
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        probs = F.sigmoid(logits)
        labels = labels.squeeze(1).long()
        probs = probs.squeeze(1) # (32, 1)

        return loss, probs, labels

    def training_step(self, batch, batch_idx):
        loss, probs, labels = self._shared_step(batch)
        self.train_metrics.update(probs, labels) # compute the train metrics
        self.log("loss/train", loss, on_epoch=True, logger=True)
        return loss

    def on_train_epoch_end(self):
        computed_metrics = self.train_metrics.compute()
        for metric_name, metric_value in computed_metrics.items():
            self.log(f"{metric_name}/train", metric_value, on_epoch=True, logger=True)
        self.train_metrics.reset()
        
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        loss, probs, labels = self._shared_step(batch)
        self.val_metrics[dataloader_idx].update(probs, labels)
        self.log(f"loss/{self.val_idx_to_name[dataloader_idx]}", loss, on_epoch=True, logger=True)
        return {"loss": loss, "dataloader_idx": dataloader_idx}

    def on_validation_epoch_end(self):
        average_metrics_raw_values = defaultdict(list) 

        for i, metrics_collection in enumerate(self.val_metrics):
            computed_metrics = metrics_collection.compute()
            current_val_set_name = self.val_idx_to_name[i]
            for base_metric_name, metric_value in computed_metrics.items():
                self.log(f"{base_metric_name}/{current_val_set_name}", metric_value, logger=True)
                
                average_metrics_raw_values[base_metric_name].append(metric_value.item())
            metrics_collection.reset()

        for base_metric_name, metric_values_list in average_metrics_raw_values.items():
            avg_value = torch.tensor(np.mean(metric_values_list))
            self.log(f"{base_metric_name}/val_avg", avg_value, logger=True)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        loss, probs, labels = self._shared_step(batch)
        self.test_metrics[dataloader_idx].update(probs, labels)
        self.log(f"loss/{self.test_idx_to_name[dataloader_idx]}", loss, on_epoch=True, logger=True)
        return {"loss": loss, "dataloader_idx": dataloader_idx}
    
    def on_test_epoch_end(self):
        for i, metric_collection in enumerate(self.test_metrics):
            computed_metrics = metric_collection.compute()
            current_dataloader_name = self.test_idx_to_name[i]
            print(f"\nResults for: {current_dataloader_name}")
            for metric_name, metric_value in computed_metrics.items():
                self.log(f"{metric_name}/{current_dataloader_name}", metric_value)
                print(f"  {metric_name}: {metric_value.item():.4f}")
            metric_collection.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()), 
            lr=self.hparams.learning_rate
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='max', 
            patience=self.hparams.patience
        )

        scheduler_config = {
            "scheduler": scheduler,
            "monitor": "BinaryAUROC/val_avg",
            "interval": "epoch",
            "frequency": 1,
        }
        
        # Return optimizers and schedulers as separate lists
        return [optimizer], [scheduler_config]