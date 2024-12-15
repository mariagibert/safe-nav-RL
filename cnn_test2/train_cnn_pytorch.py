import json
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
import pytorch_lightning as pl
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import argparse


class CustomDataset(Dataset):

    data_stats = {
    'phi': {
        'max': 179.9998779296875, 
        'min': -179.9999090489, 
        'std': 91.06641637284567, 
        'mean': 3.5729518298494485
    }, 
    'dobs': {
        'max': 10000.0, 
        'min': 30.90290932617188, 
        'std': 3075.504244269763, 
        'mean': 1561.0369180748305
    }, 
    'dc': {
        'max': 161.5544201909273, 
        'min': -97.78582734841427, 
        'std': 33.64658981706105, 
        'mean': -5.3408009442140205
    }}

    def __init__(self, depth_paths, segmt_paths, label_paths, transform=None):
        self.depth_paths = depth_paths
        self.segmt_paths = segmt_paths
        self.label_paths = label_paths
        self.transform = transform

    def __len__(self):
        return len(self.depth_paths)

    def __getitem__(self, idx):
        depth_image = Image.open(self.depth_paths[idx]).convert('RGB')
        segmt_image = Image.open(self.segmt_paths[idx]).convert('RGB')

        if self.transform:
            depth_image = self.transform(depth_image)
            segmt_image = self.transform(segmt_image)

        with open(self.label_paths[idx], 'r') as f:
            label = json.load(f)
        phi = label['phi'] / 180.0
        dobs = 2 * ((label['dobs'] - CustomDataset.data_stats['dobs']['min']) / (CustomDataset.data_stats['dobs']['max'] - CustomDataset.data_stats['dobs']['min'])) - 1
        dc = 2 * ((label['dc'] - CustomDataset.data_stats['dc']['min']) / (CustomDataset.data_stats['dc']['max'] - CustomDataset.data_stats['dc']['min'])) - 1
        label_tensor = torch.tensor([phi, dobs, dc])

        return {'image': torch.cat([depth_image, segmt_image], dim=0), 'label': label_tensor}


class MyModel(pl.LightningModule):
    def __init__(self, lr=0.01):  # Add learning rate as a parameter
        super().__init__()
        self.lr = lr
        # Define your model layers here
        self.conv_layers = nn.Sequential(
            nn.Conv2d(6, 8, kernel_size=(3, 3), padding='same'),  # Input channels = 6 (depth + segmt)
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(8, 16, kernel_size=(3, 3), padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=(3, 3), padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(7*10*32, 16),  # Adjust input size based on your output from conv layers
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Dropout(0.5),
            nn.Linear(16, 4),
            nn.ReLU(),
            nn.Linear(4, 3)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.reshape(x.shape[0], -1) # Flatten
        x = self.fc_layers(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch['image'], batch['label']
        y_hat = self(x)
        loss = nn.MSELoss()(y_hat, y)  # Or nn.L1Loss for MAE
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch['image'], batch['label']
        y_hat = self(x)
        # 1. Calculate Mean Squared Error (MSE)
        mse = nn.MSELoss()(y_hat, y)
        # 2. Calculate Mean Absolute Error (MAE)
        mae = nn.L1Loss()(y_hat, y)
        # 3. Calculate R-squared (R²)
        # (Optional, but useful for regression)
        y_bar = torch.mean(y)
        ss_tot = torch.sum((y - y_bar)**2)
        ss_res = torch.sum((y - y_hat)**2)
        r2 = 1 - (ss_res / ss_tot)
        # Log the metrics
        self.log('val_loss', mse)
        self.log('val_mae', mae)
        self.log('val_r2', r2)  # If you calculated R²

    def test_step(self, batch, batch_idx):
        # You can reuse the validation_step logic for the test_step
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr, weight_decay=0.005)  # Use self.lr here


def load_dataset(dataset_root, train_split=0.8, batch_size=8, random_state=42):
    # Load paths
    depth_paths = sorted([str(path) for path in dataset_root.iterdir() if 'depth_map' in path.stem])
    segmt_paths = sorted([str(path) for path in dataset_root.iterdir() if 'seg_image' in path.stem])
    label_paths = sorted([str(path) for path in dataset_root.iterdir() if path.suffix == '.json'])

    # Get splits
    combined = list(zip(depth_paths, segmt_paths, label_paths))
    (train_data, val_data) = train_test_split(combined, test_size=1 - train_split, random_state=random_state)

    # Unzip the data back into separate lists
    (train_depth_paths, train_segmt_paths, train_label_paths) = zip(*train_data)
    (val_depth_paths, val_segmt_paths, val_label_paths) = zip(*val_data)

    # Create datasets and dataloaders
    train_dataset = CustomDataset(
        train_depth_paths, train_segmt_paths, train_label_paths,
        transform=T.Compose([
            T.Resize((60, 80)),
            T.ToTensor(),
        ])
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = CustomDataset(
        val_depth_paths, val_segmt_paths, val_label_paths,
        transform=T.Compose([
            T.Resize((60, 80)),
            T.ToTensor(),
        ])
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    return train_loader, val_loader


def get_arguments():
    arguments = argparse.ArgumentParser()
    arguments.add_argument('-m', '--mode', choices=['train', 'eval'], required=True)
    arguments.add_argument('-lr', '--learning_rate', default=0.01, type=float)
    arguments.add_argument('-o', '--output_path', type=str)  # Not used in this example
    arguments.add_argument('-d', '--dataset_root', type=str)
    arguments.add_argument('-e', '--epochs', type=int)
    arguments.add_argument('-ckpt', '--checkpoint_path', type=str, default=None)
    arguments.add_argument('-b', '--batch_size', type=int, default=32)
    return arguments.parse_args()


if __name__ == "__main__":
    arguments = get_arguments()

    # Load Dataset
    train_loader, val_loader = load_dataset(Path(arguments.dataset_root, batch_size = arguments.batch_size))

    # Build Model
    model = MyModel(lr=arguments.learning_rate)  # Pass learning rate to the model

    if arguments.mode == 'train':
        # Early Stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',  # Monitor validation loss
            patience=5,           # Number of epochs with no improvement after which training will be stopped
            mode='min'            # 'min' because you want to minimize validation loss
        )
        # Model Checkpoint
        checkpoint_callback = ModelCheckpoint(
            dirpath=arguments.output_path,  # Directory to save checkpoints 
            filename='best_model',  # Filename for the best model
            save_weights_only=False,  # Save the entire model (not just weights)
            monitor='val_loss',  # Monitor validation loss
            mode='min',  # 'min' because you want to minimize validation loss
        )
        logger = TensorBoardLogger("tb_logs", name="cnn_test2")
        # Initialize Trainer
        trainer = pl.Trainer(
            max_epochs=arguments.epochs,
            gpus=1,
            logger=logger,
            callbacks=[early_stopping, checkpoint_callback])  # Adjust gpus if needed
        # Train the model
        trainer.fit(model, train_loader, val_loader)

    elif arguments.mode == 'eval':
        if not arguments.checkpoint_path: 
            raise ValueError('You should specify the checkpoint to evaluate')
        # Load checkpoint
        model = MyModel.load_from_checkpoint(arguments.checkpoint_path)
        # Evaluate the model (you'll need to implement an evaluation loop here)
        # For example:
        trainer = pl.Trainer(gpus=1)  # Adjust gpus if needed
        trainer.test(model, val_loader)  # This assumes you have a test_step defined in your model