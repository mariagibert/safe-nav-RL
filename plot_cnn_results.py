import matplotlib.pyplot as plt
import pandas as pd

from PIL import Image
from io import BytesIO
from pathlib import Path


if __name__ == "__main__":
    # open CSV
    root = Path('/home/ubuntu/mgibert/Development/safe-nav-RL/cnn_test2')
    train_loss = root / 'TrainLoss.csv'
    val_loss = root / 'ValLoss.csv'
    val_mae = root / 'ValMae.csv'
    val_r2 = root / 'ValR2.csv'

    # Plot training loss with Validation loss
    train_df = pd.read_csv(str(train_loss))
    val_df = pd.read_csv(str(val_loss))
    # Align training data with validation steps
    aligned_train_df = train_df[train_df['Step'].isin(val_df['Step'])]
    # Merge training and validation DataFrames on Step for easy plotting
    merged_df = pd.merge(val_df, aligned_train_df, on='Step', suffixes=('_val', '_train'))
    merged_df['Smoothed Value Train'] = merged_df['Value_train'].rolling(window=10, min_periods=1).mean()
    merged_df['Smoothed Value Val'] = merged_df['Value_val'].rolling(window=10, min_periods=1).mean()
    step_per_epoch = 999
    merged_df['Epoch'] = merged_df['Step'] // step_per_epoch
    plt.figure(figsize=(10, 8))
    plt.plot(
        merged_df['Epoch'], merged_df['Smoothed Value Train'], '-', 
        label='Train Loss', linewidth=3.5, color='blue'
    )
    plt.plot(
        merged_df['Epoch'], merged_df['Smoothed Value Val'], '-',
        label=f'Val Loss', linewidth=3.5, color='orange'
    )
    plt.title(f'Train and Val Loss for each Epoch')
    plt.xlabel('Epoch')
    plt.ylabel(f'Loss (MSE)')
    plt.legend()
    plt.grid(True)
    plt.ylim(0.08, 0.25)
    plt.tight_layout()
    plt.show()
    fig = plt.gcf()  # Get Current Figure
    buffer = BytesIO()
    fig.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    pil_image = Image.open(buffer)
    pil_image.save(f'Train&ValLoss.png')

    # Plot Mae
    mae_df = pd.read_csv(str(val_mae))
    mae_df['Smoothed Value'] = mae_df['Value'].rolling(window=10, min_periods=1).mean()
    mae_df['Epoch'] = mae_df['Step'] // step_per_epoch
    plt.figure(figsize=(10, 8))
    plt.plot(
        mae_df['Epoch'], mae_df['Smoothed Value'], '-', 
        label='MAE Loss', linewidth=3.5
    )
    plt.title(f'MAE Loss for each Epoch')
    plt.xlabel('Epoch')
    plt.ylabel(f'Loss (MAE)')
    plt.legend()
    plt.grid(True)
    plt.ylim(0.15, 0.35)
    plt.tight_layout()
    plt.show()
    fig = plt.gcf()  # Get Current Figure
    buffer = BytesIO()
    fig.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    pil_image = Image.open(buffer)
    pil_image.save(f'MAELoss.png')

    # Plot R2 
    r2_df = pd.read_csv(str(val_mae))
    r2_df['Smoothed Value'] = r2_df['Value'].rolling(window=10, min_periods=1).mean()
    r2_df['Epoch'] = r2_df['Step'] // step_per_epoch
    plt.figure(figsize=(10, 8))
    plt.plot(
        r2_df['Epoch'], r2_df['Smoothed Value'], '-', 
        label='R2 Loss', linewidth=3.5
    )
    plt.title(f'R2 Loss for each Epoch')
    plt.xlabel('Epoch')
    plt.ylabel(f'Loss (R2)')
    plt.legend()
    plt.grid(True)
    plt.ylim(0.15, 0.35)
    plt.tight_layout()
    plt.show()
    fig = plt.gcf()  # Get Current Figure
    buffer = BytesIO()
    fig.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    pil_image = Image.open(buffer)
    pil_image.save(f'R2Loss.png')

