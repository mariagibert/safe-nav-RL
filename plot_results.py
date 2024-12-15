import matplotlib.pyplot as plt
import pandas as pd

from PIL import Image
from io import BytesIO
from pathlib import Path
from argparse import ArgumentParser

def get_arguments():
    parser = ArgumentParser()
    parser.add_argument('-b', '--base_folder', default=None, type=str)
    parser.add_argument('-t', '--train_folder', default=None, type=str)
    arguments = parser.parse_args()
    return arguments

def plot_model_metrics(data, name, method, save=True):
    plt.figure(figsize=(12, 8))
    if name in ['Reward', 'Loss']:
        plt.plot(data['Step'], data['Value'], label=f'{name}', linewidth=2.5, alpha=0.8)
        # Smooth Data
        data['Smoothed Value'] = data['Value'].rolling(window=10, min_periods=1).mean()
        plt.plot(data['Step'], data['Smoothed Value'], label=f'Smooth Version of {name}', color='orange', linewidth=3.5)
    else:
        plt.plot(data['Step'], data['Value'], label=f'{name}', linewidth=3.5)
    plt.title(f'{name} for each Episode - {method}')
    plt.xlabel('Episode')
    plt.ylabel(f'{name}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    if save:
        fig = plt.gcf()  # Get Current Figure
        buffer = BytesIO()
        fig.savefig(buffer, format='png', bbox_inches='tight')
        buffer.seek(0)
        pil_image = Image.open(buffer)
        pil_image.save(f'{method}-{name}.png')

def compare_model_metrics(base, train, name, model, save=True):
    plt.figure(figsize=(10, 8))
    if name in ['Reward', 'Loss']:
        # Smooth Data
        base['Smoothed Value'] = base['Value'].rolling(window=10, min_periods=1).mean()
        train['Smoothed Value'] = train['Value'].rolling(window=10, min_periods=1).mean()
        plt.plot(
            base['Step'], base['Smoothed Value'], '--', 
            label=f'Smooth Baseline - {name}', linewidth=2.5, color='blue'
        )
        plt.plot(
            train['Step'], train['Smoothed Value'], '--',
            label=f'Smooth Trained - {name}', linewidth=2.5, color='orange'
        )
    plt.plot(base['Step'], base['Value'], label=f'Baseline - {name}', linewidth=3.5, alpha=0.7, color='blue')
    plt.plot(train['Step'], train['Value'], label=f'Trained - {name}', linewidth=3.5, alpha=0.7, color='orange')
    plt.title(f'Comparison between Baseline and Trained {model}-{name} for each Episode')
    plt.xlabel('Episode')
    plt.ylabel(f'{name}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    if save:
        fig = plt.gcf()  # Get Current Figure
        buffer = BytesIO()
        fig.savefig(buffer, format='png', bbox_inches='tight')
        buffer.seek(0)
        pil_image = Image.open(buffer)
        pil_image.save(f'Base_vs_Train_{model}_{name}.png')

if __name__ == "__main__":
    arguments = get_arguments()

    # Identify Inputs
    folders = {'base': arguments.base_folder, 'train': arguments.train_folder}
    if not any(folders.values()): raise ValueError('You need to specify at least one folder.')
    elif folders['base'] and not folders['train']: selected = [folders['base']]
    elif folders['train'] and not folders['base']: selected = [folders['train']]
    elif all(folders.values()): selected = [folders['base'], folders['train']]

    # Plot metrics from single Model
    if len(selected) < 2:
        metric_dataframes = [(pd.read_csv(csv_file), csv_file) for csv_file in Path(selected[0]).glob('*.csv')]
        for dataframe, filename in metric_dataframes:
            name_parts = filename.stem.split('-')
            method_name, plot_name = '-'.join(name_parts[:-1]), name_parts[-1]
            grouped_dataframe = dataframe.groupby('Step')['Value'].mean().reset_index()
            plot_model_metrics(grouped_dataframe, plot_name, method_name)
            
    # Plot metrics from both models
    else:
        # Sort the file paths alphabetically by name
        base_files = sorted(Path(selected[0]).glob('*.csv'), key=lambda x: x.name)
        train_files = sorted(Path(selected[1]).glob('*.csv'), key=lambda x: x.name)
        # Read the files into dataframes, keeping the file names
        base_dataframes = [(pd.read_csv(csv_file), csv_file) for csv_file in base_files]
        train_dataframes = [(pd.read_csv(csv_file), csv_file) for csv_file in train_files]
        for (base_data, base_name), (train_data, train_name) in zip(base_dataframes, train_dataframes):
            plot_name = base_name.stem.split('-')[-1]
            model = base_name.stem.split('-')[1]
            base_gr_data = base_data.groupby('Step')['Value'].mean().reset_index()
            train_gr_data = train_data.groupby('Step')['Value'].mean().reset_index()
            compare_model_metrics(base_gr_data, train_gr_data, plot_name, model)