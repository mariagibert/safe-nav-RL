import json
from pathlib import Path
import numpy as np
from tqdm import tqdm

def analyze_json_data(directory):
    """
    Analyzes JSON files in a directory to calculate statistics for 
    specific keys, using Path and tqdm for progress visualization.

    Args:
        directory: The path to the directory containing the JSON files.

    Returns:
        A dictionary containing the statistics (max, min, std, mean) for 
        'phi', 'dobs', and 'dc' keys.
    """

    phi_values = []
    dobs_values = []
    dc_values = []

    json_files = Path(directory).glob('*.json')
    for filepath in tqdm(json_files, desc="Processing JSON files"):
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                phi_values.append(data['phi'])
                dobs_values.append(data['dobs'])
                dc_values.append(data['dc'])
        except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
            print(f"Error processing file {filepath}: {e}")

    results = {}
    for key, values in zip(['phi', 'dobs', 'dc'], [phi_values, dobs_values, dc_values]):
        results[key] = {
            'max': np.max(values),
            'min': np.min(values),
            'std': np.std(values),
            'mean': np.mean(values)
        }

    return results

# Example usage:
directory_path = '/home/ubuntu/mgibert/Development/safe-nav-RL/cnn_test2/test2_dataset/test_traffic' 
results = analyze_json_data(directory_path)
print(results)


# Results:

results = {
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