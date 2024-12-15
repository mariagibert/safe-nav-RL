import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from io import BytesIO
from pathlib import Path
from argparse import ArgumentParser

def get_arguments():
    parser = ArgumentParser()
    parser.add_argument('-j', '--json_folder', default=None, type=str)
    parser.add_argument('-d', '--driving', action='store_true')
    arguments = parser.parse_args()
    return arguments

if __name__ == "__main__":
    arguments = get_arguments()
    if arguments.driving:
        columns = ["Trajectory", "Mean Reward", "Time", "Distance", "Collisions"]
        trajectory_files = [item for item in Path(arguments.json_folder).glob('*.json') if 'driving' in item.name]
        dataframe = pd.DataFrame(columns=columns)
        for idx, trajectory_file in enumerate(trajectory_files):
            with open(trajectory_file, 'r') as file:
                data = json.load(file)
            traj_name = trajectory_file.name.split('_')[-1]
            traj_reward = np.array(data['reward']).mean()
            traj_time = np.array(data['time']).mean()
            traj_dist = np.array(data['path_distance']).mean()
            traj_collisions = np.array([len(item) for item in data['collisions']]).mean()
            dataframe.loc[idx] = [traj_name, traj_reward, traj_time, traj_dist, traj_collisions]
        dataframe.to_csv('/home/ubuntu/mgibert/Development/safe-nav-RL/logs/driving_stats_base.csv')
        # Plot data from Driving:
        traj_folders = [item for item in (Path(arguments.json_folder)/'data').iterdir()]
        for idx, traj_folder in enumerate(traj_folders):
            all_dirs = [item for item in traj_folder.iterdir()]
            d_data, phi_data, vel_data, time_data = [], [], [], []
            for directory in all_dirs:
                d_data.append(np.loadtxt(str(directory / 'd.txt')))
                phi_data.append(np.loadtxt(str(directory / 'phi.txt')))
                vel_data.append(np.loadtxt(str(directory / 'vel.txt')))
                time_data.append(np.loadtxt(str(directory / 'time.txt')))
            # Compute mean and std
            min_length = min(len(d) for d in d_data)
            d_data = np.array([d[:min_length] for d in d_data])
            phi_data = np.array([phi[:min_length] for phi in phi_data])
            vel_data = np.array([vel[:min_length] for vel in vel_data])
            time_data = np.array([time[:min_length] for time in time_data])
            # Calculate means and standard deviations
            d_mean, d_std = np.mean(d_data, axis=0), np.std(d_data, axis=0)
            phi_mean, phi_std = np.mean(phi_data, axis=0), np.std(phi_data, axis=0)
            vel_mean, vel_std = np.mean(vel_data, axis=0), np.std(vel_data, axis=0)
            time_mean = np.mean(time_data, axis=0)
            # Plot them
            plt.figure(figsize=(4, 6))
            # Plot d in function of time
            plt.subplot(3, 1, 1)
            plt.plot(time_mean, d_mean, label="d", color='red')
            plt.fill_between(time_mean, d_mean - d_std, d_mean + d_std, color='red', alpha=0.2)
            plt.xlabel("Time")
            plt.ylabel("d")
            plt.legend()

            # Plot phi in function of time
            plt.subplot(3, 1, 2)
            plt.plot(time_mean, phi_mean, label="phi", color='gold')
            plt.fill_between(time_mean, phi_mean - phi_std, phi_mean + phi_std, color='gold', alpha=0.2)
            plt.xlabel("Time")
            plt.ylabel("phi")
            plt.legend()

            # Plot vel in function of time
            plt.subplot(3, 1, 3)
            plt.plot(time_mean, vel_mean, label="vel", color='green')
            plt.fill_between(time_mean, vel_mean - vel_std, vel_mean + vel_std, color='green', alpha=0.2)
            plt.xlabel("Time")
            plt.ylabel("vel")
            plt.legend()

            plt.tight_layout()
            plt.show()
            fig = plt.gcf()  # Get Current Figure
            buffer = BytesIO()
            fig.savefig(buffer, format='png', bbox_inches='tight')
            buffer.seek(0)
            pil_image = Image.open(buffer)
            pil_image.save(f'/home/ubuntu/mgibert/Development/safe-nav-RL/logs/Driving_Trajectory_{idx+1}.png')

    else:
        with open('/home/ubuntu/mgibert/Development/models/test0/evaluation_braking.json', 'r') as file:
            base_data = json.load(file)
        with open('/home/ubuntu/mgibert/Development/models/test1/evaluation_2.json') as file:
            train_data = json.load(file)
        # Compute Reward
        base_reward = np.array(base_data['reward']).mean()
        train_reward = np.array(train_data['reward']).mean()
        # Compute Time
        base_time = np.array(base_data['time']).mean()
        train_time = np.array(train_data['time']).mean()
        # Compute Path Distance
        base_dist = np.array(base_data['path_distance']).mean()
        train_dist = np.array(train_data['path_distance']).mean()
        # Compute Collisions
        base_collisions = np.array([len(item) for item in base_data['collisions']]).mean()
        train_collisions = np.array([len(item) for item in train_data['collisions']]).mean()

        dataframe = pd.DataFrame({
            "Type": ["Baseline", "Trained"],
            "Mean Reward": [base_reward, train_reward],
            "Time": [base_time, train_time],
            "Distance": [base_dist, train_dist],
            "Collisions": [base_collisions, train_collisions]
        })
        dataframe.set_index('Type')
        dataframe.to_csv('/home/ubuntu/mgibert/Development/safe-nav-RL/logs/braking_stats.csv')