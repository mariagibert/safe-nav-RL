import random
from collections import deque
import numpy as np
import cv2
import math
import time
import tensorflow as tf
import keras.backend.tensorflow_backend as backend
from keras.models import load_model
from braking_dqn_test1 import CarEnv, MEMORY_FRACTION
import carla
from carla import Transform 
from carla import Location
from carla import Rotation
import json



town2 = {1: [80, 306.6, 5, 0], 2:[150,306.6]}
curves = [0, town2]

epsilon = 0

MODEL_PATH = '/home/ubuntu/mgibert/Development/models/test1/Braking___237.00max__237.00avg__237.00min__1733673983.model'


def calculate_distance(point1, point2):
    x1, y1 = point1.x, point1.y
    x2, y2 = point2
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance


if __name__ == '__main__':
    
    FPS = 60
    # Memory fraction
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=MEMORY_FRACTION)
    backend.set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))

    # Load the model
    model = load_model(MODEL_PATH)

    # Create environment
    env = CarEnv()

    # For agent speed measurements - keeps last 60 frametimes
    fps_counter = deque(maxlen=60)

    # Initialize predictions - first prediction takes longer as of initialization that has to be done
    # It's better to do a first prediction then before we start iterating over episode steps
    model.predict(np.zeros((1, 2049)))

    # Connect to the Carla simulator
    client = carla.Client('localhost', 2000)
    client.set_timeout(20.0)
    world = client.get_world()

    # Spawn a vehicle in the world
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.find('vehicle.mini.cooper_s')

    number_of_episodes = 10
    evaluation = {
        "reward": [],
        "time": [],
        "path_distance": [],
        "collisions": []
    }
        

    # Loop over episodes
    for i in range(number_of_episodes):

        print('Restarting episode')
        episode_reward = []

        actor_list = []
        front_car_pos = np.random.randint(90,130)
        spawn_point = carla.Transform(Location(x=front_car_pos, y=306.886, z=5), Rotation(yaw=0))
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        actor_list.append(vehicle)
        #print("direction: ", direction)
        env.waypoint = env.client.get_world().get_map().get_waypoint(Location(x=curves[env.curves][1][0], y=curves[env.curves][1][1], z=curves[env.curves][1][2]), project_to_road=True)
        
        start_time = time.time()
        
        # Reset environment and get initial state
        current_state = env.reset()
        env.collision_hist = []
        env.trajectory()
        done = False

        # Loop over steps
        while True:

            # For FPS counter
            step_start = time.time()

            # Show current frame
            #cv2.imshow(f'Agent - preview', current_state[0])
            #cv2.waitKey(1)

            # Predict an action based on current observation space
            
            #print("action", action)
            if np.random.random() > epsilon:
                # Get action from Q table
                _, velocity, features = current_state
                dqn_input = np.concatenate((features, [[velocity]]), axis=1)
                qs = model.predict(dqn_input)[0]
                action = np.argmax(qs)
            else:
                # Get random action
                action = np.random.randint(0, 2)
                # This takes no time, so we add a delay matching 60 FPS (prediction above takes longer)
                time.sleep(sum(fps_counter)/len(fps_counter))
            # Step environment (additional flag informs environment to not break an episode by time limit)
            new_state, reward, done, waypoint = env.step(action, current_state)
            episode_reward.append(reward)

            # Set current step for next loop iteration
            current_state = new_state
            env.waypoint = waypoint

            # If done - agent crashed, break an episode
            if done:
                end_time = time.time()
                break

            # Measure step time, append to a deque, then print mean FPS for last 60 frames, q values and taken action
            frame_time = time.time() - step_start
            fps_counter.append(frame_time)
            print(f'Agent: {len(fps_counter)/sum(fps_counter):>4.1f} FPS | Action: [{qs[0]:>5.2f}, {qs[1]:>5.2f}] {action}')
        
        # save data in order to evaluate model
        evaluation["reward"].append(sum(episode_reward))
        evaluation["time"].append(end_time - start_time)
        evaluation["collisions"].append([collision.other_actor.type_id for collision in env.collision_history])
        evaluation["path_distance"].append(calculate_distance(env.vehicle.get_transform().location, town2[1][:2]))
            
        # Destroy an actor at end of episode
        for actor in env.actor_list:
            actor.destroy()

        for actor in actor_list:
            actor.destroy()
    
    with open('/home/ubuntu/mgibert/Development/models/test1/evaluation_2.json', 'w') as file:
        json.dump(evaluation, file)
