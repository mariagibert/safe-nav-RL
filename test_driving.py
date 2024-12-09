import os
import random
from collections import deque
import numpy as np
import cv2
import time
import tensorflow as tf
import keras.backend.tensorflow_backend as backend
from keras.models import load_model
from driving_dqn import CarEnv, MEMORY_FRACTION
import math
import json

epsilon = 0.05

MODEL_PATH = "/home/ubuntu/mgibert/Development/models/test0/Driving__4667.00max_4667.00avg_4667.00min__1733761661.model"

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
    model.predict(np.array([[0,0]]))

    number_of_episodes = 10
    evaluation = {
        "reward": [],
        "time": [],
        "collisions": [],
        "path_distance": []
    }

    # Loop over episodes
    for i in range(number_of_episodes):

        print('Restarting episode')

        start_time = time.time()
        episode_reward = []

        # Reset environment and get initial state
        current_state = env.reset()
        env.collision_hist = []
        env.trajectory()
        done = False
        env.phi = []
        env.dc = []
        env.vel = []
        env.time = []
        j = 0
        # Loop over steps
        while True:

            # For FPS counter
            step_start = time.time()

            # Show current frame
            #cv2.imshow(f'Agent - preview', current_state[0])
            #cv2.waitKey(1)

            # Predict an action based on current observation space
            
            #print("action", action)
            if np.random.random() > epsilon or j == 0:
                # Get action from Q table
                qs = model.predict(np.array(current_state).reshape(-1, *np.array(current_state).shape))[0]
                action = np.argmax(qs)
                
            else:
                # Get random action
                action = np.random.randint(0, 5)
                # This takes no time, so we add a delay matching 60 FPS (prediction above takes longer)
                time.sleep(sum(fps_counter)/len(fps_counter))

            # Step environment (additional flag informs environment to not break an episode by time limit)
            new_state, reward, done, waypoint = env.step(action, current_state)
            episode_reward.append(reward)

            # Set current step for next loop iteration
            current_state = new_state
            env.waypoint = waypoint
            j += 1
            # If done - agent crashed, break an episode
            if done:
                end_time = time.time()
                break

            # Measure step time, append to a deque, then print mean FPS for last 60 frames, q values and taken action
            frame_time = time.time() - step_start
            fps_counter.append(frame_time)
            print(f'Agent: {len(fps_counter)/sum(fps_counter):>4.1f} FPS | Action: [{qs[0]:>5.2f}, {qs[1]:>5.2f}, {qs[2]:>5.2f}, {qs[3]:>5.2f}, {qs[4]:>5.2f}] {action}')

        evaluation["reward"].append(sum(episode_reward))
        evaluation["time"].append(end_time - start_time)
        evaluation["collisions"].append([collision.other_actor.type_id for collision in env.collision_history])
        evaluation["path_distance"].append(calculate_distance(env.vehicle.get_transform().location, env.start[:2])) 
        
        os.mkdir("/home/ubuntu/mgibert/Development/models/test0/data/traj2/file"+str(i))
        np.savetxt("/home/ubuntu/mgibert/Development/models/test0/data/traj2/file"+str(i)+"/phi.txt", env.phi)
        np.savetxt("/home/ubuntu/mgibert/Development/models/test0/data/traj2/file"+str(i)+"/d.txt", env.dc)
        np.savetxt("/home/ubuntu/mgibert/Development/models/test0/data/traj2/file"+str(i)+"/vel.txt", env.vel)
        np.savetxt("/home/ubuntu/mgibert/Development/models/test0/data/traj2/file"+str(i)+"/time.txt", env.time)
            

        # Destroy an actor at end of episode
        for actor in env.actor_list:
            actor.destroy()
    
    with open('/home/ubuntu/mgibert/Development/models/test0/evaluation_driving_traj2.json', 'w') as file:
        json.dump(evaluation, file)