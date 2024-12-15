from __future__ import print_function

import time
import numpy as np
import math
import json
import random
import cv2

from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from agents.navigation.global_route_planner import GlobalRoutePlanner

import carla

from carla import Transform 
from carla import Location
from carla import Rotation


town2 = {
    1: {1: [80, 306.6, 5, 0], 2:[135.25,206]}, # trajectory 1
    2: {1: [-7.498, 284.716, 5, 90], 2:[81.98,241.954]}, # trajectory 2
    3: {1: [-7.498, 165.809, 5, 90], 2:[81.98,241.954]}, # trajectory 3
    4: {1: [106.411, 191.63, 5, 0], 2:[170.551,240.054]}, # trajectory 4
    5: {1: [80, 306.6, 5, 180], 2:[81.98, 241.954]}, # extra
    6: {1: [106.411, 191.63, 5, 0], 2:[-7.498, 284.716]}, # extra
    7: {1: [-7.498, 200.0, 5, 270], 2:[35.25,206]},
    8: {1: [35.25, 206, 5, 90], 2:[81.98,241.954]},
    9: {1: [50,241.954, 5, 0], 2:[80, 306.6]},
    10: {1:[193.75,269.2610168457031, 5, 270], 2:[135.25,206]}
}

IM_WIDTH = 640
IM_HEIGHT = 480

class CarEnv:

    STEER_AMT = 1.0   # actions that the agent can take [-1, 0, 1] --> [turn left, go straight, turn right]
    front_camera = None

    def __init__(self, save_root, tick):
        # to initialize
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(20.0)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.model_3 = self.blueprint_library.find("vehicle.tesla.model3")
        self.front_model3 = self.blueprint_library.find("vehicle.tesla.model3")
        self.via = 2
        self.crossing = 0
        self.curves = 1
        self.reached = 0
        self.actor_list = []
        self.save_root = save_root
        self.tick=tick

    def image_dep(self, image):
        self.cam = image
        
    def image_seg(self, image):
        self.seg = image
    
    def collision_data(self, event):
        self.collision_history.append(event)
    
    def set_spawn_point_and_trajectory(self):
        random_trajectory = random.choice(list(town2.keys()))
        self.start = town2[random_trajectory][1]
        self.traj = random_trajectory

    def iniciate_agent_with_sensors(self):
        '''
        To spawn the Vehicle (agent)
        '''
        initial_pos = self.start
        self.transform = Transform(Location(x=initial_pos[0], y=initial_pos[1], z=initial_pos[2]), Rotation(yaw=initial_pos[3]))
        # to spawn the actor; the veichle
        self.vehicle = self.world.spawn_actor(self.model_3, self.transform)
        self.actor_list.append(self.vehicle)

        '''
        To spawn the Segmentation camera
        '''
        self.seg_camera = self.blueprint_library.find("sensor.camera.semantic_segmentation")
        self.seg_camera.set_attribute("image_size_x", f"{IM_WIDTH}")
        self.seg_camera.set_attribute("image_size_y", f"{IM_HEIGHT}")
        self.seg_camera.set_attribute("fov", f"40")

        # to spawn the segmentation camera exactly in between the 2 depth cameras
        self.seg_camera_spawn_point = carla.Transform(carla.Location(x=2, y = 0, z=1.4), Rotation(yaw=0))
        
        # to spawn the camera
        self.seg_camera_sensor = self.world.spawn_actor(self.seg_camera, self.seg_camera_spawn_point, attach_to = self.vehicle)
        #print("Segmentation camera image sent for processing....")
        self.actor_list.append(self.seg_camera_sensor)

        self.seg_camera_sensor.listen(lambda data: self.image_seg(data))

        '''
        To spawn the Depth Image
        '''
         # to use the depth camera
        self.depth_camera = self.blueprint_library.find("sensor.camera.depth")
        self.depth_camera.set_attribute("image_size_x", f"{IM_WIDTH}")
        self.depth_camera.set_attribute("image_size_y", f"{IM_HEIGHT}")
        self.depth_camera.set_attribute("fov", f"40")

        self.camera_spawn_point = carla.Transform(carla.Location(x=2, y = 0, z=1.4), Rotation(yaw=0))

        # to spawn the camera
        self.camera_sensor = self.world.spawn_actor(self.depth_camera, self.camera_spawn_point, attach_to = self.vehicle)
        self.actor_list.append(self.camera_sensor)

        # to record the data from the camera sensor
        self.camera_sensor.listen(lambda data: self.image_dep(data))

        traj = self.trajectory()
        self.path = []
        for el in traj:
            self.path.append(el[0])

        '''
        To spawn the collision sensor
        '''
        # to introduce the collision sensor to detect what type of collision is happening
        col_sensor = self.blueprint_library.find("sensor.other.collision")
        
        # keeping the location of the sensor to be same as that of the RGB camera
        self.collision_sensor = self.world.spawn_actor(col_sensor, self.camera_spawn_point, attach_to = self.vehicle)
        self.actor_list.append(self.collision_sensor)

        # to record the data from the collision sensor
        self.collision_sensor.listen(lambda event: self.collision_data(event))
        self.collision_history = []


    def start_driving(self):
        self.vehicle.set_autopilot(True)

    def capture_data(self):

        """
        Calculate data
        """
        dis, depth_map, seg_array = self.process_images()
        phi, dc = self.compute_waypoints()

        """
        Save Data
        """
        cv2.imwrite(str(self.save_root / f'{str(self.tick).zfill(5)}_depth_map.png'), depth_map)
        cv2.imwrite(str(self.save_root / f'{str(self.tick).zfill(5)}_seg_image.png'), seg_array)
        with open(str(self.save_root / f'{str(self.tick).zfill(5)}_data.json'), 'w') as file:
            json.dump({'phi': phi, 'dobs': dis, 'dc': dc}, file)

    def process_images(self):        
        # Convert depth image to array of depth values
        depth_array1 = np.frombuffer(self.cam.raw_data, dtype=np.dtype("uint8"))
        depth_array1 = np.reshape(depth_array1, (self.cam.height, self.cam.width, 4))
        depth_array1 = depth_array1.astype(np.int32)
        
        # Using this formula to get the distances
        depth_map = (depth_array1[:, :, 0]*255*255 + depth_array1[:, :, 1]*255 + depth_array1[:, :, 2])/1000
        
        # Making the sky at 0 distance
        x = np.where(depth_map >= 16646.655)
        depth_map[x] = 0

        # Calculate distance from camera to each point in world coordinates
        distances = depth_map

        # Process segmentation image       
        image_array = np.frombuffer(self.seg.raw_data, dtype=np.dtype("uint8"))
        image_array = np.reshape(image_array, (self.seg.height, self.seg.width, 4))
        
        # removing the alpha channel
        image_array = image_array[:, :, :3]
        
        colors = {
            0: [0, 0, 0],         # None
            1: [70, 70, 70],      # Buildings
            2: [190, 153, 153],   # Fences
            3: [72, 0, 90],       # Other
            4: [220, 20, 60],     # Pedestrians
            5: [153, 153, 153],   # Poles
            6: [157, 234, 50],    # RoadLines
            7: [128, 64, 128],    # Roads
            8: [244, 35, 232],    # Sidewalks
            9: [107, 142, 35],    # Vegetation
            10: [0, 0, 255],      # Vehicles
            11: [102, 102, 156],  # Walls
            12: [220, 220, 0],    # TrafficSigns
        }
        seg_array = np.zeros_like(image_array)
        for label, color in colors.items():
            seg_array = np.where(image_array[:, :, 2, None] == label, color, seg_array)

        for key in colors:
            # to store the vehicle indices only
            if key == 10:
                self.vehicle_indices = np.where((image_array == [0, 0, key]).all(axis = 2))

        if len(self.vehicle_indices[0]) != 0:
            dis = np.sum(distances[self.vehicle_indices])/len(self.vehicle_indices[0])
        else:
            dis = 10000
        return dis, depth_map, seg_array

    def compute_waypoints(self):
        # to calculate the kmh of the vehicle
        v = self.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))
        
        # to get the position and orientation of the car
        pos = self.vehicle.get_transform().location
        rot = self.vehicle.get_transform().rotation
        
        # to get the closest waypoint to the car
        waypoint = self.client.get_world().get_map().get_waypoint(pos, project_to_road=True)
        waypoint_ind = self.get_closest_waypoint(self.path, waypoint) + 1
        waypoint = self.path[waypoint_ind]
        if len(self.path) != 1:
            next_waypoint = self.path[waypoint_ind+1]
        else:
            next_waypoint = waypoint
        waypoint_loc = waypoint.transform.location
        waypoint_rot = waypoint.transform.rotation
        next_waypoint_loc = next_waypoint.transform.location
        
        # to get the orientation difference between the car and the road "phi"
        orientation_diff = waypoint_rot.yaw - rot.yaw
        phi = orientation_diff%360 -360*(orientation_diff%360>180)
        
        u = [waypoint_loc.x-next_waypoint_loc.x, waypoint_loc.y-next_waypoint_loc.y]
        v = [pos.x-next_waypoint_loc.x, pos.y-next_waypoint_loc.y]
        if np.linalg.norm(u) > 0.1 and np.linalg.norm(v) > 0.1:
            signed_dis = np.linalg.norm(v)*np.sin(np.sign(np.cross(u,v))*np.arccos(np.dot(u,v)/(np.linalg.norm(u)*np.linalg.norm(v))))
        else:
            signed_dis = 0

        return phi, signed_dis
    
    def trajectory(self, draw = False):
        amap = self.world.get_map()
        sampling_resolution = 0.5
        # dao = GlobalRoutePlannerDAO(amap, sampling_resolution)
        grp = GlobalRoutePlanner(amap, sampling_resolution)
        # grp.setup()
        
        #start_location = self.vehicle.get_transform().location
        start_location = carla.Location(x=self.start[0], y=self.start[1], z=0)
        end_location = carla.Location(x=town2[self.traj][2][0], y=town2[self.traj][2][1], z=0)
        a = amap.get_waypoint(start_location, project_to_road=True)
        b = amap.get_waypoint(end_location, project_to_road=True)
        spawn_points = self.world.get_map().get_spawn_points()
        #print(spawn_points)
        a = a.transform.location
        b = b.transform.location
        w1 = grp.trace_route(a, b) # there are other funcations can be used to generate a route in GlobalRoutePlanner.
        i = 0
        if draw:
            for w in w1:
                if i % 10 == 0:
                    self.world.debug.draw_string(w[0].transform.location, 'O', draw_shadow=False,
                    color=carla.Color(r=255, g=0, b=0), life_time=120.0,
                    persistent_lines=True)
                else:
                    self.world.debug.draw_string(w[0].transform.location, 'O', draw_shadow=False,
                    color = carla.Color(r=0, g=0, b=255), life_time=1000.0,
                    persistent_lines=True)
                i += 1
        return w1

    def get_closest_waypoint(self, waypoint_list, target_waypoint):
        closest_waypoint = None
        closest_distance = float('inf')
        for i, waypoint in enumerate(waypoint_list):
            distance = math.sqrt((waypoint.transform.location.x - target_waypoint.transform.location.x)**2 +
                                 (waypoint.transform.location.y - target_waypoint.transform.location.y)**2)
            if distance < closest_distance:
                closest_waypoint = i
                closest_distance = distance
        return closest_waypoint


if __name__ == '__main__':
    save_root = Path('/home/ubuntu/mgibert/Development/safe-nav-RL/cnn_test2/test2_dataset/test_traffic')
    save_root.mkdir(exist_ok=True, parents=True)
    previous_ticks = [path.stem for path in save_root.iterdir()]
    if previous_ticks:
        tick = int(sorted([path.stem for path in save_root.iterdir()])[-1].split("_")[0])
    else:
        tick=0
    max_dataset_items = 10000
    env = CarEnv(save_root=save_root, tick=tick)
    while env.tick < max_dataset_items:
        failed=False
        try:
            env.set_spawn_point_and_trajectory()
            print(env.traj)
            env.iniciate_agent_with_sensors()
            print('Start')
            env.start_driving()
            while len(env.collision_history) == 0 and env.tick < max_dataset_items and not failed:
                try:
                    if env.vehicle.is_at_traffic_light():
                        if env.vehicle.get_traffic_light().get_state() == carla.TrafficLightState.Red:
                            env.vehicle.set_target_velocity(carla.Vector3D(2.5,0,0))
                    env.capture_data()
                    env.world.tick()
                    env.tick += 1
                    print('Collected')
                    time.sleep(1)
                except Exception:
                    failed=True
            for actor in env.actor_list:
                actor.destroy()
            env.actor_list = []
            print("All actors have been killed")
        except:
            for actor in env.actor_list:
                actor.destroy()
            print("All actors have been killed")