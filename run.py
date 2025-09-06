# This file is modified from <https://github.com/cjy1992/gym-carla.git>:
# Copyright (c) 2019: Jianyu Chen (jianyuchen@berkeley.edu)
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import gym
import gym_carla
import carla
from stable_baselines import DQN
from stable_baselines3 import PPO
from stable_baselines.deepq.policies import MlpPolicy

def main():
  # parameters for the gym_carla environment
  params = {
    'number_of_vehicles': 1,
    'number_of_walkers': 5,
    'display_size': 256,  # screen size of bird-eye render
    'max_past_step': 1,  # the number of past steps to draw
    'dt': 0.1,  # time interval between two frames
    'discrete': True,  # whether to use discrete control space
    'discrete_acc': [-3.0, 0.0, 3.0],  # discrete value of accelerations
    'discrete_steer': [-0.2, 0.0, 0.2],  # discrete value of steering angles
    'continuous_accel_range': [-3.0, 3.0],  # continuous acceleration range
    'continuous_steer_range': [-0.3, 0.3],  # continuous steering angle range
    'ego_vehicle_filter': 'vehicle.lincoln*',  # filter for defining ego vehicle
    'port': 4000,  # connection port
    'town': 'Town03',  # which town to simulate
    'max_time_episode': 1000,  # maximum timesteps per episode
    'max_waypt': 12,  # maximum number of waypoints
    'obs_range': 32,  # observation range (meter)
    'lidar_bin': 0.125,  # bin size of lidar sensor (meter)
    'd_behind': 12,  # distance behind the ego vehicle (meter)
    'out_lane_thres': 2.0,  # threshold for out of lane
    'desired_speed': 8,  # desired speed (m/s)
    'max_ego_spawn_times': 200,  # maximum times to spawn ego vehicle
    'display_route': True,  # whether to render the desired route
  }

  # Set gym-carla environment
  env = gym.make('carla-v0', params=params)

  model = DQN(MlpPolicy, env, verbose=1, tensorboard_log="./tensorboard/")
  model.learn(total_timesteps=10000)

  obs = env.reset()
  i = 0
  while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    print(i)
    i += 1

if __name__ == '__main__':
  main()
