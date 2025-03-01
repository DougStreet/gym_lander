import gymnasium as gym
import matplotlib.pyplot as plt 
import time
import numpy as np
import pandas as pd
import keyboard
import random
import torch_directml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import base64, io

io.open

from collections import deque, defaultdict, namedtuple

# For visualization
from gymnasium.utils.save_video import save_video
from IPython.display import HTML
from IPython import display 
import glob

isHuman = False

env = gym.make('LunarLander-v3').env

# Use cuda if available else use cpu
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#device = torch_directml.device()
print(f"Using DirectML device: {device}")

# Basic Parallel PID Class
# Integral accumulates the error and Ki is applied after
# Integral is clamped to max/min int before Ki is applied, so on accumulated error
# Differential is unfiltered, so only useful in ideal simulations
class PIDController:
    def __init__(self, kp, ki, kd, max_int, min_int, min_output=-1.0, max_output=1.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.min_output = min_output
        self.max_output = max_output
        self.max_int = max_int
        self.min_int = min_int
        self.prev_error = 0
        self.integral = 0
        
    def update(self, error, dt):
        # Proportional term
        p_term = self.kp * error
        
        # Integral term with anti-windup
        self.integral = np.clip(self.integral + error * dt, self.min_int, self.max_int)
        i_term = self.ki * self.integral
        
        # Derivative term
        derivative = (error - self.prev_error) / dt
        d_term = self.kd * derivative
        
        # Update previous error
        self.prev_error = error
        
        # Clip output to limits
        output = np.clip(p_term + i_term + d_term, self.min_output, self.max_output)
        return output

def run_lander_with_pid():
    # Test Parameters
    numTests = 50
    testCount = 0
    rewards = np.zeros(50)
    accReward = 0.0

    #Recording Parameters
    step_starting_index = 0
    episode_index = 0
    frames = []

    env = gym.make('LunarLander-v3', continuous=True, render_mode="rgb_array")#, render_mode='human')
    
    # Create three PID controllers with adjusted gains
    # Vertical PID
    vertical_pid = PIDController(
        kp=4.0,
        ki=0.2, #Moderate I to stiffen and remove steady state error
        kd=0.0, #No D as vertical control is trivial with PI
        min_output=0.0,
        max_output=1.0,
        max_int=1000,
        min_int=-1000
    )
    
    # Horizontal PID
    horizontal_pid = PIDController(
        kp=2.25,
        ki=1.0, #Large I to force lander to X coordinate
        kd=4.0, #Big D to factor X velocity
        min_output=-1.0,
        max_output=1.0,
        max_int=1000,
        min_int=-1000
    )
    
    # Angle control
    # Note the higher output than other controllers to ensure
    # this can overpower all others - we need to stay upright...
    angle_pid = PIDController(
        kp=3.5, #Large P to act quickly and proportionately
        ki=0.0000, #No I as the angle PID is best without any windup
        kd=0.5, #A moderate D to reduce wobbling
        min_output=-1.5,
        max_output=1.5,
        max_int=1000,
        min_int=-1000
    )
    
    observation, info = env.reset()
    dt = 1/60.0  # Assuming 60Hz simulation
    
    while testCount < numTests:
        
        # Extract state information
        x, y = observation[0], observation[1]
        vel_x, vel_y = observation[2], observation[3]
        angle = observation[4]
        angular_vel = observation[5]
        leg_l = observation[6]
        leg_r = observation[7]
        
        # Target Parameters
        target_y = 0.2  # Increased target height to maintain altitude
        target_x = 0.0    # Target x position
        fast_x_margin = 0.25 # Distance from target x within which we fall quickly
        target_angle = 0.0  # Target angle (upright)
        
        # Error calculations
        horizontal_error = target_x - x

        #If we're close to horizontal position, descend, otherwise reposition
        if abs(horizontal_error) < fast_x_margin:
            #If we're close to the ground, hover-slam
            if y < target_y:
                #Super slow vertical vel setpoint for slam
                #Note it is very slowly descending just in case we over-correct
                target_vel_y = -0.05
            else:
                #Fast fall rate - tiny fuel burn for control
                target_vel_y = -0.45
        else:
            # Not stopping outside X window, but slow for control
            target_vel_y = -0.2
        
        vertical_error = target_vel_y - vel_y
        angle_error = target_angle - angle
        
        # Get PID outputs
        vertical_thrust = vertical_pid.update(vertical_error, dt)
        horizontal_correction = horizontal_pid.update(horizontal_error, dt)
        angle_correction = angle_pid.update(angle_error, dt)
        
        # Combine horizontal and angle control for side engines
        # Horizontal correction is negative as you need to 
        # invert the horizontal correction
        side_thrust = angle_correction - horizontal_correction
        
        #Cutting thrust once we've touched the ground
        if(leg_l and leg_r):
            #Building continuous action structure
            action = np.array([0.0, -side_thrust])
        else:
            #Building continuous action structure
            action = np.array([vertical_thrust, -side_thrust])
        
        #Run the next frame, make actions and get any observations
        observation, reward, terminated, truncated, info = env.step(action)
        # Collect the frame after each step
        frames.append(env.render())

        #Acculumlating reward
        accReward = accReward + reward
        
        #If environment has reported solved, reset and rerun
        if terminated or truncated:
            rewards[testCount] = accReward
            observation, info = env.reset()
            # Reset all variables
            vertical_pid.integral = 0
            horizontal_pid.integral = 0
            angle_pid.integral = 0
            accReward = 0

            # Save video (if desired) - uncomment above env.render as well...
            save_video(
                frames=frames,
                video_folder="videos",
                fps=env.metadata["render_fps"],
                episode_index=testCount)

            testCount += 1
            
    env.close()

    print(rewards)
    print("min reward: ", rewards.min(), "   max reward: ", rewards.max(), "   avg reward: ", rewards.mean())
    

if __name__ == "__main__":
    run_lander_with_pid()