#!/usr/bin/env python3
import os
import sys
import torch
import pygame
import numpy as np

# Adjust the paths if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from environment.catcher_image import CatchEnvImage as CatchEnv  # Ensure this matches your training import
from networks.QNetwork import CNNQNetwork  # Ensure this matches your training import

def load_and_play_model(env, model_path="cnn_dqn_model.pth"):
    """
    Loads a trained CNN Q-network and runs it in the given environment.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # The environment is set to grid_size=20 with 2 channels => in_channels=2
    # and action_size = 3 (LEFT, STAY, RIGHT)
    grid_size = env.grid_size
    action_size = env.action_space.n
    in_channels = env.in_channels

    # Create the same model architecture as during training
    model = CNNQNetwork(grid_size=grid_size, action_size=action_size, in_channels=in_channels).to(device)
    
    # Load state dict
    if not os.path.exists(model_path):
        print(f"Model file '{model_path}' not found!")
        return
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print(f"Model successfully loaded from '{model_path}'.")

    # Reset environment
    obs, _ = env.reset()
    done = False
    total_reward = 0.0

    while not done:
        # Render the game in a Pygame window
        env.render()

        # Convert observation to a 4D tensor: (batch=1, channels=2, H=20, W=20)
        # obs is shape (2, 20, 20), we need shape (1, 2, 20, 20)
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)

        # Forward pass: no gradient needed
        with torch.no_grad():
            q_values = model(obs_t)        # shape [1, action_size]
            action = torch.argmax(q_values, dim=1).item()

        # Step the environment
        next_obs, reward, done, _, _ = env.step(action)
        total_reward += reward
        obs = next_obs

        # Check for quit events in Pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                print("Received pygame.QUIT event. Exiting...")
                done = True

    env.close()
    print(f"Game ended. Total reward: {total_reward:.2f}")

def main():
    # Same grid_size you used during training
    env = CatchEnv(grid_size=20)

    # Path to the .pth file you saved during training
    model_path = "cnn_dqn_model.pth"

    load_and_play_model(env, model_path)

if __name__ == "__main__":
    main()
