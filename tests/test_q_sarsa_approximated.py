#!/usr/bin/env python3
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from environment.catcher_discrete import CatchEnv, CatchEnvChangeDirection

import torch.nn as nn
from networks.QNetwork import QNetwork

def state_to_tensor(state, direction):
    if direction:
        state_flat = list(state)
    else:
        state_flat = list(state)
    return torch.FloatTensor(state_flat).unsqueeze(0)

def test_model(env, model, episodes, direction, device):
    rewards = []
    steps_per_episode = []
    success_rates = []  # Tasso di successo (oggetti catturati / totali)
    lives_remaining = []  # Vite rimanenti alla fine di ogni episodio
    caught_objects = []  # Numero di oggetti presi
    missed_objects = []  # Numero di oggetti mancati
    malicious_catches = []  # Numero di bombe catturate
    
    model.eval()
    
    for e in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        caught = 0
        total_objects = 0
        done = False
        
        while not done:
            env.render()
            #env.metadata["render_fps"] = 10000
            state_tensor = state_to_tensor(state, direction).to(device)
            with torch.no_grad():
                q_values = model(state_tensor)
            action = torch.argmax(q_values, dim=1).item()
            
            next_state, reward, done, _, info = env.step(action)
            total_reward += reward
            steps += 1
            
            if reward > 0:
                caught += 1
            total_objects += 1
            state = next_state
        
        rewards.append(total_reward)
        steps_per_episode.append(steps)
        success_rates.append(env.caught_objects / max(1, (env.caught_objects + env.missed_objects)))
        lives_remaining.append(env.lives)
        caught_objects.append(env.caught_objects)
        missed_objects.append(env.missed_objects)
        malicious_catches.append(env.malicious_catches)
        
        print(f"Episode {e+1}/{episodes} - Reward: {total_reward}, Steps: {steps}, Success Rate: {success_rates[-1]*100:.2f}%, Lives Remaining: {env.lives}, Caught: {env.caught_objects}, Missed: {env.missed_objects}, Malicious Catches: {env.malicious_catches}")

    return rewards, steps_per_episode, success_rates, lives_remaining, caught_objects, missed_objects, malicious_catches

def plot_results(rewards, steps, success_rates, lives_remaining, caught_objects, missed_objects, malicious_catches):
    episodes = np.arange(1, len(rewards) + 1)
    
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 3, 1)
    plt.boxplot(rewards)
    plt.title("Ricompensa per Episodio")
    plt.ylabel("Ricompensa Totale")
    plt.grid(True)

    plt.subplot(2, 3, 2)
    plt.boxplot(steps)
    plt.title("Steps per Episodio")
    plt.ylabel("Numero di Steps")
    plt.grid(True)

    plt.subplot(2, 3, 3)
    plt.boxplot(np.array(success_rates) * 100)
    plt.title("Tasso di Successo (%)")
    plt.ylabel("Success Rate (%)")
    plt.grid(True)

    plt.subplot(2, 3, 4)
    plt.boxplot(lives_remaining)
    plt.title("Vite Rimanenti")
    plt.ylabel("Numero di Vite")
    plt.grid(True)

    plt.subplot(2, 3, 5)
    plt.boxplot(caught_objects)
    plt.title("Oggetti Presi")
    plt.ylabel("Numero di Oggetti Presi")
    plt.grid(True)

    plt.subplot(2, 3, 6)
    plt.boxplot(missed_objects)
    plt.title("Oggetti Mancati")
    plt.ylabel("Numero di Oggetti Mancati")
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def main():
    print("Scegliere l'ambiente per il test:")
    print("1) CatchEnv")
    print("2) CatchEnvChangeDirection")
    choice = input("Inserisci la tua scelta (1 o 2): ")
    
    if choice == "1":
        env = CatchEnv(grid_size=15)
        direction = False
    elif choice == "2":
        env = CatchEnvChangeDirection(grid_size=15)
        direction = True
    else:
        print("Scelta non valida. Uscita.")
        return
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilizzo device: {device}")
    
    input_size = 11 if direction else 9
    hidden_sizes = [128, 128]
    num_actions = env.action_space.n
    
    model = QNetwork(input_size, hidden_sizes, num_actions).to(device)
    
    model_path = "models/best_sarsa_approximated/direction/q_network_final.pth" if direction else "models/best_sarsa_approximated/no_direction/q_network_final.pth"
    if not os.path.exists(model_path):
        print(f"Modello non trovato in {model_path}")
        return
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    try:
        episodes = int(input("Inserisci il numero di episodi di test: "))
    except ValueError:
        print("Input non valido. Verranno eseguiti 50 episodi di test.")
        episodes = 50
    
    rewards, steps, success_rates, lives_remaining, caught_objects, missed_objects, malicious_catches = test_model(env, model, episodes, direction, device)
    env.close()
    plot_results(rewards, steps, success_rates, lives_remaining, caught_objects, missed_objects, malicious_catches)


if __name__ == "__main__":
    main()