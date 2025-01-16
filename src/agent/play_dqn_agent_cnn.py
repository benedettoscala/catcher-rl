import torch
import pygame
import numpy as np

import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from environment.catcher_cnn import CatchEnv
from agent.dqn_cnn import CNNQNetwork


def load_and_play_model(env, model_path):
    """
    Carica un modello addestrato e gioca mostrando le azioni prese dal modello.
    
    Parameters:
        env: L'ambiente CatchEnv.
        model_path: Percorso del modello salvato.
    """
    # Caricare il modello salvato
    grid_size = env.observation_space.shape[0]  # Es: 20x20
    action_size = env.action_space.n

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNQNetwork(grid_size, action_size).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print(f"Modello caricato da {model_path}. Usando il dispositivo: {device}")

    obs, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        env.render()

        # Convertire lo stato in tensore e passarlo al modello
        state_t = torch.FloatTensor(obs).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = model(state_t)
            action = torch.argmax(q_values, dim=1).item()

        # Aggiornare l'ambiente
        obs, reward, done, _, _ = env.step(action)
        total_reward += reward

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

    env.close()
    print(f"Gioco terminato. Ricompensa totale: {total_reward:.2f}")


if __name__ == "__main__":
    env = CatchEnv(grid_size=20)

    model_path = "cnn_dqn_model.pth"  # Percorso del modello salvato
    if not os.path.exists(model_path):
        print(f"Errore: Il file del modello '{model_path}' non esiste.")
    else:
        load_and_play_model(env, model_path)