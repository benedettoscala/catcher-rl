#!/usr/bin/env python3
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from environment.catcher_image import CatchEnvImage, CatchEnvImageChangeDirection
from networks.QNetwork import CNNQNetwork


def play_agent(env, model_path, episodes=5):
    """
    Carica il modello da 'model_path' e fa giocare l'agente per 'episodes' episodi,
    rendendo visibile il gameplay all'utente.
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilizzo device: {device}")

    # Dimensioni e in_channels dell'osservazione
    grid_size = env.observation_space.shape[1]
    action_size = env.action_space.n
    in_channels = env.in_channels

    # Inizializza la rete e carica i pesi salvati
    model = CNNQNetwork(grid_size, action_size, in_channels).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # Modalità evaluation

    for episode in range(1, episodes + 1):
        state, _ = env.reset()
        done = False
        total_reward = 0

        print(f"\n--- Episodio {episode} ---")

        while not done:
            env.render()

            # Converti lo stato in tensore
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            
            # Calcola i Q-values e seleziona l'azione migliore
            with torch.no_grad():
                q_values = model(state_tensor)
            action = torch.argmax(q_values, dim=1).item()

            # Esegui l'azione
            next_state, reward, done, _, _ = env.step(action)
            total_reward += reward
            state = next_state

        print(f"Ricompensa totale nell'episodio {episode}: {total_reward}")

    env.close()


def main():
    print("Scegliere l'ambiente per il test:")
    print("1) CatchEnvImage (no direction)")
    print("2) CatchEnvImageChangeDirection (direction)")
    choice = input("Inserisci la tua scelta (1 o 2): ")

    # Percorso dove si trovano i modelli salvati
    savedir = "models/dqn_cnn_models"

    if choice == "1":
        env = CatchEnvImage(grid_size=15)
        model_path = os.path.join(savedir, "no_direction", "dqn_model_final.pth")
    elif choice == "2":
        env = CatchEnvImageChangeDirection(grid_size=15)
        model_path = os.path.join(savedir, "direction", "dqn_model_final.pth")
    else:
        print("Scelta non valida. Uscita.")
        return

    if not os.path.exists(model_path):
        print(f"Il file del modello non è stato trovato in: {model_path}")
        return

    # Chiedi all'utente il numero di episodi da giocare (default 5 se non valido)
    try:
        episodes = int(input("Inserisci il numero di episodi da giocare: "))
    except ValueError:
        print("Inserimento non valido. Verrà usato il valore predefinito di 5 episodi.")
        episodes = 5

    play_agent(env, model_path, episodes)


if __name__ == "__main__":
    main()
