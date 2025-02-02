#!/usr/bin/env python3
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

# Importa gli ambienti e la rete Q (assicurati che il path sia corretto)
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from environment.catcher_image import CatchEnvImage, CatchEnvImageChangeDirection
from networks.QNetwork import CNNQNetwork

def test_model(env, model_path, episodes=50):
    """
    Carica il modello da 'model_path' e testa l'agente per un numero di episodi,
    restituendo le ricompense totali e il numero di passi per episodio.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilizzo device: {device}")

    # Determina le dimensioni della griglia, il numero di azioni e i canali in ingresso
    grid_size = env.observation_space.shape[1]  # ricorda: observation.shape = (2, grid_size, grid_size)
    action_size = env.action_space.n
    in_channels = env.in_channels

    # Inizializza la rete e carica i pesi salvati
    model = CNNQNetwork(grid_size, action_size, in_channels).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # Modalità evaluation

    rewards = []
    steps_per_episode = []

    for e in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        done = False

        while not done:
           
            #env.render()
            #modifica il framerate dell'ambiente così da velocizzare il test
            env.metadata["render_fps"] = 10000

            # Converti lo stato in tensore e calcola i Q-values
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)  # shape [1, 2, grid_size, grid_size]
            with torch.no_grad():
                q_values = model(state_tensor)
            # Seleziona l'azione migliore (senza esplorazione)
            action = torch.argmax(q_values, dim=1).item()

            next_state, reward, done, _, _ = env.step(action)
            total_reward += reward
            steps += 1
            state = next_state

        rewards.append(total_reward)
        steps_per_episode.append(steps)
        print(f"Episode {e+1}/{episodes} - Reward: {total_reward}, Steps: {steps}")

    return rewards, steps_per_episode

def plot_results(rewards, steps):
    """
    Visualizza dei grafici della ricompensa e dei passi per episodio.
    """
    episodes = np.arange(1, len(rewards) + 1)
    
    plt.figure(figsize=(12, 5))

    # Grafico della ricompensa per episodio
    plt.subplot(1, 2, 1)
    plt.plot(episodes, rewards, marker='o', linestyle='-', color='blue')
    plt.title("Ricompensa per Episodio")
    plt.xlabel("Episodio")
    plt.ylabel("Ricompensa Totale")
    plt.grid(True)

    # Grafico dei passi per episodio
    plt.subplot(1, 2, 2)
    plt.plot(episodes, steps, marker='o', linestyle='-', color='orange')
    plt.title("Steps per Episodio")
    plt.xlabel("Episodio")
    plt.ylabel("Numero di Steps")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def main():
    print("Scegliere l'ambiente per il test:")
    print("1) CatchEnv")
    print("2) CatchEnvImageChangeDirection")
    choice = input("Inserisci la tua scelta (1 o 2): ")

    savedir = "models\dqn_cnn_models"
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

    try:
        episodes = int(input("Inserisci il numero di episodi di test: "))
    except ValueError:
        print("Inserimento non valido. Verrà usato il valore predefinito di 50 episodi.")
        episodes = 50

    rewards, steps = test_model(env, model_path, episodes)
    env.close()

    # Visualizza i grafici dei risultati
    plot_results(rewards, steps)

if __name__ == "__main__":
    main()
