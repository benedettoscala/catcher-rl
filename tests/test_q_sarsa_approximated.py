#!/usr/bin/env python3
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

# Aggiungi il percorso del modulo (assicurati che la struttura delle cartelle sia corretta)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from environment.catcher_discrete import CatchEnv, CatchEnvChangeDirection

# Definizione della rete Q come nel training
import torch.nn as nn
from networks.QNetwork import QNetwork

def state_to_tensor(state, direction):
    """
    Converte lo stato (tuple) in un tensore adatto alla rete neurale.
    
    Se 'direction' è True, lo stato ha 11 elementi:
      (basket_pos, row1, col1, type1, v_speed1, h_speed1, row2, col2, type2, v_speed2, h_speed2)
    Altrimenti, ha 9 elementi:
      (basket_pos, row1, col1, type1, speed1, row2, col2, type2, speed2)
    """
    if direction:
        (basket_pos,
         row1, col1, type1, v_speed1, h_speed1,
         row2, col2, type2, v_speed2, h_speed2) = state
        state_flat = [basket_pos, row1, col1, type1, v_speed1, h_speed1,
                      row2, col2, type2, v_speed2, h_speed2]
    else:
        (basket_pos,
         row1, col1, type1, speed1,
         row2, col2, type2, speed2) = state
        state_flat = [basket_pos, row1, col1, type1, speed1,
                      row2, col2, type2, speed2]
    # Restituisce un tensore di shape [1, input_size]
    return torch.FloatTensor(state_flat).unsqueeze(0)

def test_model(env, model, episodes, direction, device):
    """
    Esegue il test del modello per un certo numero di episodi.
    Per ciascun episodio:
      - Resetta l'ambiente
      - Finché non è raggiunto lo stato terminale, seleziona l'azione greedy (argmax(Q))
      - Accumula la ricompensa totale e conta gli steps
    Restituisce due liste: ricompensa per episodio e steps per episodio.
    """
    rewards = []
    steps_per_episode = []
    model.eval()
    
    for e in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        done = False

        while not done:
            # Converte lo stato in tensore
            state_tensor = state_to_tensor(state, direction).to(device)
            with torch.no_grad():
                q_values = model(state_tensor)
            # Selezione greedy: azione con il valore Q massimo
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
    Visualizza i grafici della ricompensa e del numero di steps per episodio.
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
    
    # Grafico degli steps per episodio
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

    # Determina le dimensioni dell'input in base alla presenza del canale 'direction'
    if direction:
        input_size = 11  # basket_pos + 5 per il primo oggetto + 5 per il secondo
    else:
        input_size = 9   # basket_pos + 4 per il primo oggetto + 4 per il secondo

    hidden_sizes = [128, 128]
    num_actions = env.action_space.n

    # Inizializza la rete e carica il modello salvato
    model = QNetwork(input_size, hidden_sizes, num_actions).to(device)

    # Il training ha salvato il modello finale con questo nome.
    # Se hai usato un nome diverso per l'ambiente con direction, modifica di conseguenza.
    model_path = "models/sarsa_approximated/q_network_episode_9000.pth"
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

    rewards, steps = test_model(env, model, episodes, direction, device)
    env.close()
    plot_results(rewards, steps)

if __name__ == "__main__":
    main()
