#!/usr/bin/env python3
import os
import sys
import torch

# Aggiungi il path al modulo con gli ambienti
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from environment.catcher_discrete import CatchEnv, CatchEnvChangeDirection

# Import della rete
import torch.nn as nn
from networks.QNetwork import QNetwork

def state_to_tensor(state, direction):
    """
    Converte lo stato in un tensore PyTorch utilizzabile dalla rete Q.
    """
    state_list = list(state)  # Lo stato è una tupla di dimensione variabile
    return torch.FloatTensor(state_list).unsqueeze(0)

def play_agent(env, model, episodes, direction, device):
    """
    Fa giocare l'agente per un certo numero di episodi, mostrando l'ambiente.
    Non raccoglie statistiche, ma stampa solo la ricompensa finale di ogni episodio.
    """

    model.eval()  # Imposta il modello in modalità di valutazione

    for e in range(1, episodes + 1):
        state, _ = env.reset()
        done = False
        total_reward = 0

        print(f"\n--- Episodio {e} ---")
        
        while not done:
            # Mostra l'ambiente
            env.render()

            # Conversione dello stato in tensore
            state_tensor = state_to_tensor(state, direction).to(device)

            # Calcolo dei Q-values e scelta dell'azione greedy
            with torch.no_grad():
                q_values = model(state_tensor)
            action = torch.argmax(q_values, dim=1).item()

            # Esegui l'azione nell'ambiente
            next_state, reward, done, _, info = env.step(action)
            total_reward += reward

            # Aggiorna lo stato
            state = next_state

        # Stampa la ricompensa totale ottenuta in questo episodio
        print(f"Ricompensa totale nell'episodio {e}: {total_reward}")

    env.close()

def main():
    print("Scegliere l'ambiente di gioco:")
    print("1) CatchEnv (movimento standard)")
    print("2) CatchEnvChangeDirection (movimento con cambio di direzione)")
    choice = input("Inserisci la tua scelta (1 o 2): ")

    # In base alla scelta, carichiamo l'ambiente e settiamo la dimensione dell'input
    if choice == "1":
        env = CatchEnv(grid_size=15)
        direction = False
        model_path = "models/best_sarsa_approximated/no_direction/q_network_final.pth"
        input_size = 9   # dimensione dello stato in CatchEnv
    elif choice == "2":
        env = CatchEnvChangeDirection(grid_size=15)
        direction = True
        model_path = "models/best_sarsa_approximated/direction/q_network_final.pth"
        input_size = 11  # dimensione dello stato in CatchEnvChangeDirection
    else:
        print("Scelta non valida. Uscita.")
        return

    # Controlliamo se il modello esiste
    if not os.path.exists(model_path):
        print(f"Modello non trovato in {model_path}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilizzo device: {device}")

    # Definizione della rete Q (con la stessa struttura usata in addestramento)
    hidden_sizes = [128, 128]
    num_actions = env.action_space.n
    model = QNetwork(input_size, hidden_sizes, num_actions).to(device)

    # Carichiamo i pesi del modello
    model.load_state_dict(torch.load(model_path, map_location=device))

    # Numero di episodi di default
    try:
        episodes = int(input("Inserisci il numero di episodi da giocare (default 5): "))
    except ValueError:
        print("Input non valido. Utilizzo del valore di default: 5 episodi.")
        episodes = 5

    # Avvia il gameplay
    play_agent(env, model, episodes, direction, device)

if __name__ == "__main__":
    main()
