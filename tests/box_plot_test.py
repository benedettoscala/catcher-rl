#!/usr/bin/env python3
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

# Aggiungi il path dei moduli (assicurati che la struttura delle cartelle sia corretta)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

##############################################
# 1) TEST DEL MODELLO CNN (immagini)         #
##############################################
def test_cnn_model(episodes=50):
    """
    Testa il modello CNN (basato su immagini) utilizzando l'ambiente CatchEnvImage.
    Restituisce un dizionario con le metriche raccolte.
    """
    print("\n----- Test Modello CNN (CatchEnvImage) -----")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilizzo device: {device}")

    # Importa ambiente e rete CNN
    try:
        from environment.catcher_image import CatchEnvImage  # ambiente per testare il modello CNN
        from networks.QNetwork import CNNQNetwork
    except ImportError as e:
        print("Errore nell'import dei moduli per il modello CNN:", e)
        return None

    # Crea l'ambiente (versione senza cambio di direzione)
    env = CatchEnvImage(grid_size=15)
    # Costruisci il path del modello
    model_path = os.path.join("models", "dqn_cnn_models", "no_direction", "dqn_model_final.pth")
    if not os.path.exists(model_path):
        print(f"File del modello CNN non trovato in: {model_path}")
        return None

    # Inizializza e carica il modello CNN
    model = CNNQNetwork(grid_size=15, action_size=env.action_space.n, in_channels=env.in_channels).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Dizionario per salvare le metriche
    metrics = {
        "rewards": [],
        "steps": [],
        "success_rates": [],
        "lives_remaining": [],
        "caught_objects": [],
        "missed_objects": [],
        "malicious_catches": []
    }

    for e in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        done = False

        while not done:
            # Se vuoi abilitare il rendering, decommenta la riga seguente
            # env.render()

            # Imposta un framerate alto per accelerare il test
            env.metadata["render_fps"] = 10000

            # Prepara lo stato per la rete e calcola i Q-values
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)  # shape [1, 2, grid_size, grid_size]
            with torch.no_grad():
                q_values = model(state_tensor)
            action = torch.argmax(q_values, dim=1).item()

            next_state, reward, done, _, _ = env.step(action)
            total_reward += reward
            steps += 1
            state = next_state

        # Salva le metriche per l'episodio
        metrics["rewards"].append(total_reward)
        metrics["steps"].append(steps)
        success_rate = env.caught_objects / max(1, (env.caught_objects + env.missed_objects))
        metrics["success_rates"].append(success_rate)
        metrics["lives_remaining"].append(env.lives)
        metrics["caught_objects"].append(env.caught_objects)
        metrics["missed_objects"].append(env.missed_objects)
        metrics["malicious_catches"].append(env.malicious_catches)
        print(f"[CNN] Episodio {e+1}/{episodes} - Reward: {total_reward}, Steps: {steps}")

    env.close()
    return metrics

##############################################
# 2) TEST DEL MODELLO QNETWORK (discreto)     #
##############################################
def test_qnetwork_model(episodes=50):
    """
    Testa il modello QNetwork (rete fully-connected per ambiente discreto) utilizzando CatchEnv.
    Restituisce un dizionario con le metriche raccolte.
    """
    print("\n----- Test Modello QNetwork (CatchEnv) -----")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilizzo device: {device}")

    try:
        from environment.catcher_discrete import CatchEnv
        from networks.QNetwork import QNetwork
    except ImportError as e:
        print("Errore nell'import dei moduli per il modello QNetwork:", e)
        return None

    # Crea l'ambiente (versione senza cambio di direzione)
    env = CatchEnv(grid_size=15)
    direction = False  # non utilizziamo informazioni di direzione
    input_size = 9   # osservazione senza direzione
    hidden_sizes = [128, 128]
    num_actions = env.action_space.n
    model_path = os.path.join("models", "best_sarsa_approximated", "no_direction", "q_network_final.pth")
    if not os.path.exists(model_path):
        print(f"File del modello QNetwork non trovato in: {model_path}")
        return None

    # Inizializza e carica il modello QNetwork
    model = QNetwork(input_size, hidden_sizes, num_actions).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Funzione ausiliaria per convertire lo stato in tensore
    def state_to_tensor(state):
        state_flat = list(state)
        return torch.FloatTensor(state_flat).unsqueeze(0)

    metrics = {
        "rewards": [],
        "steps": [],
        "success_rates": [],
        "lives_remaining": [],
        "caught_objects": [],
        "missed_objects": [],
        "malicious_catches": []
    }

    for e in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        done = False

        while not done:
            state_tensor = state_to_tensor(state).to(device)
            with torch.no_grad():
                q_values = model(state_tensor)
            action = torch.argmax(q_values, dim=1).item()

            next_state, reward, done, _, _ = env.step(action)
            total_reward += reward
            steps += 1
            state = next_state

        metrics["rewards"].append(total_reward)
        metrics["steps"].append(steps)
        success_rate = env.caught_objects / max(1, (env.caught_objects + env.missed_objects))
        metrics["success_rates"].append(success_rate)
        metrics["lives_remaining"].append(env.lives)
        metrics["caught_objects"].append(env.caught_objects)
        metrics["missed_objects"].append(env.missed_objects)
        metrics["malicious_catches"].append(env.malicious_catches)
        print(f"[QNetwork] Episodio {e+1}/{episodes} - Reward: {total_reward}, Steps: {steps}")

    env.close()
    return metrics

##############################################
# 3) TEST DEL MODELLO Q-TABLE (discreto)      #
##############################################
def test_qtable_model(episodes=50):
    """
    Testa il modello basato su Q-table utilizzando CatchEnv.
    Restituisce un dizionario con le metriche raccolte.
    """
    print("\n----- Test Modello Q-Table (CatchEnv) -----")
    try:
        from environment.catcher_discrete import CatchEnv
    except ImportError as e:
        print("Errore nell'import dei moduli per il modello Q-Table:", e)
        return None

    # Crea l'ambiente (versione senza direzione)
    env = CatchEnv(grid_size=15, max_objects_in_state=2, render_mode="none")
    direction = False  # non utilizziamo la componente 'direction' in questo caso
    q_table_file = os.path.join("models", "best_sarsa_q_table_model", "q_table_final.npy")
    if not os.path.exists(q_table_file):
        print(f"File Q-table non trovato: {q_table_file}")
        return None

    # Carica la Q-table salvata
    Q_table = np.load(q_table_file)
    print(f"Q-table caricata da {q_table_file}. Forma: {Q_table.shape}")

    # Funzione ausiliaria per convertire l'osservazione in indici (versione senza direction)
    def obs_to_indices(obs):
        try:
            (basket_pos, row1, col1, type1, speed1, row2, col2, type2, speed2) = obs
        except Exception as e:
            print(f"Errore nell'unpacking dell'osservazione: {e}")
            raise
        return (int(basket_pos), int(row1), int(col1), int(type1), int(speed1),
                int(row2), int(col2), int(type2), int(speed2))

    metrics = {
        "rewards": [],
        "steps": [],
        "success_rates": [],
        "lives_remaining": [],
        "caught_objects": [],
        "missed_objects": [],
        "malicious_catches": []
    }

    for e in range(episodes):
        obs, _ = env.reset()
        total_reward = 0
        steps = 0
        done = False

        while not done:
            # Converte l'osservazione in indici per la Q-table
            state_idx = obs_to_indices(obs)
            q_values = Q_table[state_idx]
            action = np.argmax(q_values)

            obs, reward, done, _, _ = env.step(action)
            total_reward += reward
            steps += 1

        metrics["rewards"].append(total_reward)
        metrics["steps"].append(steps)
        success_rate = env.caught_objects / max(1, (env.caught_objects + env.missed_objects))
        metrics["success_rates"].append(success_rate)
        metrics["lives_remaining"].append(env.lives)
        metrics["caught_objects"].append(env.caught_objects)
        metrics["missed_objects"].append(env.missed_objects)
        metrics["malicious_catches"].append(env.malicious_catches)
        print(f"[QTable] Episodio {e+1}/{episodes} - Reward: {total_reward}, Steps: {steps}")

    env.close()
    return metrics

##############################################
# FUNZIONE PER LA VISUALIZZAZIONE DEI RISULTATI#
##############################################
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
import threading

def plot_combined_metrics(metrics_cnn, metrics_qnet, metrics_qtable):
    """
    Crea box plot separati per ogni metrica, mettendo a confronto
    i risultati dei 3 modelli in finestre diverse.
    """
    # Prepara i dati: per ogni metrica, metti in una lista i risultati dei tre modelli
    combined = {
        "Ricompensa per Episodio": [metrics_cnn["rewards"], metrics_qnet["rewards"], metrics_qtable["rewards"]],
        "Steps per Episodio": [metrics_cnn["steps"], metrics_qnet["steps"], metrics_qtable["steps"]],
        "Tasso di Successo (%)": [
            [x * 100 for x in metrics_cnn["success_rates"]],
            [x * 100 for x in metrics_qnet["success_rates"]],
            [x * 100 for x in metrics_qtable["success_rates"]]
        ],
        "Vite Rimanenti": [metrics_cnn["lives_remaining"], metrics_qnet["lives_remaining"], metrics_qtable["lives_remaining"]],
        "Oggetti Presi": [metrics_cnn["caught_objects"], metrics_qnet["caught_objects"], metrics_qtable["caught_objects"]],
        "Oggetti Mancati": [metrics_cnn["missed_objects"], metrics_qnet["missed_objects"], metrics_qtable["missed_objects"]],
        "Malicious Catches": [metrics_cnn["malicious_catches"], metrics_qnet["malicious_catches"], metrics_qtable["malicious_catches"]]
    }
    
    # Nomi e colori dei modelli
    model_names = ["DQN con CNN", "SARSA con rete neurale", "SARSA Q-Table"]
    colors = ['skyblue', 'lightgreen', 'salmon']
    
    for metric_name, data_list in combined.items():
        plt.figure(figsize=(6, 5))  # Crea una nuova finestra per ogni metrica
        
        # Crea il boxplot per i 3 modelli
        bp = plt.boxplot(data_list, patch_artist=True)
        
        # Assegna un colore diverso a ciascuna scatola
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        plt.title(metric_name)
        plt.xticks([1, 2, 3], model_names)
        plt.grid(True)
        
        # Crea una legenda personalizzata
        import matplotlib.patches as mpatches
        legend_handles = [mpatches.Patch(color=color, label=name) for color, name in zip(colors, model_names)]
        plt.legend(handles=legend_handles, loc='upper right')
        
        plt.show(block=False)  # Mostra i grafici senza bloccare
    
    plt.show()  # Mantiene aperte tutte le finestre
    


##############################################
# FUNZIONE MAIN                              #
##############################################
def main():
    try:
        episodes = int(input("Inserisci il numero di episodi di test per ciascun modello (default 50): "))
    except ValueError:
        episodes = 50
        print("Input non valido. Verranno eseguiti 50 episodi per modello.")

    # Esegui i test per ciascun modello
    metrics_cnn   = test_cnn_model(episodes)
    metrics_qnet  = test_qnetwork_model(episodes)
    metrics_qtable = test_qtable_model(episodes)

    if metrics_cnn is None or metrics_qnet is None or metrics_qtable is None:
        print("Errore nel caricamento di uno o più modelli. Impossibile continuare.")
        return

    # Visualizza i risultati combinati
    plot_combined_metrics(metrics_cnn, metrics_qnet, metrics_qtable)

if __name__ == "__main__":
    main()
