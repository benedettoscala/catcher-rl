#!/usr/bin/env python3
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Aggiungi il percorso del modulo (assicurati che la struttura delle cartelle sia corretta)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from environment.catcher_discrete import CatchEnv, CatchEnvChangeDirection

def obs_to_indices(obs, direction):
    """
    Converte l'osservazione in una tupla di interi per indicizzare la Q-table.
    
    Se direction è True, l'osservazione è composta da 11 elementi:
        (basket_pos, row1, col1, type1, v_speed1, h_speed1, row2, col2, type2, v_speed2, h_speed2)
    Altrimenti, l'osservazione ha 9 elementi:
        (basket_pos, row1, col1, type1, speed1, row2, col2, type2, speed2)
    """
    if direction:
        try:
            (basket_pos,
             row1, col1, type1, v_speed1, h_speed1,
             row2, col2, type2, v_speed2, h_speed2) = obs
        except Exception as e:
            print(f"Errore nell'unpacking dell'osservazione con direction: {e}")
            raise
        return (int(basket_pos),
                int(row1), int(col1), int(type1), int(v_speed1), int(h_speed1),
                int(row2), int(col2), int(type2), int(v_speed2), int(h_speed2))
    else:
        try:
            (basket_pos,
             row1, col1, type1, speed1,
             row2, col2, type2, speed2) = obs
        except Exception as e:
            print(f"Errore nell'unpacking dell'osservazione senza direction: {e}")
            raise
        return (int(basket_pos),
                int(row1), int(col1), int(type1), int(speed1),
                int(row2), int(col2), int(type2), int(speed2))

def test_model(env, Q_table, direction, episodes=50):
    """
    Testa la politica greedy basata sulla Q-table per un certo numero di episodi.
    Per ciascun episodio:
      - Resetta l'ambiente
      - Finché non viene raggiunto uno stato terminale, seleziona l'azione con il massimo Q-valore
        (ossia, argmax(Q[state_indici]) )
      - Accumula la ricompensa totale e conta il numero di passi
    Restituisce due liste: rewards (ricompensa per episodio) e steps (numero di passi per episodio).
    """
    rewards = []
    steps_per_episode = []
    success_rates = []  # Tasso di successo (oggetti catturati / totali)
    lives_remaining = []  # Vite rimanenti alla fine di ogni episodio
    caught_objects = []  # Numero di oggetti presi
    missed_objects = []  # Numero di oggetti mancati
    malicious_catches = []  # Numero di bombe catturate
        
    for e in range(episodes):
        obs, _ = env.reset()
        total_reward = 0
        steps = 0
        done = False
        
        while not done:
            # Converti l'osservazione in indici per la Q-table
            state_idx = obs_to_indices(obs, direction)
            # Seleziona l'azione greedy: quella con il valore massimo
            q_values = Q_table[state_idx]
            action = np.argmax(q_values)
            
            # Esegui l'azione e aggiorna lo stato
            obs, reward, done, _, _ = env.step(action)
            total_reward += reward
            steps += 1
        
        rewards.append(total_reward)
        steps_per_episode.append(steps)
        success_rates.append(env.caught_objects / max(1, (env.caught_objects + env.missed_objects)))
        lives_remaining.append(env.lives)
        caught_objects.append(env.caught_objects)
        missed_objects.append(env.missed_objects)
        malicious_catches.append(env.malicious_catches)
        print(f"Episodio {e+1}/{episodes} - Ricompensa: {total_reward}, Steps: {steps}")
    
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
    choice = "1"
    
    if choice == "1":
        env = CatchEnv(grid_size=15, max_objects_in_state=2, render_mode="none")
        direction = False
        q_table_file = "models/best_sarsa_q_table_model/q_table_final.npy"
    elif choice == "2":
        env = CatchEnvChangeDirection(grid_size=15, max_objects_in_state=2, render_mode="none")
        direction = True
        q_table_file = "q_table_final_changedirection.npy"
    else:
        print("Scelta non valida. Uscita.")
        return
    
    if not os.path.exists(q_table_file):
        print(f"File Q-table non trovato: {q_table_file}")
        return
    
    # Carica la Q-table salvata
    Q_table = np.load(q_table_file)
    print(f"Q-table caricata da {q_table_file}. Forma: {Q_table.shape}")
    
    try:
        episodes = int(input("Inserisci il numero di episodi di test: "))
    except ValueError:
        print("Input non valido. Verranno eseguiti 50 episodi.")
        episodes = 50
    
    rewards, steps, success_rates, lives_remaining, caught_objects, missed_objects, malicious_catches = test_model(env, Q_table, direction, episodes)
    env.close()
    plot_results(rewards, steps, success_rates, lives_remaining, caught_objects, missed_objects, malicious_catches)

if __name__ == "__main__":
    main()
