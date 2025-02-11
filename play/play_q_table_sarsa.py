#!/usr/bin/env python3
import os
import sys
import numpy as np

# Aggiungi il percorso del modulo con gli ambienti (modifica se necessario)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from environment.catcher_discrete import CatchEnv, CatchEnvChangeDirection

def obs_to_indices(obs, direction):
    """
    Converte l'osservazione in una tupla di indici per la Q-table.
    """
    if direction:
        # Osservazione con cambio direzione: 11 elementi
        # (basket_pos, row1, col1, type1, v_speed1, h_speed1,
        #  row2, col2, type2, v_speed2, h_speed2)
        (basket_pos,
         row1, col1, type1, v_speed1, h_speed1,
         row2, col2, type2, v_speed2, h_speed2) = obs

        return (int(basket_pos),
                int(row1), int(col1), int(type1), int(v_speed1), int(h_speed1),
                int(row2), int(col2), int(type2), int(v_speed2), int(h_speed2))
    else:
        # Osservazione classica: 9 elementi
        # (basket_pos, row1, col1, type1, speed1,
        #  row2, col2, type2, speed2)
        (basket_pos,
         row1, col1, type1, speed1,
         row2, col2, type2, speed2) = obs

        return (int(basket_pos),
                int(row1), int(col1), int(type1), int(speed1),
                int(row2), int(col2), int(type2), int(speed2))

def play_agent(env, Q_table, direction, episodes=5):
    """
    Fa giocare l’agente per un certo numero di episodi, mostrando l'ambiente.
    Non vengono raccolte statistiche, si mostra solo la ricompensa totale alla fine di ciascun episodio.
    """

    for e in range(1, episodes + 1):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        
        print(f"\n--- Episodio {e} ---")
        
        while not done:
            # Renderizza l'ambiente
            env.render()

            # Converte l'osservazione in indici per la Q-table
            state_idx = obs_to_indices(obs, direction)
            # Seleziona l’azione con Q-value massimo
            q_values = Q_table[state_idx]
            action = np.argmax(q_values)

            # Esegui l'azione
            obs, reward, done, _, _ = env.step(action)
            total_reward += reward

        print(f"Ricompensa totale nell'episodio {e}: {total_reward}")
    
    env.close()

def main():
    print("Scegliere l'ambiente di gioco:")
    print("1) CatchEnv (standard, no direction)")
    print("2) CatchEnvChangeDirection (con cambio di direzione)")
    choice = input("Inserisci la tua scelta (1 o 2): ")

    # Carica l'ambiente e la Q-table in base alla scelta
    if choice == "1":
        env = CatchEnv(grid_size=15, max_objects_in_state=2, render_mode="human")
        q_table_file = "models/best_sarsa_q_table_model/q_table_final.npy"
        direction = False
    elif choice == "2":
        env = CatchEnvChangeDirection(grid_size=15, max_objects_in_state=2, render_mode="human")
        q_table_file = "models/best_sarsa_q_table_model/q_table_final_changedirection.npy"
        direction = True
    else:
        print("Scelta non valida. Uscita.")
        return

    if not os.path.exists(q_table_file):
        print(f"File Q-table non trovato: {q_table_file}")
        return

    # Carica la Q-table
    Q_table = np.load(q_table_file)
    print(f"Q-table caricata da {q_table_file}. Forma: {Q_table.shape}")

    # Chiedi il numero di episodi da giocare
    try:
        episodes = int(input("Inserisci il numero di episodi da giocare (default 5): "))
    except ValueError:
        episodes = 5

    # Avvia il gameplay
    play_agent(env, Q_table, direction, episodes)

if __name__ == "__main__":
    main()
