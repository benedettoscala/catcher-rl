import pickle
import numpy as np
from environment.catcher_discretized import CatchEnv

# Caricamento del modello SARSA (Q-table)
with open("q_table.pkl", "rb") as f:
    Q = pickle.load(f)

# Funzione per scegliere l'azione basata sulla Q-table (greedy)
def select_action(state, Q):
    state_tuple = tuple(state)
    if state_tuple in Q:
        return np.argmax(Q[state_tuple])
    else:
        return np.random.randint(0, 3)  # Se lo stato non è noto, scegli casualmente

# Inizializzazione dell'ambiente
env = CatchEnv(render_mode="human", grid_size=15)

def play_with_sarsa_model(env, Q, n_episodes=1):
    """
    Fa giocare il modello SARSA nell'ambiente e visualizza il gioco.
    
    :param env: Ambiente CatchEnv
    :param Q: Q-table addestrata
    :param n_episodes: Numero di episodi da giocare
    """
    for episode in range(n_episodes):
        print(f"Inizio episodio {episode + 1}")
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            # Seleziona l'azione basata sulla Q-table
            action = select_action(state, Q)

            # Esegui un passo nell'ambiente
            next_state, reward, done, _, _ = env.step(action)

            # Aggiorna lo stato corrente
            state = next_state

            # Accumula il reward
            total_reward += reward

            # Renderizza l'ambiente
            env.render()

        print(f"Episodio {episode + 1} terminato. Reward totale: {total_reward}")

# Esegui il modello SARSA in modalità visibile
try:
    play_with_sarsa_model(env, Q, n_episodes=5)
except KeyboardInterrupt:
    print("Interrotto manualmente.")
finally:
    env.close()