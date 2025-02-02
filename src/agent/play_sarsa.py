import pickle
import numpy as np
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from environment.catcher_discrete import CatchEnv
from environment.catcher_discrete import CatchEnvChangeDirection

# Caricamento del modello SARSA (Q-table)


# Funzione per scegliere l'azione basata sulla Q-table (greedy)
def select_action(state, Q):
    """
    Sceglie l'azione basata sulla Q-table.
    
    :param state: Stato corrente
    :param Q: Q-table addestrata
    :return: Azione scelta
    """
    state_idx = tuple(state)
    return np.argmax(Q[state_idx])


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
# Scegli il modello o catch o catch_change_direction
choice = input("Scegliere l'ambiente (catch (1) /catch_change_direction (2) ): ")
if choice == "1":
    env = CatchEnv(render_mode="human", grid_size=15)
    #apri solo il file q_table_final.npy con bfloat 32
    with open("src/sarsa_q_table/q_table_episode_100000.npy", "rb") as f:
        Q = np.load(f, allow_pickle=True)
else:
    env = CatchEnvChangeDirection(render_mode="human", grid_size=10)
    with open("q_table_final_changedirection.npy", "rb") as f:
        Q = np.load(f, allow_pickle=True)
try:
    play_with_sarsa_model(env, Q, n_episodes=5)
except KeyboardInterrupt:
    print("Interrotto manualmente.")
finally:
    env.close()