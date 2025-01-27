import numpy as np
import logging
import pickle
from collections import defaultdict
from environment.catcher_discretized import CatchEnv

# Configurazione del logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Inizializzazione dell'ambiente
env = CatchEnv(grid_size=20)

# Parametri SARSA
Q = defaultdict(lambda: np.zeros(env.action_space.n))
alpha = 0.1
gamma = 0.99
epsilon = 0.1
n_episodes = 100000

# Finestra di rolling per il calcolo della media dei reward
rolling_window = 100
reward_history = []

def epsilon_greedy(state, Q, epsilon):
    if np.random.rand() < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(Q[tuple(state)])

logging.info("Inizio del training SARSA")

for episode in range(n_episodes):
    state, _ = env.reset()
    done = False
    action = epsilon_greedy(state, Q, epsilon)
    total_reward = 0  # Per tracciare il reward totale dell'episodio

    while not done:
        next_state, reward, done, _, _ = env.step(action)
        next_action = epsilon_greedy(next_state, Q, epsilon)

        # Aggiorno la Q-table (SARSA)
        old_value = Q[tuple(state)][action]
        td_target = reward + gamma * Q[tuple(next_state)][next_action] * (0 if done else 1)
        Q[tuple(state)][action] = old_value + alpha * (td_target - old_value)

        # Accumulo del reward totale
        total_reward += reward

        # Aggiornamento dello stato e dell'azione
        state = next_state
        action = next_action

    # Aggiungo il reward totale alla storia
    reward_history.append(total_reward)

    # Calcolo della media rolling
    if len(reward_history) > rolling_window:
        reward_history = reward_history[-rolling_window:]
    rolling_avg = np.mean(reward_history)

    if episode % 20 == 0:
    # Logging del reward totale per episodio e della media rolling
        logging.info(f"Episodio {episode + 1}/{n_episodes}: Reward totale = {total_reward}, Rolling avg (ultimi {rolling_window}) = {rolling_avg:.2f}")

logging.info("Training completato")

# Salvataggio del modello (Q-table)
with open("q_table.pkl", "wb") as f:
    pickle.dump(dict(Q), f)
    logging.info("Modello salvato su q_table.pkl")

env.close()