import numpy as np
import logging
import pickle
from collections import defaultdict
# Adjust the paths if needed
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from environment.catcher_discretized import CatchEnv

# Configurazione del logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Inizializzazione dell'ambiente
env = CatchEnv(grid_size=15)

# Parametri SARSA(\u03bb)
Q = defaultdict(lambda: np.zeros(env.action_space.n))  # Q-table
E = defaultdict(lambda: np.zeros(env.action_space.n))  # Tracce di eleggibilità
alpha = 0.1  # Learning rate
gamma = 0.99  # Fattore di sconto
epsilon = 0.1  # Epsilon-greedy per l'esplorazione
lmbda = 0.9  # Parametro di decadimento delle tracce di eleggibilità
n_episodes = 1000

# Finestra di rolling per il calcolo della media dei reward
rolling_window = 100
reward_history = []

def epsilon_greedy(state, Q, epsilon):
    if np.random.rand() < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(Q[tuple(state)])

logging.info("Inizio del training SARSA(\u03bb)")

for episode in range(n_episodes):
    state, _ = env.reset()
    done = False
    action = epsilon_greedy(state, Q, epsilon)
    total_reward = 0  # Per tracciare il reward totale dell'episodio

    # Reset delle tracce di eleggibilità
    for key in E:
        E[key].fill(0)

    while not done:
        next_state, reward, done, _, _ = env.step(action)
        next_action = epsilon_greedy(next_state, Q, epsilon)

        # Calcolo dell'errore TD
        td_error = reward + gamma * Q[tuple(next_state)][next_action] * (0 if done else 1) - Q[tuple(state)][action]

        # Aggiorno la traccia di eleggibilità per lo stato-attuale
        E[tuple(state)][action] += 1

        # Aggiorno Q e decadimento delle tracce
        for s in E:
            Q[s] += alpha * td_error * E[s]
            E[s] *= gamma * lmbda  # Decadimento delle tracce

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

    if episode % 1 == 0:
        # Logging del reward totale per episodio e della media rolling
        logging.info(f"Episodio {episode + 1}/{n_episodes}: Reward totale = {total_reward}, Rolling avg (ultimi {rolling_window}) = {rolling_avg:.2f}")

logging.info("Training completato")

# Salvataggio del modello (Q-table)
with open("q_table_sarsa_lambda.pkl", "wb") as f:
    pickle.dump(dict(Q), f)
    logging.info("Modello salvato su q_table_sarsa_lambda.pkl")

env.close()
