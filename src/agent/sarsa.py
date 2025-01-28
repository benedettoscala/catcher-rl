import numpy as np
import logging
import pickle
from collections import defaultdict
import os, sys
from tensorboardX import SummaryWriter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from environment.catcher_discretized import CatchEnv

# Configurazione del logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Inizializzazione di TensorBoardX
writer = SummaryWriter(logdir="runs/sarsa_training")

# Inizializzazione dell'ambiente
env = CatchEnv(grid_size=15)

# Parametri SARSA
Q = defaultdict(lambda: np.zeros(env.action_space.n))
alpha = 0.1
gamma = 0.99
epsilon = 0.1
n_episodes = 300000

# Finestra di rolling per il calcolo della media dei reward
rolling_window = 100
reward_history = []
td_errors = []  # Per tracciare l'errore TD

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

        # Calcolo e salvataggio dell'errore TD
        old_value = Q[tuple(state)][action]
        td_target = reward + gamma * Q[tuple(next_state)][next_action] * (0 if done else 1)
        td_error = td_target - old_value
        td_errors.append(td_error)

        # Aggiorno la Q-table (SARSA)
        Q[tuple(state)][action] = old_value + alpha * td_error

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

    # Logging su console e TensorBoardX
    if episode % 100 == 0:
        avg_td_error = np.mean(td_errors[-rolling_window:]) if len(td_errors) >= rolling_window else np.mean(td_errors)
        logging.info(f"Episodio {episode + 1}/{n_episodes}: Reward totale = {total_reward}, Rolling avg (ultimi {rolling_window}) = {rolling_avg:.2f}, TD Error Medio = {avg_td_error:.4f}")
        writer.add_scalar("Reward_Totale", total_reward, episode)
        writer.add_scalar("Media_Rolling", rolling_avg, episode)
        writer.add_scalar("TD_Error_Medio", avg_td_error, episode)

logging.info("Training completato")

# Salvataggio del modello (Q-table)
with open("q_table.pkl", "wb") as f:
    pickle.dump(dict(Q), f)
    logging.info("Modello salvato su q_table.pkl")

# Valutazione Post-Training
logging.info("Inizio della valutazione post-training")
test_episodes = 100
total_test_rewards = []

for _ in range(test_episodes):
    state, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = np.argmax(Q[tuple(state)])  # Politica greedy senza esplorazione
        next_state, reward, done, _, _ = env.step(action)
        total_reward += reward
        state = next_state

    total_test_rewards.append(total_reward)

avg_test_reward = np.mean(total_test_rewards)
logging.info(f"Reward medio su {test_episodes} episodi di test: {avg_test_reward:.2f}")
writer.add_scalar("Reward_Medio_Post_Training", avg_test_reward)

# Salvataggio dei risultati di test
with open("test_rewards.pkl", "wb") as f:
    pickle.dump(total_test_rewards, f)
    logging.info("Risultati dei test salvati su test_rewards.pkl")

# Chiusura di TensorBoardX
writer.close()

env.close()