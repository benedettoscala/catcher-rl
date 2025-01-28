import numpy as np
import logging
from collections import deque
import pickle
from tensorboardX import SummaryWriter
from environment.catcher_discretized import CatchEnv  # Usa il tuo file del tuo env

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Inizializzazione di TensorBoardX
writer = SummaryWriter(logdir="runs/sarsa_training")

# Creiamo l'ambiente
env = CatchEnv(
    grid_size=15,
      # 3 bin di velocità
    render_mode="none" # o "human" per vedere la grafica (più lento)
)

# Parametri SARSA
alpha = 0.1
gamma = 0.99
epsilon = 0.1
n_episodes = 500000   # Attenzione ai tempi e memoria

num_actions = env.action_space.n  # 3
grid_size = env.grid_size         # 10
offset = env.offset               # 10 -> row va [0..19]
speed_bins = env.speed_bins       # 3

# Calcoliamo le dimensioni di ogni "asse" dello stato
row_size = grid_size - 1 + offset + 1  # 20
col_size = grid_size                  # 10
type_size = 2                         # 0=frutto,1=bomba
speed_size = speed_bins               # 3

# Costruiamo la shape di Q:
#  [ basket, row1, col1, type1, speed1, row2, col2, type2, speed2, action ]
Q_shape = (
    grid_size,
    row_size, col_size, type_size, speed_size,
    row_size, col_size, type_size, speed_size,
    num_actions
)
logging.info(f"Q-table shape = {Q_shape}, totale elementi = {np.prod(Q_shape):,}")

# Creiamo l'array Q
Q = np.random.rand(*Q_shape)

def obs_to_indices(obs):
    """
    Converte l'osservazione in una tupla di 8 interi (senza l'azione).
    Formato: obs = [basket, row1, col1, type1, speed1, row2, col2, type2, speed2]
    """
    (basket_pos,
     row1, col1, type1, speed1,
     row2, col2, type2, speed2) = obs

    return (basket_pos,
            row1, col1, type1, speed1,
            row2, col2, type2, speed2)

def epsilon_greedy(state_idx, Q, epsilon):
    """
    state_idx è una tupla (basket, row1, col1, type1, speed1, row2, col2, type2, speed2).
    Restituisce un'azione in [0..num_actions-1] in modo epsilon-greedy.
    """
    if np.random.rand() < epsilon:
        return np.random.randint(num_actions)
    else:
        # Q[state_idx] è un vettore di dimensione num_actions
        return np.argmax(Q[state_idx])

rolling_window = 100
reward_history = deque(maxlen=rolling_window)
td_error_history = deque(maxlen=rolling_window)

logging.info("Inizio training SARSA con array NumPy multidimensionale...")

for episode in range(n_episodes):
    obs, _ = env.reset()
    s_idx = obs_to_indices(obs)
    a = epsilon_greedy(s_idx, Q, epsilon)
    done = False

    total_reward = 0

    while not done:
        next_obs, reward, done, _, _ = env.step(a)
        s_next_idx = obs_to_indices(next_obs)

        # Scegli la prossima azione col SARSA
        if not done:
            a_next = epsilon_greedy(s_next_idx, Q, epsilon)
        else:
            a_next = 0  # placeholder per il calcolo successivo

        # Aggiorna Q (SARSA)
        old_val = Q[s_idx + (a,)]
        if done:
            td_target = reward
        else:
            td_target = reward + gamma * Q[s_next_idx + (a_next,)]
        td_error = td_target - old_val
        Q[s_idx + (a,)] = old_val + alpha * (td_error)
        
        # Passo successivo
        s_idx = s_next_idx
        a = a_next
        total_reward += reward

    reward_history.append(total_reward)
    td_error_history.append(td_error)

    if (episode+1) % 1000 == 0:
        avg_reward = np.mean(reward_history)
        avg_td_error = np.mean(td_error_history)

        logging.info(f"Episodio {episode+1}/{n_episodes} - Reward Totale: {total_reward:.2f} - Reward medio: {avg_reward:.2f} - TD Error medio: {avg_td_error}")
        writer.add_scalar("reward", total_reward, episode)
        writer.add_scalar("avg_reward", avg_reward, episode)
        writer.add_scalar("avg_td_error", avg_td_error, episode)

logging.info("Training completato.")

# Salviamo la Q-table in un file NumPy
with open("q_table_multi.npy", "wb") as f:
    np.save(f, Q)
logging.info("Q-table salvata in q_table_multi.npy")

writer.close()
env.close()
