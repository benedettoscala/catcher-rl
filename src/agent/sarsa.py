import numpy as np
import logging
from collections import deque
from tensorboardX import SummaryWriter
import os
import sys

# Aggiungi il percorso del modulo
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from environment.catcher_discrete import CatchEnvChangeDirection  # Usa il tuo file del tuo env
from environment.catcher_discrete import CatchEnv  # Usa il tuo file del tuo env

class SarsaTrainer:
    def __init__(self,
                 grid_size=15,
                 max_objects_in_state=2,
                 render_mode="none",
                 alpha=0.1,
                 gamma=0.99,
                 epsilon=1.0,  # Inizializza epsilon a 1 per massima esplorazione
                 epsilon_min=0.01,  # Valore minimo di epsilon
                 decay_rate=0.9999,  # Tasso di decadimento per episodio
                 n_episodes=100000,  # Aumentato per consentire il decadimento
                 save_interval=10000,
                 logdir="runs/sarsa_training",
                 rolling_window=100,
                 direction_on=True):
        # Configura il logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
        self.logger = logging.getLogger(__name__)

        # Inizializzazione di TensorBoardX
        self.writer = SummaryWriter(logdir=logdir)

        if direction_on:
            self.env = CatchEnvChangeDirection(
                grid_size=grid_size,
                max_objects_in_state=max_objects_in_state,
                render_mode=render_mode
            )
        else:
            self.env = CatchEnv(
                grid_size=grid_size,
                max_objects_in_state=max_objects_in_state,
                render_mode=render_mode
            )

        # Verifica se l'ambiente ha l'attributo 'direction'
        self.direction = getattr(self.env, 'direction', False)  # Imposta True se direction non è definito
        self.logger.info(f"Ambiente con direzione: {self.direction}")

        # Parametri SARSA
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.decay_rate = decay_rate
        self.n_episodes = n_episodes
        self.save_interval = save_interval

        # Altri parametri
        self.num_actions = self.env.action_space.n
        self.grid_size = self.env.grid_size
        self.offset = self.env.offset
        self.speed_bins = self.env.speed_bins
        if self.direction:
            self.h_speed_bins = self.env.h_speed_bins

        self.direction_size = 2  # 0 o 1

        # Calcoliamo le dimensioni di ogni "asse" dello stato
        self.row_size = self.grid_size - 1 + self.offset + 1  # 15 -1 +10 +1 = 25
        self.col_size = self.grid_size  # 15
        self.type_size = 2  # 0=frutto, 1=bomba
        self.speed_size = self.speed_bins  # 3
        if self.direction:
            self.h_speed_size = self.h_speed_bins  # 3

        # Costruiamo la shape di Q:
        if self.direction:
            self.Q_shape = (
                self.grid_size,
                self.row_size, self.col_size, self.type_size, self.speed_size, self.h_speed_size,
                self.row_size, self.col_size, self.type_size, self.speed_size, self.h_speed_size,
                self.num_actions
            )
        else:
            self.Q_shape = (
                self.grid_size,
                self.row_size, self.col_size, self.type_size, self.speed_size,
                self.row_size, self.col_size, self.type_size, self.speed_size,
                self.num_actions
            )

        self.logger.info(f"Q-table shape = {self.Q_shape}, totale elementi = {np.prod(self.Q_shape):,}")

        # Creiamo l'array Q inizializzato a zero
        self.Q = np.zeros(self.Q_shape, dtype=np.float32)

        # Storico delle ricompense e degli errori TD
        self.rolling_window = rolling_window
        self.reward_history = deque(maxlen=self.rolling_window)
        self.td_error_history = deque(maxlen=self.rolling_window)

    def obs_to_indices(self, obs):
        """
        Converte l'osservazione in una tupla di interi per indicizzare la Q-table.
        """
        if self.direction:
            (basket_pos,
             row1, col1, type1, v_speed1, h_speed1,
             row2, col2, type2, v_speed2, h_speed2) = obs
            return (
                basket_pos,
                row1, col1, type1, v_speed1, h_speed1,
                row2, col2, type2, v_speed2, h_speed2
            )
        else:
            (basket_pos,
             row1, col1, type1, speed1,
             row2, col2, type2, speed2) = obs
            return (
                basket_pos,
                row1, col1, type1, speed1,
                row2, col2, type2, speed2
            )

    def epsilon_greedy(self, state_idx):
        """
        Seleziona un'azione usando una politica epsilon-greedy con decadimento.
        """
        if np.random.rand() < self.epsilon:
            # Esplorazione: scegli un'azione casuale
            action = np.random.randint(self.num_actions)
            self.logger.debug(f"Esplorazione: azione scelta casualmente {action}")
            return action
        else:
            # Sfruttamento: scegli l'azione con il massimo valore Q
            action = np.argmax(self.Q[state_idx])
            self.logger.debug(f"Sfruttamento: azione scelta {action} con Q-valore {self.Q[state_idx + (action,)]}")
            return action

    def update_epsilon(self):
        """
        Aggiorna il valore di epsilon diminuendolo progressivamente fino a epsilon_min.
        """
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.decay_rate
            self.epsilon = max(self.epsilon, self.epsilon_min)
            self.logger.debug(f"Epsilon aggiornato a {self.epsilon}")

    def save_q_table(self, episode):
        """
        Salva la Q-table in un file NumPy.
        """
        save_path = f"q_table_episode_{episode}.npy"
        np.save(save_path, self.Q)
        self.logger.info(f"Q-table salvata in {save_path}")

    def train(self):
        """
        Esegue il ciclo di addestramento SARSA.
        """
        self.logger.info("Inizio training SARSA con array NumPy multidimensionale...")

        for episode in range(1, self.n_episodes + 1):
            obs, _ = self.env.reset()
            try:
                s_idx = self.obs_to_indices(obs)
            except ValueError as e:
                self.logger.error(f"Errore nell'osservazione durante il reset: {e}")
                continue  # Salta all'episodio successivo

            a = self.epsilon_greedy(s_idx)
            done = False
            total_reward = 0

            while not done:
                next_obs, reward, done, _, _ = self.env.step(a)
                try:
                    s_next_idx = self.obs_to_indices(next_obs)
                except ValueError as e:
                    self.logger.error(f"Errore nell'osservazione durante il passo: {e}")
                    s_next_idx = s_idx  # Mantieni lo stato corrente
                    done = True  # Termina l'episodio

                # Scegli la prossima azione con SARSA
                if not done:
                    a_next = self.epsilon_greedy(s_next_idx)
                else:
                    a_next = 0  # Placeholder per il calcolo successivo

                # Aggiorna Q (SARSA)
                try:
                    old_val = self.Q[s_idx + (a,)]
                except IndexError:
                    self.logger.error(f"Indice Q non valido: {s_idx + (a,)}")
                    break  # Termina l'episodio in caso di errore

                if done:
                    td_target = reward
                else:
                    td_target = reward + self.gamma * self.Q[s_next_idx + (a_next,)]
                td_error = td_target - old_val
                self.Q[s_idx + (a,)] += self.alpha * td_error

                # Passo successivo
                s_idx = s_next_idx
                a = a_next
                total_reward += reward

            self.reward_history.append(total_reward)
            self.td_error_history.append(td_error)

            # Aggiorna epsilon dopo ogni episodio
            self.update_epsilon()

            # Logging periodico
            if episode % 1000 == 0:
                avg_reward = np.mean(self.reward_history)
                avg_td_error = np.mean(self.td_error_history)

                self.logger.info(
                    f"Episodio {episode}/{self.n_episodes} - "
                    f"Reward Totale: {total_reward:.2f} - "
                    f"Reward medio: {avg_reward:.2f} - "
                    f"TD Error medio: {avg_td_error:.4f} - "
                    f"Epsilon: {self.epsilon:.4f}"
                )
                self.writer.add_scalar("reward", total_reward, episode)
                self.writer.add_scalar("avg_reward", avg_reward, episode)
                self.writer.add_scalar("avg_td_error", avg_td_error, episode)
                self.writer.add_scalar("epsilon", self.epsilon, episode)


        self.logger.info("Training completato.")

        # Salviamo la Q-table finale in un file NumPy
        if self.direction:
            final_save_path = "q_table_final_changedirection.npy"
        else:
            final_save_path = "q_table_final.npy"

        with open(final_save_path, "wb") as f:
            np.save(f, self.Q)
        self.logger.info(f"Q-table salvata in {final_save_path}")

    def close(self):
        """
        Chiude TensorBoard e l'ambiente.
        """
        self.writer.close()
        self.env.close()

if __name__ == "__main__":
    # Fai scegliere all'utente su che tipo di ambiente addestrare il modello
    choice = input("Scegliere l'ambiente su cui addestrare il modello: 1) CatchEnv 2) CatchEnvChangeDirection: ")
    
    if choice == "1":
        trainer = SarsaTrainer(direction_on=False, logdir="runs/sarsa_training_catchenv")
    elif choice == "2":
        trainer = SarsaTrainer(direction_on=True, logdir="runs/sarsa_training_catchenv_changedirection")
    else:
        print("Scelta non valida.")
        sys.exit(1)

    try:
        trainer.train()
    finally:
        trainer.close()
