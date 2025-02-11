import numpy as np
import logging
from collections import deque
from tensorboardX import SummaryWriter
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim

# Aggiungi il percorso del modulo
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from environment.catcher_discrete import CatchEnvChangeDirection
from environment.catcher_discrete import CatchEnv
from networks.QNetwork import QNetwork

class SarsaTrainer:
    def __init__(self,
                 grid_size=11,
                 max_objects_in_state=2,
                 render_mode="none",
                 alpha=0.001,  # Learning rate for optimizer
                 gamma=0.99,
                 epsilon_start=1.0,  # Valore iniziale di epsilon
                 epsilon_end=0.01,    # Valore finale di epsilon
                 epsilon_decay_episodes=9000,  # Numero di episodi su cui decadenza
                 n_episodes=10000,  # Numero totale di episodi
                 save_interval=100,
                 logdir="runs/sarsa_training",
                 rolling_window=100,
                 direction_on=True,
                 hidden_sizes=[128, 128]):
        # Configura il logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        self.epsilon = epsilon_start
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
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_episodes = epsilon_decay_episodes
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
        if self.direction:
            self.state_size = (
                1,  # basket_pos
                6,  # row1, col1, type1, v_speed1, h_speed1, direction1
                6  # row2, col2, type2, v_speed2, h_speed2, direction2
            )
            self.input_size = self._calculate_input_size_with_direction()
        else:
            self.state_size = (
                1,  # basket_pos
                4,  # row1, col1, type1, speed1
                4   # row2, col2, type2, speed2
            )
            self.input_size = self._calculate_input_size()

        self.logger.info(f"Input size for neural network: {self.input_size}")

        # Initialize the neural network
        self.q_network = QNetwork(
            input_size=self.input_size,
            hidden_sizes=hidden_sizes,
            output_size=self.num_actions
        )
        self.q_network.train()  # Set to training mode

        # Define optimizer and loss function
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=alpha)
        self.loss_fn = nn.MSELoss()

        # Storico delle ricompense, TD error e loss
        self.rolling_window = rolling_window
        self.reward_history = deque(maxlen=self.rolling_window)
        self.td_error_history = deque(maxlen=self.rolling_window)
        self.loss_history = deque(maxlen=self.rolling_window)

        # Tracciare l'episodio corrente per la decadenza lineare di epsilon
        self.current_episode = 0

    def _calculate_input_size(self):
        # Define how to flatten the state for the neural network
        # Example: basket_pos + object1 + object2
        # Adjust based on your actual state representation
        return 1 + 4 + 4  # Example: 1 for basket_pos, 4 per ogni oggetto

    def _calculate_input_size_with_direction(self):
        # Define how to flatten the state with direction information
        # Example: basket_pos + object1 + object2 + direction
        return 1 + 6 + 6  # basket_pos + 6 per object1 + 6 per object2 = 13

    def state_to_tensor(self, state):
        """
        Converts the state tuple into a normalized tensor suitable for the neural network.
        """
        if self.direction:
            (basket_pos,
             row1, col1, type1, v_speed1, h_speed1, direction_1,
             row2, col2, type2, v_speed2, h_speed2, direction_2) = state
            state_flat = [
                basket_pos,
                row1, col1, type1, v_speed1, h_speed1, direction_1,
                row2, col2, type2, v_speed2, h_speed2, direction_2
            ]
        else:
            (basket_pos,
             row1, col1, type1, speed1,
             row2, col2, type2, speed2) = state
            state_flat = [
                basket_pos,
                row1, col1, type1, speed1,
                row2, col2, type2, speed2
            ]
        self.logger.debug(f"Stato flat: {state_flat}, lunghezza: {len(state_flat)}")
        state_tensor = torch.FloatTensor(state_flat).unsqueeze(0)  # Shape: [1, input_size]
        return state_tensor

    def epsilon_greedy(self, state_tensor):
        """
        Selects an action using an epsilon-greedy policy based on the Q-network's predictions.
        """
        if np.random.rand() < self.epsilon:
            # Exploration: choose a random action
            action = np.random.randint(self.num_actions)
            self.logger.debug(f"Exploration: chosen random action {action}")
            return action
        else:
            # Exploitation: choose the action with the highest Q-value
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
                action = torch.argmax(q_values).item()
            self.logger.debug(f"Exploitation: chosen action {action} with Q-value {q_values[0, action].item()}")
            return action

    def update_epsilon(self):
        """
        Updates the epsilon value using linear interpolation (lerp) over episodes.
        Epsilon decays from epsilon_start to epsilon_end over epsilon_decay_episodes.
        """
        if self.current_episode < self.epsilon_decay_episodes:
            # Calcola la proporzione dell'episodio attuale rispetto agli episodi di decadimento
            fraction = self.current_episode / self.epsilon_decay_episodes
            # Interpola linearmente tra epsilon_start e epsilon_end
            self.epsilon = self.epsilon_start + fraction * (self.epsilon_end - self.epsilon_start)
        else:
            # Dopo il decadimento, mantieni epsilon a epsilon_end
            self.epsilon = self.epsilon_end
        self.logger.debug(f"Epsilon aggiornato a {self.epsilon:.4f} (episodio {self.current_episode})")

    def save_model(self, episode):
        """
        Saves the neural network's state.
        """
        if self.direction:
            # Create folder if it doesn't exist
            if not os.path.exists("sarsa_approximated_with_direction"):
                os.makedirs("sarsa_approximated_with_direction")

            save_path = f"sarsa_approximated_with_direction/q_network_episode_{episode}.pth"
            torch.save(self.q_network.state_dict(), save_path)
            self.logger.info(f"Q-network saved to {save_path}")
        else:
            #create folder if it doesn't exist
            if not os.path.exists("sarsa_approximated_without_direction"):
                os.makedirs("sarsa_approximated_without_direction")

            save_path = f"sarsa_approximated_without_direction/q_network_episode_{episode}.pth"
            torch.save(self.q_network.state_dict(), save_path)
            self.logger.info(f"Q-network saved to {save_path}")

    def train(self):
        """
        Executes the SARSA training loop using a neural network for Q-value approximation.
        """
        self.logger.info("Inizio training SARSA con rete neurale...")

        for episode in range(1, self.n_episodes + 1):
            self.current_episode = episode  # Aggiorna l'episodio corrente
            obs, _ = self.env.reset()
            s_tensor = self.state_to_tensor(obs)
            a = self.epsilon_greedy(s_tensor)
            done = False
            total_reward = 0

            while not done:
                next_obs, reward, done, _, _ = self.env.step(a)
                next_s_tensor = self.state_to_tensor(next_obs)

                # Select next action using epsilon-greedy
                a_next = self.epsilon_greedy(next_s_tensor) if not done else 0

                # Compute Q(s,a)
                q_values = self.q_network(s_tensor)
                q_sa = q_values[0, a]

                # Compute Q(s',a')
                with torch.no_grad():
                    q_values_next = self.q_network(next_s_tensor)
                    q_s_next_a_next = q_values_next[0, a_next] if not done else 0.0

                # Compute TD target
                td_target = reward + self.gamma * q_s_next_a_next

                # Compute TD error
                td_error = td_target - q_sa.item()

                # Compute loss
                td_target_tensor = torch.tensor([td_target], dtype=torch.float32)
                loss = self.loss_fn(q_sa, td_target_tensor)

                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Update state and action
                s_tensor = next_s_tensor
                a = a_next
                total_reward += reward

            self.reward_history.append(total_reward)
            self.td_error_history.append(td_error)
            self.loss_history.append(loss.item())

            # Update epsilon after each episode using lerp
            self.update_epsilon()

            # Periodic logging
            if episode % 10 == 0:
                avg_reward = np.mean(self.reward_history)
                avg_td_error = np.mean(self.td_error_history)
                avg_loss = np.mean(self.loss_history)

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
                self.writer.add_scalar("avg_loss", avg_loss, episode)

            # Save the model at intervals
            if episode % self.save_interval == 0:
                self.save_model(episode)

        self.logger.info("Training completato.")

        # Save the final model
        final_save_path = "q_network_final_without_direction.pth"
        torch.save(self.q_network.state_dict(), final_save_path)
        self.logger.info(f"Q-network finale salvata in {final_save_path}")

    def close(self):
        """
        Closes TensorBoard and the environment.
        """
        self.writer.close()
        self.env.close()

if __name__ == "__main__":
    # Fai scegliere all'utente su che tipo di ambiente addestrare il modello
    choice = input("Scegliere l'ambiente su cui addestrare il modello: 1) CatchEnv 2) CatchEnvChangeDirection: ")
    
    if choice == "1":
        trainer = SarsaTrainer(direction_on=False, logdir="runs/sarsa_training_catchenv_nn", grid_size=15)
    elif choice == "2":
        trainer = SarsaTrainer(direction_on=True, logdir="runs/sarsa_training_catchenv_changedirection_nn", grid_size=15)
    else:
        print("Scelta non valida.")
        sys.exit(1)

    try:
        trainer.train()
    finally:
        trainer.close()
