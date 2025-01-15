import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame

import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import matplotlib.pyplot as plt
import os
import sys

# Per usare TensorBoardX (opzionale, se lo desideri installare)
from tensorboardX import SummaryWriter

###############################################################################
#                            ENVIRONMENT: CatchEnv
###############################################################################

LEFT = -1
STAY = 0
RIGHT = 1
TIME_LIMIT = 250

class CatchEnv(gym.Env):
    """
    Gymnasium Environment for the Catch game with variable falling speeds and malicious objects.
    """

    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(
        self,
        grid_size=20,
        num_objects=6,
        min_row_gap=3,
        max_row_gap=6,
        spawn_probability=0.1,
        malicious_probability=0.4,  # Probabilità di spawnare un oggetto malevolo
        min_speed=0.5,
        max_speed=1.5
    ):
        super(CatchEnv, self).__init__()

        self.time_limit = TIME_LIMIT
        self.grid_size = grid_size
        self.num_objects = num_objects
        self.min_row_gap = min_row_gap
        self.max_row_gap = max_row_gap

        # Probabilità di spawnare un frutto (o bomba)
        self.spawn_probability = spawn_probability
        self.malicious_probability = malicious_probability

        # Range velocità oggetti
        self.min_speed = min_speed
        self.max_speed = max_speed

        # Conteggio delle bombe prese
        self.malicious_catches = 0
        self.max_malicious_catches = 5

        # Definizione spazi Gym
        self.action_space = spaces.Discrete(3)  # 0 (LEFT), 1 (STAY), 2 (RIGHT)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(grid_size, grid_size), dtype=np.float32
        )

        # Stato
        self.state = None
        self.window = None
        self.clock = None

        # Liste per tracciare caratteristiche dei frutti/bombe
        self.fruit_cols = []
        self.fruit_rows = []
        self.fruit_speeds = []         # Velocità verticale di caduta
        self.fruit_is_malicious = []   # Indica se l'oggetto è malevolo (bomba)

        # Posizione del basket
        self.basket_pos = self.grid_size // 2


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.time_limit = TIME_LIMIT
        
        # Reset basket e conteggi
        self.basket_pos = self.grid_size // 2
        self.malicious_catches = 0

        # Svuotiamo le liste degli oggetti
        self.fruit_cols.clear()
        self.fruit_rows.clear()
        self.fruit_speeds.clear()
        self.fruit_is_malicious.clear()

        # Ritorniamo l’osservazione iniziale
        return self.observe(), {}


    def step(self, action):
        self.time_limit -= 1

        # Se abbiamo finito il time limit, done
        if self.time_limit <= 0:
            return self.observe(), 0, True, False, {}

        # Aggiorniamo la posizione del basket e lo stato
        self._update_state(action)

        # Calcoliamo ricompensa
        reward = self._get_reward()

        # Controllo se abbiamo raccolto troppi oggetti malevoli
        if self.malicious_catches >= self.max_malicious_catches:
            # Hai preso troppe bombe: si perde.
            return self.observe(), reward, True, False, {}

        # Proviamo a spawnare nuovi frutti/bombe
        self._spawn_new_fruits()

        return self.observe(), reward, False, False, {}


    def _update_state(self, action):
        # Decodifica azione in movimento
        move = {0: LEFT, 1: STAY, 2: RIGHT}.get(action, STAY)
        self.basket_pos = min(max(1, self.basket_pos + move), self.grid_size - 2)

        # Aggiorniamo la posizione di ogni frutto/bomba in base alla sua velocità
        for i in range(len(self.fruit_rows)):
            self.fruit_rows[i] += self.fruit_speeds[i]


    def _get_reward(self):
        """
        Calcola il reward considerando frutti e bombe catturate.
        - +1 se prendi un frutto non malevolo
        - -1 se prendi un frutto malevolo
        - -1 se un frutto non malevolo cade oltre l'ultima riga (mancato)
        - 0 se un frutto malevolo cade oltre l'ultima riga (mancato)
        """
        reward = 0
        to_remove = []

        for i in range(len(self.fruit_rows)):
            # Se un oggetto arriva all'ultima riga (o la supera)
            if self.fruit_rows[i] >= self.grid_size - 1:
                # Verifichiamo se il basket lo prende
                caught = (abs(self.fruit_cols[i] - self.basket_pos) <= 1)
                if caught:
                    # Controlla se è malevolo
                    if self.fruit_is_malicious[i]:
                        reward -= 1
                        self.malicious_catches += 1
                    else:
                        reward += 1
                    to_remove.append(i)
                else:
                    # Oggetto perso
                    if self.fruit_is_malicious[i]:
                        reward -= 0
                    else:
                        reward -= 1
                    to_remove.append(i)

        # Rimuoviamo i frutti/bombe processati
        for index in sorted(set(to_remove), reverse=True):
            del self.fruit_rows[index]
            del self.fruit_cols[index]
            del self.fruit_speeds[index]
            del self.fruit_is_malicious[index]

        return reward


    # =========================================================================
    #                          LOGICA DI SPAWN
    # =========================================================================
    def _spawn_new_fruits(self):
        """
        Tenta di spawnare un oggetto (max uno per step) se:
        - Siamo sotto il numero massimo di oggetti desiderato (self.num_objects)
        - self.spawn_probability lo consente
        - Troviamo una (row,col) valida rispettando il min_row_gap
        """
        if len(self.fruit_rows) < self.num_objects and np.random.rand() < self.spawn_probability:
            new_row = self._generate_unique_row()
            if new_row is None:
                return

            new_col = self._generate_unique_column()
            if new_col is None:
                return

            # Generiamo la velocità e decidiamo se è malevolo
            speed = np.random.uniform(self.min_speed, self.max_speed)
            is_malicious = (np.random.rand() < self.malicious_probability)

            self.fruit_rows.append(new_row)
            self.fruit_cols.append(new_col)
            self.fruit_speeds.append(speed)
            self.fruit_is_malicious.append(is_malicious)


    def _generate_unique_row(self):
        """
        Genera una row NEGATIVA, lontana almeno self.min_row_gap
        da ogni frutto già presente 'sopra' la griglia (row < 0).
        Ritorna None se non trova nulla dopo un certo numero di tentativi.
        """
        max_tries = 100
        for _ in range(max_tries):
            candidate_row = -np.random.randint(self.min_row_gap, self.max_row_gap + 1)
            if all(abs(candidate_row - r) >= self.min_row_gap for r in self.fruit_rows if r < 0):
                return candidate_row
        return None


    def _generate_unique_column(self):
        """
        Genera una colonna evitando di collidere con le colonne già occupate.
        Se vuoi permettere più frutti sulla stessa colonna, rimuovi il check.
        """
        max_tries = 100
        for _ in range(max_tries):
            candidate_col = np.random.randint(0, self.grid_size)
            if candidate_col not in self.fruit_cols:
                return candidate_col
        return None
    # =========================================================================


    def observe(self):
        """
        Restituisce una matrice grid_size x grid_size in cui:
        - 0: cella vuota
        - 1: oggetto benevolo
        - 2: oggetto malevolo
        - 3: cestino (basket)
        """
        canvas = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        for fruit_row, fruit_col, is_malicious in zip(self.fruit_rows, self.fruit_cols, self.fruit_is_malicious):
            row_int = int(round(fruit_row))
            if 0 <= row_int < self.grid_size:
                if is_malicious:
                    canvas[row_int, fruit_col] = 2.0
                else:
                    canvas[row_int, fruit_col] = 1.0

        basket_row = self.grid_size - 1
        canvas[basket_row, self.basket_pos - 1 : self.basket_pos + 2] = 3.0
        return canvas


    def render(self, mode="human"):
        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode((500, 500))
            pygame.display.set_caption("Catch Game with Continuous Objects and Malicious Items")
            self.clock = pygame.time.Clock()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                self.window = None
                return

        cell_size = 500 // self.grid_size
        self.window.fill((0, 0, 0))  # Sfondo nero

        # Disegniamo i frutti/bombe
        for row, col, malicious in zip(self.fruit_rows, self.fruit_cols, self.fruit_is_malicious):
            row_int = int(round(row))
            if 0 <= row_int < self.grid_size:
                fruit_rect = pygame.Rect(col * cell_size, row_int * cell_size, cell_size, cell_size)
                color = (255, 0, 0) if malicious else (255, 255, 0)
                pygame.draw.ellipse(self.window, color, fruit_rect)

        # Disegniamo il basket (3 celle)
        basket_rect = pygame.Rect(
            (self.basket_pos - 1) * cell_size,
            (self.grid_size - 1) * cell_size,
            cell_size * 3, cell_size
        )
        pygame.draw.rect(self.window, (0, 255, 0), basket_rect)

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])


    def close(self):
        if self.window is not None:
            pygame.quit()
            self.window = None


###############################################################################
#                          DQN AGENT WITH CNN
###############################################################################

# Parametri generali
REPLAY_BUFFER_SIZE = 30000
BATCH_SIZE = 64
LEARNING_RATE = 0.0001
EPISODES = 300
EPSILON_DECAY = 0.9995
MIN_EPSILON = 0.01
GAMMA = 0.99
TAU = 0.001  # Tasso di soft update (Polyak)

class CNNQNetwork(nn.Module):
    """
    Rete CNN per elaborare input di dimensione (1, grid_size, grid_size).
    """
    def __init__(self, grid_size, action_size):
        super(CNNQNetwork, self).__init__()
        
        # Convoluzioni: in_channels=1, out_channels=16,32 con kernel_size=3
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            # Se vuoi ridurre la dimensione, puoi aggiungere un MaxPool2d
            # nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # Se NON fai pooling, dimensione di out = (32, grid_size, grid_size).
        # Flatten => 32 * grid_size * grid_size
        self.grid_size = grid_size
        self.flatten_size = 32 * grid_size * grid_size

        self.fc_layers = nn.Sequential(
            nn.Linear(self.flatten_size, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )

    def forward(self, x):
        # x shape: [batch_size, 1, grid_size, grid_size]
        out = self.conv_layers(x)           # [batch_size, 32, grid_size, grid_size]
        out = out.view(out.size(0), -1)     # flatten
        out = self.fc_layers(out)           # [batch_size, action_size]
        return out

class DQNAgent:
    def __init__(self, action_size, grid_size=20):
        self.action_size = action_size
        self.gamma = GAMMA
        self.epsilon = 1.0
        self.epsilon_min = MIN_EPSILON
        self.epsilon_decay = EPSILON_DECAY
        self.learning_rate = LEARNING_RATE
        self.memory = deque(maxlen=REPLAY_BUFFER_SIZE)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Usando il device: {self.device}")

        # Inizializziamo la CNN
        self.model = CNNQNetwork(grid_size, action_size).to(self.device)
        self.target_model = CNNQNetwork(grid_size, action_size).to(self.device)
        self._hard_update_target()

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def _hard_update_target(self):
        """
        Copia i pesi del modello online (self.model) all'inizio o in situazioni speciali.
        """
        self.target_model.load_state_dict(self.model.state_dict())

    def _soft_update_target(self):
        """
        Polyak averaging: target = tau * online + (1 - tau) * target
        """
        with torch.no_grad():
            for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
                target_param.data.copy_(TAU * param.data + (1.0 - TAU) * target_param.data)

    def act(self, state):
        """
        Epsilon-greedy: con probabilità epsilon, esploriamo con azione random.
        Altrimenti, prendiamo argmax(Q(s,a)).
        'state' è una matrice (grid_size, grid_size).
        """
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        # Passiamo a tensore: [1, 1, grid_size, grid_size]
        state_t = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(self.device)
        q_values = self.model(state_t)
        action = torch.argmax(q_values, dim=1).item()
        return action

    def remember(self, state, action, reward, next_state, done):
        """
        Salviamo la transizione nel replay buffer.
        """
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        """
        Esegui un update su un minibatch di transizioni dal replay buffer.
        """
        if len(self.memory) < batch_size:
            return None

        minibatch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        # Convertono in tensori
        # shape -> [batch_size, grid_size, grid_size]
        states_t = torch.FloatTensor(states).unsqueeze(1).to(self.device)      # [B, 1, H, W]
        next_states_t = torch.FloatTensor(next_states).unsqueeze(1).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(np.clip(rewards, -1, 1)).to(self.device) # se desideri clipping
        dones_t = torch.FloatTensor(dones).to(self.device)

        current_q = self.model(states_t)               # [B, action_size]
        with torch.no_grad():
            next_q = self.target_model(next_states_t)  # [B, action_size]
            max_next_q = next_q.max(dim=1)[0]          # [B]

        # Creiamo i target, partendo da current_q
        target_q = current_q.clone()
        for i in range(batch_size):
            target_q[i, actions_t[i]] = rewards_t[i] + self.gamma * max_next_q[i] * (1 - dones_t[i])

        loss = self.criterion(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10)  # clip gradienti
        self.optimizer.step()

        # Aggiorniamo la rete target con soft update
        self._soft_update_target()

        # Decadimento epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.epsilon_min)

        return loss.item()


###############################################################################
#                    FUNZIONE DI TRAINING CON CNN
###############################################################################

def train_dqn(env, episodes=EPISODES, batch_size=BATCH_SIZE):
    grid_size = env.observation_space.shape[0]  # = 20 se l'ambiente è 20x20
    action_size = env.action_space.n

    agent = DQNAgent(action_size=action_size, grid_size=grid_size)

    # Per salvare reward e loss
    all_rewards = []
    all_losses = []

    writer = SummaryWriter(log_dir='runs/CatchExperimentCNN')

    for e in range(episodes):
        state, _ = env.reset()  # state: shape (grid_size, grid_size)

        total_reward = 0.0
        total_loss = 0.0

        # Ep. loop
        for t in range(500):
            action = agent.act(state)
            next_state, reward, done, _, _ = env.step(action)

            # Salviamo transizione
            agent.remember(state, action, reward, next_state, done)

            # Replay su un minibatch
            loss_value = agent.replay(batch_size)
            if loss_value is not None:
                total_loss += loss_value

            state = next_state
            total_reward += reward

            if done:
                break

        all_rewards.append(total_reward)
        all_losses.append(total_loss)

        # TensorBoard
        writer.add_scalar("Train/Reward", total_reward, e)
        writer.add_scalar("Train/Loss", total_loss, e)
        writer.add_scalar("Train/Epsilon", agent.epsilon, e)

        rolling_mean = np.mean(all_rewards[-50:]) if len(all_rewards) >= 50 else np.mean(all_rewards)
        print(f"Episode {e+1}/{episodes}, "
              f"Reward: {total_reward:.2f}, "
              f"Rolling: {rolling_mean:.2f}, "
              f"Epsilon: {agent.epsilon:.3f}, "
              f"Loss: {total_loss:.4f}")

    # Salvataggio finale
    torch.save(agent.model.state_dict(), "cnn_dqn_model.pth")
    print("Modello CNN salvato in cnn_dqn_model.pth")

    writer.close()

    # Grafici a fine training
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(all_rewards, label='Rewards per Episode')
    plt.title('Ricompensa')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(all_losses, label='Loss per Episode', color='red')
    plt.title('Loss')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()


###############################################################################
#                    FUNZIONE PER GIOCARE DA UMANO
###############################################################################

def play_human(env):
    obs, _ = env.reset()
    done = False
    while not done:
        env.render()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
                break
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    action = 0
                elif event.key == pygame.K_RIGHT:
                    action = 2
                else:
                    action = 1
                _, _, done, _, _ = env.step(action)
    env.close()


###############################################################################
#                                 MAIN
###############################################################################

if __name__ == "__main__":
    env = CatchEnv(grid_size=20)

    mode = input("Choose mode (train/human): ").strip().lower()
    if mode == "train":
        train_dqn(env)
    elif mode == "human":
        play_human(env)
    else:
        print("Invalid mode. Choose 'train' or 'human'.")
