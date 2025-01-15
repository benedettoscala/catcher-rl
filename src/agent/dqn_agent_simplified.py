import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import gymnasium as gym
import matplotlib.pyplot as plt
import os
import sys

# Per usare TensorBoardX
from tensorboardX import SummaryWriter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from environment.catcher_simplified import CatchEnv
import pygame

# Parametri "migliorati" con consigli
REPLAY_BUFFER_SIZE = 30000
BATCH_SIZE = 64
LEARNING_RATE = 0.0001        # Abbassato da 0.0005
EPISODES = 3000
EPSILON_DECAY = 0.9995        # Decadimento epsilon un po' più lento
MIN_EPSILON = 0.01
GAMMA = 0.99
TAU = 0.001                   # Tasso di “soft update” (Polyak averaging)

# Define the Q-Network
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

# Define the DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        self.gamma = GAMMA           # Discount factor
        self.epsilon = 1.0           # Exploration rate
        self.epsilon_min = MIN_EPSILON
        self.epsilon_decay = EPSILON_DECAY
        self.learning_rate = LEARNING_RATE
        
        self.memory = deque(maxlen=REPLAY_BUFFER_SIZE)

        # Usa la GPU se disponibile
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Usando il device: {self.device}")

        self.model = QNetwork(state_size, action_size).to(self.device)
        self.target_model = QNetwork(state_size, action_size).to(self.device)
        self._hard_update_target()  # Inizialmente partono uguali

        # Possibili alternative: nn.SmoothL1Loss() o altre
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def _hard_update_target(self):
        """
        Copia i pesi del modello online (self.model)
        all'inizio o in situazioni particolari.
        """
        self.target_model.load_state_dict(self.model.state_dict())

    def _soft_update_target(self):
        """
        Polyak averaging: target = tau * online + (1-tau) * target
        Eseguito a ogni step per maggiore stabilità.
        """
        with torch.no_grad():
            for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
                target_param.data.copy_(
                    TAU * param.data + (1.0 - TAU) * target_param.data
                )

    def act(self, state):
        # Epsilon-greedy
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.model(state)
        return torch.argmax(q_values, dim=1).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return None

        # Estraiamo un minibatch
        minibatch = random.sample(self.memory, batch_size)

        # Spezziamo il minibatch nei singoli tensori (batch)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)

        # Esempio di clipping reward: se non vuoi, commenta la riga:
        rewards = np.clip(rewards, -1, 1)  # Taglia reward a [-1, +1]
        rewards = torch.FloatTensor(rewards).to(self.device)

        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Calcoliamo Q corrente
        current_q = self.model(states)  # shape [batch_size, action_size]

        # Calcoliamo i valori Q target
        with torch.no_grad():
            next_q = self.target_model(next_states).max(dim=1)[0]  # max su ogni riga, shape [batch_size]
        
        # Creiamo la copia di current_q per riempirla con i target
        target_q = current_q.clone()
        for i in range(batch_size):
            target_q[i, actions[i]] = rewards[i] + self.gamma * next_q[i] * (1 - dones[i])

        # Ora calcoliamo la loss MSE tra current_q e target_q
        loss = self.criterion(current_q, target_q)

        # Backprop con gradient clipping
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10)  # Clip del gradiente
        self.optimizer.step()

        # Soft update del target network a ogni step
        self._soft_update_target()

        # Aggiorniamo epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.epsilon_min)

        return loss.item()


def train_dqn(env, episodes=EPISODES, batch_size=BATCH_SIZE):
    state_size = env.observation_space.shape[0] * env.observation_space.shape[1]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)

    # Per salvare reward e loss e poi visualizzarle a fine training
    all_rewards = []
    all_losses = []

    # Inizializziamo il logger di TensorBoardX
    writer = SummaryWriter(log_dir='runs/CatchExperiment')

    for e in range(episodes):
        state, _ = env.reset()
        state = state.flatten()
        total_reward = 0.0
        total_loss = 0.0

        for t in range(500):
            action = agent.act(state)
            next_state, reward, done, _, _ = env.step(action)
            next_state = next_state.flatten()

            # Salvataggio (ricordare transizione)
            agent.remember(state, action, reward, next_state, done)

            # Training su un minibatch
            loss_value = agent.replay(batch_size)
            if loss_value is not None:
                total_loss += loss_value

            # Aggiornamento stato e reward accumulata
            state = next_state
            total_reward += reward

            if done:
                break

        all_rewards.append(total_reward)
        all_losses.append(total_loss)

        # Scriviamo su TensorBoard
        writer.add_scalar("Train/Reward", total_reward, e)
        writer.add_scalar("Train/Loss", total_loss, e)
        writer.add_scalar("Train/Epsilon", agent.epsilon, e)

        # Calcoliamo rolling mean sulle ultime 50 ep
        rolling_mean = np.mean(all_rewards[-50:]) if len(all_rewards) >= 50 else np.mean(all_rewards)
        
        print(f"Episode {e+1}/{episodes}, "
              f"Reward: {total_reward:.2f}, "
              f"Rolling: {rolling_mean:.2f}, "
              f"Epsilon: {agent.epsilon:.3f}, "
              f"Loss: {total_loss:.4f}")

    # Salviamo il modello a fine training
    torch.save(agent.model.state_dict(), "dqn_model.pth")
    print("Modello salvato in dqn_model.pth")

    # Chiudiamo il writer di TensorBoard a fine training
    writer.close()

    # (Opzionale) Grafico a fine training
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


def play_human(env):
    env.reset()
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


if __name__ == "__main__":
    env = CatchEnv(grid_size=20)

    mode = input("Choose mode (train/human): ").strip().lower()
    if mode == "train":
        train_dqn(env)
    elif mode == "human":
        play_human(env)
    else:
        print("Invalid mode. Choose 'train' or 'human'.")
