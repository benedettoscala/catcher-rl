import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from environment.catcher_cnn import CatchEnv
from networks.QNetwork import CNNQNetwork
from tensorboardX import SummaryWriter


# from your_file import CatchEnv  # Import your custom environment if it's in another file.
# Otherwise, ensure the CatchEnv code is already defined in this script/notebook.

# -------------------------------------------------------------------
# 1) Istanziamo l'ambiente
# -------------------------------------------------------------------
class SarsaCNN:
    def __init__(self, grid_size=20, num_objects=6, spawn_probability=0.1, malicious_probability=0.4, min_speed=0.5, max_speed=1.5, lr=1e-3, gamma=0.99, num_episodes=200, max_steps_per_episode=300, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.995):
        self.env = CatchEnv(
            grid_size=grid_size,
            num_objects=num_objects,
            spawn_probability=spawn_probability,
            malicious_probability=malicious_probability,
            min_speed=min_speed,
            max_speed=max_speed
        )
        self.n_actions = self.env.action_space.n
        self.grid_size = self.env.grid_size
        self.obs_shape = self.env.observation_space.shape
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_net = CNNQNetwork(grid_size=self.grid_size, action_size=self.n_actions).to(self.device)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.gamma = gamma
        self.num_episodes = num_episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.scores = []
        self.epsilon = epsilon_start

    def select_action(self, state):
        if random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_net(state_t)
            return q_values.argmax(dim=1).item()

    def train(self):

        writer = SummaryWriter(log_dir='runs/SarsaCNNExperiment')

        for episode in range(self.num_episodes):
            obs, _ = self.env.reset()
            action = self.select_action(obs)
            total_reward = 0.0

            for t in range(self.max_steps_per_episode):
                next_obs, reward, done, truncated, info = self.env.step(action)
                next_action = self.select_action(next_obs)
                obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                next_obs_t = torch.FloatTensor(next_obs).unsqueeze(0).to(self.device)
                q_values = self.q_net(obs_t)
                q_value = q_values[0, action]
                with torch.no_grad():
                    q_values_next = self.q_net(next_obs_t)
                    q_value_next = q_values_next[0, next_action]
                target = reward + (0.0 if (done or truncated) else self.gamma * q_value_next)
                q_value = q_value.unsqueeze(0)
                target = torch.FloatTensor([target]).to(self.device)
                loss = F.mse_loss(q_value, target)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                obs = next_obs
                action = next_action
                total_reward += reward
                if done or truncated:
                    break
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
            self.scores.append(total_reward)

            writer.add_scalar('Reward', total_reward, episode)
            writer.add_scalar('Epsilon', self.epsilon, episode)
            writer.add_scalar('Loss', loss.item(), episode)


            if (episode + 1) % 10 == 0:
                avg_score = np.mean(self.scores[-10:])
                print(f"Episode: {episode + 1}, Avg Reward (last 10): {avg_score:.2f}, Epsilon: {self.epsilon:.2f}, Loss {loss.item():.4f}")

        torch.save(self.q_net.state_dict(), 'sarsa_cnn.pth')

        writer.close()
        self.env.close()

sarsa_cnn = SarsaCNN()
sarsa_cnn.train()
print("Training finished!")
