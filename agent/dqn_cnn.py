import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import numpy as np
import matplotlib.pyplot as plt
import sys, os
from tensorboardX import SummaryWriter
import pygame

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from environment.catcher_image import CatchEnvImageChangeDirection
from environment.catcher_image import CatchEnvImage
from networks.QNetwork import CNNQNetwork

# Parametri generali
REPLAY_BUFFER_SIZE = 30000
BATCH_SIZE = 64
LEARNING_RATE = 0.0001
EPISODES = 1000
# EPSILON_DECAY = 0.9995  # Removed exponential decay
MIN_EPSILON = 0.01
GAMMA = 0.99
TAU = 0.001  # Tasso di soft update (Polyak)
EPSILON_DECAY_EPISODES = 700  # Number of episodes over which to decay epsilon

###############################################################################
#                      AGENTE DQN CON CNN (MODIFICATO)
###############################################################################
class DQNAgent:
    def __init__(self, action_size, grid_size=15, in_channels=2):
        
        self.action_size = action_size
        self.gamma = GAMMA
        self.epsilon = 1.0
        self.epsilon_min = MIN_EPSILON
        # self.epsilon_decay = EPSILON_DECAY  # Removed exponential decay
        self.learning_rate = LEARNING_RATE
        self.memory = deque(maxlen=REPLAY_BUFFER_SIZE)
        self.in_channels = in_channels

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Usando il device: {self.device}")

        print(f"I channels sono {in_channels}")
        # Inizializziamo la CNN con 2 canali in input
        self.model = CNNQNetwork(grid_size, action_size, in_channels).to(self.device)
        self.target_model = CNNQNetwork(grid_size, action_size, in_channels).to(self.device)
        self._hard_update_target()

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def _hard_update_target(self):
        """Copia i pesi del modello online (self.model) all'inizio o in situazioni speciali."""
        self.target_model.load_state_dict(self.model.state_dict())

    def _soft_update_target(self):
        """Polyak averaging: target = tau * online + (1 - tau) * target"""
        with torch.no_grad():
            for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
                target_param.data.copy_(TAU * param.data + (1.0 - TAU) * target_param.data)

    def act(self, state):
        """
        Epsilon-greedy: con probabilità epsilon, esploriamo con azione random.
        Altrimenti, prendiamo argmax(Q(s,a)).
        
        'state' dovrebbe avere shape (2, grid_size, grid_size) se l'ambiente 
        include il canale di velocità.
        """
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        # Passiamo a tensore: [1, 2, grid_size, grid_size]
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.model(state_t)
        action = torch.argmax(q_values, dim=1).item()
        return action

    def remember(self, state, action, reward, next_state, done):
        """Salviamo la transizione nel replay buffer."""
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        """
        Esegui un update su un minibatch di transizioni dal replay buffer.
        """
        if len(self.memory) < batch_size:
            return None

        minibatch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        # Le dimensioni di 'states' sono [batch_size, 2, grid_size, grid_size]
        states_t = torch.FloatTensor(states).to(self.device)       # [B, 2, H, W]
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(np.clip(rewards, -1, 1)).to(self.device) 
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

        # Removed epsilon decay from replay
        # if self.epsilon > self.epsilon_min:
        #     self.epsilon *= self.epsilon_decay
        #     self.epsilon = max(self.epsilon, self.epsilon_min)

        return loss.item()


###############################################################################
#                    FUNZIONE DI TRAINING CON CNN
###############################################################################
def train_dqn(env, logdir, savedir, episodes=EPISODES, batch_size=BATCH_SIZE):
    # Check if savedir exists 
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    
    # ora observation_space.shape = (2, grid_size, grid_size)
    grid_size = env.observation_space.shape[1]  # la dimensione spaziale è al secondo posto (2, H, W) => H=grid_size
    action_size = env.action_space.n
    agent = DQNAgent(action_size=action_size, grid_size=grid_size, in_channels=env.in_channels)

    # Per salvare reward e loss
    all_rewards = []
    all_losses = []

    writer = SummaryWriter(log_dir=logdir)

    # Calcola quanto diminuire epsilon ogni episodio
    epsilon_step = (agent.epsilon - agent.epsilon_min) / EPSILON_DECAY_EPISODES

    for e in range(episodes):
        state, _ = env.reset()  # state: shape (2, grid_size, grid_size)
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

        rolling_mean = np.mean(all_rewards[-50:]) if len(all_rewards) >= 50 else np.mean(all_rewards)

        # TensorBoard
        writer.add_scalar("Train/Reward", total_reward, e)
        writer.add_scalar("Train/Loss", total_loss, e)
        writer.add_scalar("Train/Epsilon", agent.epsilon, e)
        writer.add_scalar("Train/rolling", rolling_mean, e)
        

        print(f"Episode {e+1}/{episodes}, "
              f"Reward: {total_reward:.2f}, "
              f"Rolling: {rolling_mean:.2f}, "
              f"Epsilon: {agent.epsilon:.3f}, "
              f"Loss: {total_loss:.4f}")
        
        # Decadimento epsilon lineare basato su EPSILON_DECAY_EPISODES
        if e < EPSILON_DECAY_EPISODES:
            agent.epsilon -= epsilon_step
            agent.epsilon = max(agent.epsilon, agent.epsilon_min)
        else:
            agent.epsilon = agent.epsilon_min  # Ensure epsilon stays at min after decay period

        # Save the model every 10 episodes
        if (e + 1) % 10 == 0:
            torch.save(agent.model.state_dict(), os.path.join(savedir, f"dqn_model_ep{e+1}.pth"))
            print(f"Modello CNN salvato in {savedir}")

    # Salvataggio finale
    torch.save(agent.model.state_dict(), os.path.join(savedir, "dqn_model_final.pth"))
    print(f"Modello CNN salvato in {savedir}")

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
    # Assicurati che il tuo CatchEnv restituisca osservazioni di shape (2, grid_size, grid_size)
    
    # Fai scegliere all'utente che tipo di ambiente vuole usare
    choice = input("Scegliere l'ambiente da utilizzare: 1) CatchEnv 2) CatchEnvImageChangeDirection: ")
    # Se la cartella models non esiste, creala
    savedir = "dqn_cnn_models"
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    
    if choice == "1":
        env = CatchEnvImage(grid_size=15)

        no_direction_savedir = os.path.join(savedir, "no_direction")
        if not os.path.exists(no_direction_savedir):
            os.makedirs(no_direction_savedir)
        train_dqn(env, logdir="runs/cnn_dqn_catchenv", savedir=no_direction_savedir)
        
    elif choice == "2":
        env = CatchEnvImageChangeDirection(grid_size=15)
        direction_savedir = os.path.join(savedir, "direction")
        if not os.path.exists(direction_savedir):
            os.makedirs(direction_savedir)
        train_dqn(env, logdir="runs/cnn_dqn_catchenv_with_direction", savedir=direction_savedir)
    else:
        print("Scelta non valida. Uscita.")
