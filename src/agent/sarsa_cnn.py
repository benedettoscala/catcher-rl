import sys
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import pygame

import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

# Se la cartella environment è un modulo Python e contiene catcher_cnn.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from environment.catcher_cnn import CatchEnv  # importa il tuo environment
from networks.QNetwork import CNNQNetwork     # importa la tua rete CNN

# Hyperparam
EPISODES = 1000
MAX_STEPS = 500
LEARNING_RATE = 1e-4
GAMMA = 0.99
LAMBDA = 0.9        # <-- NUOVO: lambda per le eligibility traces
EPSILON_DECAY = 0.9999
MIN_EPSILON = 0.01

###############################################################################
#                        AGENTE SARSA(λ) CON CNN
###############################################################################
class SarsaLambdaAgent:
    def __init__(self, action_size, grid_size=20, lambda_=LAMBDA):
        """
        SARSA(λ) agent che usa una CNN come approssimatore di Q(s,a).
        """
        self.action_size = action_size
        self.epsilon = 1.0
        self.epsilon_min = MIN_EPSILON
        self.epsilon_decay = EPSILON_DECAY
        self.gamma = GAMMA
        self.alpha = LEARNING_RATE
        self.lambda_ = lambda_

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Usando il device: {self.device}")

        # Rete CNN (due canali in input: [2, grid_size, grid_size])
        self.model = CNNQNetwork(grid_size, action_size).to(self.device)

        # Creiamo le eligibility traces per ognuno dei parametri della rete
        self.eligibility = []
        for param in self.model.parameters():
            # Stessa shape, inizializzata a zero
            self.eligibility.append(torch.zeros_like(param, device=self.device))

    def reset_eligibility(self):
        """
        Resetta le eligibility traces a inizio episodio (o quando l'episodio termina).
        """
        for e in self.eligibility:
            e.zero_()

    def act(self, state):
        """
        Epsilon-greedy: con probabilità epsilon, esploro con azione random;
        altrimenti uso argmax(Q(s,a)).
        """
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        # Converte in tensore float su device
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # [1, 2, H, W]
        q_values = self.model(state_t)  # [1, action_size]
        action = torch.argmax(q_values, dim=1).item()
        return action

    def learn(self, state, action, reward, next_state, next_action, done):
        """
        Update di SARSA(λ) backward view:
          δ = r + γ Q(s', a') - Q(s,a)
          e_θ ← γ λ e_θ + ∂Q(s,a)/∂θ
          θ ← θ + α δ e_θ

        Se done=True, Q(s',a')=0 e azzeriamo le eligibility a fine episodio.
        """

        # Converte in tensori float su device
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        next_state_t = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)

        # Otteniamo Q(s,·) e Q(s',·)
        q_values = self.model(state_t)            # [1, action_size]
        q_sa = q_values[0, action]                # scalare

        if done:
            # Se l'episodio è terminato, non c'è Q(s', a')
            q_s_next_a_next = 0.0
        else:
            # Altrimenti valutiamo la rete su s' e prendiamo Q(s', next_action)
            q_values_next = self.model(next_state_t)  # [1, action_size]
            q_s_next_a_next = q_values_next[0, next_action]

        # TD error
        delta = reward + self.gamma * q_s_next_a_next - q_sa

        # Calcoliamo il gradiente di Q(s,a) wrt i parametri
        self.model.zero_grad()        # azzera i gradienti
        q_sa.backward()               # calcola gradiente di q_sa = Q(s,a)
        
        # Aggiorniamo le eligibility e poi i parametri
        with torch.no_grad():
            # Per ogni parametro p e relativa trace e_z
            for (p, e_z) in zip(self.model.parameters(), self.eligibility):
                # e_z ← γ λ e_z + ∂Q(s,a)/∂p
                e_z *= (self.gamma * self.lambda_)
                e_z += p.grad

                # p ← p + α * δ * e_z
                p += self.alpha * delta * e_z

        # Decadimento epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.epsilon_min)

        # Se l'episodio è finito, azzeriamo le eligibility traces
        if done:
            self.reset_eligibility()

        # Ritorno un valore di loss “simbolico”: usiamo la TD-error al posto del MSE
        # giusto per tracciare un numero e avere log su TensorBoard.
        td_error_value = delta.detach().cpu().item()
        return abs(td_error_value)

###############################################################################
#                  FUNZIONE DI TRAINING SARSA(λ) + CNN
###############################################################################
def train_sarsa_lambda(env, episodes=EPISODES):
    grid_size = env.observation_space.shape[1]  # (2, grid_size, grid_size)
    action_size = env.action_space.n

    agent = SarsaLambdaAgent(action_size=action_size, grid_size=grid_size, lambda_=LAMBDA)

    # Per logging e statistiche
    writer = SummaryWriter(log_dir='runs/CatchExperiment_SARSA_Lambda_CNN')
    all_rewards = []
    all_td_errors = []

    for e in range(episodes):
        state, _ = env.reset()
        agent.reset_eligibility()   # azzeriamo le trace a inizio episodio
        total_reward = 0.0
        total_td_error = 0.0

        # Scegli azione iniziale con la policy epsilon-greedy
        action = agent.act(state)

        for step in range(MAX_STEPS):
            # Esegui azione, ottieni next_state
            next_state, reward, done, _, _ = env.step(action)

            # Scegli la prossima azione (on-policy)
            next_action = agent.act(next_state)

            # Update SARSA(λ)
            td_error = agent.learn(state, action, reward, next_state, next_action, done)
            total_td_error += td_error

            state = next_state
            action = next_action
            total_reward += reward

            if done:
                break

        all_rewards.append(total_reward)
        all_td_errors.append(total_td_error)

        # Logging su TensorBoard (usiamo la TD error come "loss")
        writer.add_scalar("Train/Reward", total_reward, e)
        writer.add_scalar("Train/TD_Error", total_td_error, e)
        writer.add_scalar("Train/Epsilon", agent.epsilon, e)

        rolling_mean = np.mean(all_rewards[-50:]) if len(all_rewards) >= 50 else np.mean(all_rewards)
        print(f"Episode {e+1}/{episodes} | "
              f"Reward: {total_reward:.2f} | "
              f"Rolling: {rolling_mean:.2f} | "
              f"Epsilon: {agent.epsilon:.3f} | "
              f"TD Error: {total_td_error:.4f}")

    # Salvataggio pesi finali
    torch.save(agent.model.state_dict(), "cnn_sarsa_lambda_model.pth")
    print("Modello CNN (SARSA λ) salvato in cnn_sarsa_lambda_model.pth")

    writer.close()

    # Grafici a fine training
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(all_rewards, label='Rewards per Episode')
    plt.title('Ricompensa (SARSA λ)')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(all_td_errors, label='TD Error per Episode', color='red')
    plt.title('TD Error (SARSA λ)')
    plt.xlabel('Episode')
    plt.ylabel('Sum of TD Errors')
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
                # Azioni: 0=LEFT, 1=STAY, 2=RIGHT
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
    # Assicurati che CatchEnv restituisca un'osservazione shape (2, grid_size, grid_size)
    env = CatchEnv(grid_size=20)

    mode = input("Choose mode (train/human): ").strip().lower()
    if mode == "train":
        train_sarsa_lambda(env)   # <-- usa SARSA(λ)
    elif mode == "human":
        play_human(env)
    else:
        print("Invalid mode. Choose 'train' or 'human'.")
