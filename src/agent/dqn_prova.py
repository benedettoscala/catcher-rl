import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from environment.catcher import CatcherEnv

# Modello della rete neurale
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.network(x)

# Funzione per selezionare l'azione
def select_action(state, policy_net, epsilon, action_space):
    if random.random() < epsilon:
        return random.choice(range(action_space))
    with torch.no_grad():
        return torch.argmax(policy_net(state)).item()

# Funzione di aggiornamento del DQN
def optimize_model(policy_net, target_net, memory, optimizer, batch_size, gamma, device):
    if len(memory) < batch_size:
        return

    transitions = random.sample(memory, batch_size)
    state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*transitions)

    state_batch = torch.cat(state_batch).to(device)
    action_batch = torch.tensor(action_batch, device=device).unsqueeze(1)
    reward_batch = torch.tensor(reward_batch, device=device).unsqueeze(1)
    next_state_batch = torch.cat(next_state_batch).to(device)
    done_batch = torch.tensor(done_batch, device=device, dtype=torch.float32).unsqueeze(1)

    q_values = policy_net(state_batch).gather(1, action_batch)
    next_q_values = target_net(next_state_batch).max(1)[0].detach().unsqueeze(1)
    expected_q_values = reward_batch + (gamma * next_q_values * (1 - done_batch))

    loss = nn.MSELoss()(q_values, expected_q_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Configurazione
num_episodes = 500
gamma = 0.99
batch_size = 64
epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = 500
learning_rate = 1e-3
memory_size = 10000
target_update = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Utilizzo GPU: {torch.cuda.is_available()}")

# Inizializzazione ambiente e rete
env = CatcherEnv()
state_dim = env.observation_space['paddle_pos'].shape[0] + env.num_objects * 4
num_actions = env.action_space.n

policy_net = DQN(state_dim, num_actions).to(device)
target_net = DQN(state_dim, num_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
memory = deque(maxlen=memory_size)

epsilon = epsilon_start

import pygame
# Configura un clock per il frame rate
clock = pygame.time.Clock()

# Addestramento
for episode in range(num_episodes):
    # Attiva il render ogni 10 episodi
    render_mode = (episode % 1000 == 0) and episode != 0
    
    
    if render_mode:
        env.render_mode = True

    

    state = env.reset()
    state_tensor = torch.tensor(
        np.concatenate((state['paddle_pos'].cpu().numpy().flatten(), state['objects'].cpu().numpy().reshape(-1))),
        dtype=torch.float32
    ).unsqueeze(0).to(device)

    total_reward = 0
    done = False

    step_count = 0
    
    while not done:
        step_count += 1
        # Gestione degli eventi Pygame per evitare blocchi
        if render_mode:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True  # Termina il ciclo se la finestra viene chiusa

        action = select_action(state_tensor, policy_net, epsilon, num_actions)
        next_state, reward, done, _, _ = env.step(action)

        next_state_tensor = torch.tensor(
            np.concatenate((next_state['paddle_pos'].cpu().numpy().flatten(), next_state['objects'].cpu().numpy().reshape(-1))),
            dtype=torch.float32
        ).unsqueeze(0).to(device)

        memory.append((state_tensor, action, reward, next_state_tensor, done))

        state_tensor = next_state_tensor
        total_reward += reward

        optimize_model(policy_net, target_net, memory, optimizer, batch_size, gamma, device)

        # dopo il 1000esimo step si ferma
        if step_count > 2000:
            print("Step count > 1000")
            step_count = 0
            done = True
        
        # Limita il frame rate durante il rendering
        if render_mode:
            clock.tick(30)

    # Disabilita il render se attivato per questo episodio
    if render_mode:
        env.render_mode = False

    # Aggiorna il target network
    if episode % target_update == 0:
        target_net.load_state_dict(policy_net.state_dict())

    # Salva un checkpoint ogni 50 episodi
    if (episode + 1) % 50 == 0:
        checkpoint_path = f"checkpoint_episode_{episode + 1}.pth"
        torch.save({
            'policy_net_state_dict': policy_net.state_dict(),
            'target_net_state_dict': target_net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'episode': episode + 1,
            'epsilon': epsilon
        }, checkpoint_path)
        print(f"Checkpoint salvato: {checkpoint_path}")

    # Aggiorna epsilon
    epsilon = max(epsilon_end, epsilon * np.exp(-1 / epsilon_decay))

    print(f"Episodio {episode + 1}, Ricompensa Totale: {total_reward}")

# Salvataggio dei modelli finali
torch.save(policy_net.state_dict(), "policy_net.pth")
torch.save(target_net.state_dict(), "target_net.pth")
print("Modelli salvati con successo.")

env.close()
