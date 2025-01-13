import torch
import numpy as np
from environment.catcher import CatcherEnv
import torch.nn as nn
import pygame  # Necessario per gestire gli eventi

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
def select_action(state, policy_net):
    with torch.no_grad():
        return torch.argmax(policy_net(state)).item()

# Configurazione
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Utilizzo GPU: {torch.cuda.is_available()}")

# Inizializzazione ambiente e rete
env = CatcherEnv(render_mode=True)
state_dim = env.observation_space['paddle_pos'].shape[0] + env.num_objects * 4
num_actions = env.action_space.n

policy_net = DQN(state_dim, num_actions).to(device)

# Caricamento del checkpoint
checkpoint_path = "checkpoint_episode_400.pth"  # Cambia il nome se necessario
checkpoint = torch.load(checkpoint_path)
policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
policy_net.eval()
print(f"Modello caricato dal checkpoint: {checkpoint_path}")

# Gioco automatizzato
state = env.reset()
state_tensor = torch.tensor(
    np.concatenate((state['paddle_pos'].cpu().numpy().flatten(), state['objects'].cpu().numpy().reshape(-1))),
    dtype=torch.float32
).unsqueeze(0).to(device)

done = False
total_reward = 0
clock = pygame.time.Clock()

while not done:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True  # Permette di chiudere la finestra manualmente

    action = select_action(state_tensor, policy_net)
    next_state, reward, done, _, _ = env.step(action)

    next_state_tensor = torch.tensor(
        np.concatenate((next_state['paddle_pos'].cpu().numpy().flatten(), next_state['objects'].cpu().numpy().reshape(-1))),
        dtype=torch.float32
    ).unsqueeze(0).to(device)

    state_tensor = next_state_tensor
    total_reward += reward

    clock.tick(30)  # Limita il frame rate

print(f"Ricompensa Totale: {total_reward}")
env.close()
