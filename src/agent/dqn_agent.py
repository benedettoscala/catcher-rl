import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple, deque
import random
import math
from itertools import count
import matplotlib.pyplot as plt
from environment.catcher import CatcherEnv

# Definizione di DQN (Deep Q-Network)
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# Replay Memory
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# Impostazioni principali
BATCH_SIZE = 64
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 1000
LR = 1e-4
MEMORY_CAPACITY = 10000
NUM_EPISODES = 500

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Inizializzazione dell'ambiente
env = CatcherEnv(render_mode=False, object_speed=7, paddle_speed=15, num_objects=5)
obs = env.reset()

# Determinazione delle dimensioni input-output
input_size = len(obs['paddle_pos']) + obs['objects'].shape[0] * obs['objects'].shape[1]
output_size = env.action_space.n

# Reti neurali e ottimizzatore
policy_net = DQN(input_size, output_size).to(device)
target_net = DQN(input_size, output_size).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LR)
memory = ReplayMemory(MEMORY_CAPACITY)

steps_done = 0

# Funzione per selezionare un'azione
def select_action(state):
    global steps_done
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if random.random() > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(output_size)]], device=device, dtype=torch.long)

# Ottimizzazione del modello
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

# Addestramento
for i_episode in range(NUM_EPISODES):
    obs = env.reset()
    state = torch.cat([
        torch.tensor(obs['paddle_pos'], device=device, dtype=torch.float32),
        torch.tensor(obs['objects'].flatten(), device=device, dtype=torch.float32)
    ]).unsqueeze(0)

    for t in count():
        action = select_action(state)
        obs, reward, done, truncated, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)

        if not done:
            next_state = torch.cat([
                torch.tensor(obs['paddle_pos'], device=device, dtype=torch.float32),
                torch.tensor(obs['objects'].flatten(), device=device, dtype=torch.float32)
            ]).unsqueeze(0)
        else:
            next_state = None

        memory.push(state, action, next_state, reward)
        state = next_state

        optimize_model()

        if done or truncated:
            break

    # Aggiornamento della rete target
    target_net.load_state_dict(policy_net.state_dict())

print("Addestramento completato")
env.close()
