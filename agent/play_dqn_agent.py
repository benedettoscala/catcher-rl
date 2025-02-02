import torch
import pygame
import os
import numpy as np
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from environment.catcher_simplified import CatchEnv

# Define the Q-Network class
class QNetwork(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

def play_agent(env, model_path="dqn_model.pth"):
    # Carica il modello salvato
    state_size = env.observation_space.shape[0] * env.observation_space.shape[1]
    action_size = env.action_space.n

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = QNetwork(state_size, action_size).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    state, _ = env.reset()
    state = state.flatten()
    done = False

    while not done:
        env.render()

        # Converte lo stato in un tensore e calcola l'azione migliore
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = model(state_tensor)
            action = torch.argmax(q_values, dim=1).item()

        # Esegui l'azione e aggiorna lo stato
        next_state, _, done, _, _ = env.step(action)
        state = next_state.flatten()

        # Controlla gli eventi di pygame per chiudere il gioco manualmente
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
                break

    env.close()

if __name__ == "__main__":
    env = CatchEnv(grid_size=20)

    # Specifica il percorso del modello se necessario
    model_path = "dqn_model.pth"
    
    if not model_path or not os.path.exists(model_path):
        print(f"Il modello '{model_path}' non esiste. Assicurati di averlo addestrato e salvato correttamente.")
    else:
        play_agent(env, model_path=model_path)
