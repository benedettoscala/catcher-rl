import numpy as np
import logging
from collections import deque
from tensorboardX import SummaryWriter
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import time

# Aggiungi il percorso del modulo (assumendo che questo script sia nella stessa cartella del trainer)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from environment.catcher_discrete import CatchEnvChangeDirection  # Usa il tuo file del tuo env
from environment.catcher_discrete import CatchEnv  # Usa il tuo file del tuo env

class QNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(QNetwork, self).__init__()
        layers = []
        last_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(last_size, hidden_size))
            layers.append(nn.ReLU())
            last_size = hidden_size
        layers.append(nn.Linear(last_size, output_size))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class CatcherPlayer:
    def __init__(self, model_path, direction_on=False, grid_size=15, hidden_sizes=[128, 128], render_mode="human"):
        # Configura il logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # Inizializza l'ambiente
        if direction_on:
            self.env = CatchEnvChangeDirection(
                grid_size=grid_size,
                max_objects_in_state=2,
                render_mode=render_mode
            )
        else:
            self.env = CatchEnv(
                grid_size=grid_size,
                max_objects_in_state=2,
                render_mode=render_mode
            )
        
        # Verifica se l'ambiente ha l'attributo 'direction'
        self.direction = getattr(self.env, 'direction', False)
        self.logger.info(f"Ambiente con direzione: {self.direction}")
        
        # Imposta i parametri dello stato
        if self.direction:
            self.state_size = (
                1,  # basket_pos
                5,  # row1, col1, type1, v_speed1, h_speed1
                5   # row2, col2, type2, v_speed2, h_speed2
            )
            self.input_size = 1 + 5 + 5  # 11
        else:
            self.state_size = (
                1,  # basket_pos
                4,  # row1, col1, type1, speed1
                4   # row2, col2, type2, speed2
            )
            self.input_size = 1 + 4 + 4  # 9
        
        self.num_actions = self.env.action_space.n
        self.logger.info(f"Input size per la rete neurale: {self.input_size}")
        self.logger.info(f"Numero di azioni: {self.num_actions}")
        
        # Inizializza la rete neurale
        self.q_network = QNetwork(
            input_size=self.input_size,
            hidden_sizes=hidden_sizes,
            output_size=self.num_actions
        )
        
        # Carica i pesi del modello
        if not os.path.exists(model_path):
            self.logger.error(f"Il percorso del modello {model_path} non esiste.")
            sys.exit(1)
        
        self.q_network.load_state_dict(torch.load(model_path))
        self.q_network.eval()  # Imposta la rete in modalità valutazione
        self.logger.info(f"Modello caricato da {model_path}")
    
    def state_to_tensor(self, state):
        """
        Converte lo stato in un tensore normalizzato adatto alla rete neurale.
        """
        if self.direction:
            (basket_pos,
             row1, col1, type1, v_speed1, h_speed1,
             row2, col2, type2, v_speed2, h_speed2) = state
            state_flat = [
                basket_pos,
                row1, col1, type1, v_speed1, h_speed1,
                row2, col2, type2, v_speed2, h_speed2
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
        state_tensor = torch.FloatTensor(state_flat).unsqueeze(0)  # Shape: [1, input_size]
        return state_tensor
    
    def select_action(self, state_tensor):
        """
        Seleziona l'azione con il più alto valore Q senza esplorazione.
        """
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
            action = torch.argmax(q_values).item()
        return action
    
    def play_episode(self, render=True):
        """
        Esegue un episodio di gioco utilizzando il modello caricato.
        """
        obs, _ = self.env.reset()
        state_tensor = self.state_to_tensor(obs)
        done = False
        total_reward = 0
        step = 0
        
        while not done:
            if render:
                self.env.render()
                time.sleep(0.1)  # Aggiungi un piccolo ritardo per rendere il rendering visibile
            
            action = self.select_action(state_tensor)
            next_obs, reward, done, _, _ = self.env.step(action)
            next_state_tensor = self.state_to_tensor(next_obs)
            total_reward += reward
            state_tensor = next_state_tensor
            step += 1
        
        if render:
            self.env.close()
        
        self.logger.info(f"Episodio terminato in {step} step con reward totale: {total_reward}")
        return total_reward

    def close(self):
        """
        Chiude l'ambiente.
        """
        self.env.close()

def main():
    parser = argparse.ArgumentParser(description="Gioca a Catcher utilizzando un modello SARSA addestrato.")
    parser.add_argument('--model', type=str, required=True, help='Percorso al file del modello (.pth)')
    parser.add_argument('--environment', type=int, choices=[1, 2], default=1,
                        help='Tipo di ambiente: 1 per CatchEnv, 2 per CatchEnvChangeDirection')
    parser.add_argument('--grid_size', type=int, default=15, help='Dimensione della griglia dell\'ambiente')
    args = parser.parse_args()
    
    direction_on = True if args.environment == 2 else False
    logdir = "play_catcher_logs"
    
    player = CatcherPlayer(
        model_path=args.model,
        direction_on=direction_on,
        grid_size=args.grid_size
    )
    
    try:
        while True:
            player.play_episode(render=True)
            # Chiedi all'utente se vuole continuare
            cont = input("Vuoi giocare un altro episodio? (s/n): ").strip().lower()
            if cont != 's':
                break
    except KeyboardInterrupt:
        print("\nInterrotto dall'utente.")
    finally:
        player.close()

if __name__ == "__main__":
    main()
