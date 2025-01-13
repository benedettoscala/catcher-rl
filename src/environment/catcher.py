import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random
import pygame
import time
import torch

class CatcherEnv(gym.Env):
    """
    Ambiente per il gioco Catcher con supporto per rendering tramite Pygame e controllo utente.
    Modificato per supportare GPU tramite PyTorch.
    """
    def __init__(self, render_mode=False, user_control=False, object_speed=5, paddle_speed=10, num_objects=5):
        super(CatcherEnv, self).__init__()

        # Dimensione dell'area di gioco
        self.width = 800
        self.height = 600

        # Configurazioni dinamiche
        self.render_mode = render_mode
        self.user_control = user_control
        self.object_speed = object_speed
        self.paddle_speed = paddle_speed
        self.num_objects = num_objects

        # Azioni possibili: sinistra (-1), destra (+1), stazionamento (0)
        self.action_space = spaces.Discrete(3)

        # Osservazioni: posizione del paddle, posizione degli oggetti (x, y), velocità, punteggio, vite, tempo che il paddle è rimasto fermo
        self.observation_space = spaces.Dict({
            "paddle_pos": spaces.Box(low=0, high=self.width, shape=(1,), dtype=np.float32),
            "objects": spaces.Box(
                low=np.tile(np.array([0, 0, -self.object_speed]), (self.num_objects, 1)),  # Estende [0, 0, -speed] a (num_objects, 3)
                high=np.tile(np.array([self.width, self.height, self.object_speed]), (self.num_objects, 1)),  # Estende [width, height, speed] a (num_objects, 3)
                shape=(self.num_objects, 3),
                dtype=np.float32
            ),
            "score_beneficial": spaces.Discrete(1000),
            "score_harmful": spaces.Discrete(1000),
            "lives": spaces.Discrete(3),
            "time_stationary": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),  # Nuova chiave
        })

        # Variabili Pygame
        self.window = None
        self.clock = None
        self.paddle_width = 100
        self.paddle_height = 20
        self.object_size = 30

        # Tracciamento del movimento del paddle
        self.last_paddle_pos = self.width / 2  # Posizione iniziale del paddle
        self.no_move_steps = 0  # Contatore dei passi senza movimento
        self.max_no_move_steps = 6  # Soglia massima di passi senza movimento per penalità
        
        # GPU support
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.reset()

    def reset(self):
        """
        Resetta l'ambiente ai valori iniziali.
        """
        self.paddle_pos = self.width / 2
        self.objects = [
            {
                "pos": [random.uniform(0, self.width), random.uniform(-self.height, 0)],
                "velocity": [0, random.uniform(self.object_speed * 0.8, self.object_speed * 1.2)],
                "type": random.choice(["beneficial", "harmful"]),
            }
            for _ in range(self.num_objects)
        ]
        self.score_beneficial = 0
        self.score_harmful = 0
        self.lives = 3
        self.start_time = time.time()

        # Resetta il tracciamento del movimento
        self.last_paddle_pos = self.paddle_pos
        self.no_move_steps = 0  # Reset del contatore dei passi

        if self.render_mode:
            self._init_pygame()

        return self._get_obs()

    def step(self, action):
        """
        Esegue un passo nell'ambiente dato un'azione.
        """
        if not self.user_control:
            if action == 0:  # Sinistra
                self.paddle_pos = max(0, self.paddle_pos - self.paddle_speed)
            elif action == 2:  # Destra
                self.paddle_pos = min(self.width - self.paddle_width, self.paddle_pos + self.paddle_speed)

        reward = 0

        # Verifica se il paddle è rimasto fermo
        if self.paddle_pos != self.last_paddle_pos:
            self.last_paddle_pos = self.paddle_pos
            self.no_move_steps = 0  # Reset del contatore
        else:
            self.no_move_steps += 1

        # Penalità per essere rimasti fermi troppo a lungo
        if self.no_move_steps > self.max_no_move_steps:
            reward -= 100  # Penalità per essere rimasti fermi troppo a lungo
            self.no_move_steps = 0  # Reset del contatore dopo la penalità

        for obj in self.objects:
            obj["pos"][1] += obj["velocity"][1]

            # Controlla collisioni
            if obj["pos"][1] >= self.height - self.paddle_height:
                if self.paddle_pos <= obj["pos"][0] <= self.paddle_pos + self.paddle_width:
                    if obj["type"] == "beneficial":
                        reward += 10  # Ricompensa per oggetto benefico raccolto
                        self.score_beneficial += 1
                    else:
                        reward -= 15  # Penalità per oggetto dannoso raccolto
                        self.lives -= 1
                else:
                    if obj["type"] == "beneficial":
                        reward -= 5  # Penalità per mancato oggetto benefico
                    elif obj["type"] == "harmful":
                        reward += 3  # Piccola ricompensa per aver evitato oggetto dannoso
                        self.score_harmful += 1

                # Respawn oggetto
                obj["pos"] = [random.uniform(0, self.width), random.uniform(-self.height, 0)]

        done = self.lives <= 0
        truncated = False

        if self.render_mode:
            self.render()

        return self._get_obs(), reward, done, truncated, {}

    def _get_obs(self):
        """
        Restituisce lo stato osservabile dell'ambiente, inclusa la posizione e il tipo degli oggetti.
        """
        objects_with_type_and_velocity = [
            obj["pos"] + [obj["velocity"][1]] + [1 if obj["type"] == "beneficial" else 0]  # Aggiunge velocità e tipo
            for obj in self.objects
        ]

        # Tempo fermo in secondi
        time_stationary = self.no_move_steps * (1 / 30)  # Assume 30 FPS

        return {
            "paddle_pos": torch.tensor([self.paddle_pos], dtype=torch.float32, device=self.device),
            "objects": torch.tensor(objects_with_type_and_velocity, dtype=torch.float32, device=self.device),
            "score_beneficial": self.score_beneficial,
            "score_harmful": self.score_harmful,
            "lives": self.lives,
            "time_stationary": torch.tensor([time_stationary], dtype=torch.float32, device=self.device),  # Nuovo
        }

    def render(self, mode="human"):
        """
        Renderizza l'ambiente usando Pygame.
        """
        if not self.render_mode:
            return

        self.window.fill((0, 0, 0))

        # Disegna il paddle
        paddle_rect = pygame.Rect(
            self.paddle_pos,
            self.height - self.paddle_height,
            self.paddle_width,
            self.paddle_height,
        )
        pygame.draw.rect(self.window, (0, 255, 0), paddle_rect)

        # Disegna gli oggetti
        for obj in self.objects:
            color = (0, 255, 255) if obj["type"] == "beneficial" else (255, 0, 0)
            pygame.draw.rect(
                self.window,
                color,
                pygame.Rect(
                    obj["pos"][0],
                    obj["pos"][1],
                    self.object_size,
                    self.object_size,
                ),
            )

        # Mostra punteggio, vite e tempo
        font = pygame.font.SysFont("Arial", 24)
        score_text = font.render(f"Beneficial: {self.score_beneficial}  Harmful Avoided: {self.score_harmful}", True, (255, 255, 255))
        time_text = font.render(f"Time: {int(time.time() - self.start_time)}s  Lives: {self.lives}", True, (255, 255, 255))
        self.window.blit(score_text, (10, 10))
        self.window.blit(time_text, (self.width - 250, 10))

        pygame.display.flip()
        self.clock.tick(30)

    def _init_pygame(self):
        """
        Inizializza le risorse Pygame.
        """
        if self.render_mode:
            pygame.init()
            self.window = pygame.display.set_mode((self.width, self.height))
            self.clock = pygame.time.Clock()

    def close(self):
        """
        Chiude eventuali risorse di Pygame.
        """
        if self.window is not None:
            pygame.quit()
            self.window = None
            self.clock = None

    def handle_user_input(self):
        """
        Gestisce l'input dell'utente per il controllo manuale del paddle.
        """
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            self.paddle_pos = max(0, self.paddle_pos - self.paddle_speed)
        if keys[pygame.K_RIGHT]:
            self.paddle_pos = min(self.width - self.paddle_width, self.paddle_pos + self.paddle_speed)

# Test dell'ambiente
if __name__ == "__main__":
    env = CatcherEnv(render_mode=True, user_control=True, object_speed=7, paddle_speed=15, num_objects=5)
    obs = env.reset()
    done = False
    while not done:
        if env.user_control and env.render_mode:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print("Chiusura forzata.")
                    done = True
                    env.close()
            env.handle_user_input()
        obs, reward, done, truncated, info = env.step(1)  # Stazionamento
        #print(f"Reward: {reward}")
    env.close()
