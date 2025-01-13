import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random
import pygame
import time

class CatcherEnv(gym.Env):
    """
    Ambiente per il gioco Catcher con supporto per rendering tramite Pygame e controllo utente.
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

        # Osservazioni: posizione del paddle, posizione degli oggetti (x, y), punteggio, vite
        self.observation_space = spaces.Dict({
            "paddle_pos": spaces.Box(low=0, high=self.width, shape=(1,), dtype=np.float32),
            "objects": spaces.Box(low=0, high=self.width, shape=(self.num_objects, 2), dtype=np.float32),
            "score": spaces.Discrete(1000),
            "lives": spaces.Discrete(5),
        })

        # Variabili Pygame
        self.window = None
        self.clock = None
        self.paddle_width = 100
        self.paddle_height = 20
        self.object_size = 30

        # Sprite
        self.paddle_sprite = None
        self.object_sprite_beneficial = None
        self.object_sprite_harmful = None

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

        if self.render_mode:
            self._init_pygame()

        return self._get_obs()

    def step(self, action):
        """
        Esegue un passo nell'ambiente dato un'azione.
        
        Args:
            action (int): 0 -> sinistra, 1 -> stazionamento, 2 -> destra
        
        Returns:
            obs (dict): Lo stato attuale dell'ambiente
            reward (float): Ricompensa ottenuta
            done (bool): Se l'episodio è terminato
            info (dict): Informazioni addizionali
        """
        if not self.user_control:
            if action == 0:  # Sinistra
                self.paddle_pos = max(0, self.paddle_pos - self.paddle_speed)
            elif action == 2:  # Destra
                self.paddle_pos = min(self.width - self.paddle_width, self.paddle_pos + self.paddle_speed)

        reward = 0

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

        if self.render_mode:
            self.render()

        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        """
        Restituisce lo stato osservabile dell'ambiente.
        """
        return {
            "paddle_pos": np.array([self.paddle_pos], dtype=np.float32),
            "objects": np.array([obj["pos"] for obj in self.objects], dtype=np.float32),
            "score_beneficial": self.score_beneficial,
            "score_harmful": self.score_harmful,
            "lives": self.lives,
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
                    done = True
            env.handle_user_input()
        obs, reward, done, info = env.step(1)  # Stazionamento
        print(f"Reward: {reward}")
    env.close()
