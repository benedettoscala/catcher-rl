import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame

# Movements
LEFT = -1
STAY = 0
RIGHT = 1
TIME_LIMIT = 10000

class CatchEnv(gym.Env):
    """
    Gymnasium Environment for the Catch game with continuous falling objects.
    """

    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(self, grid_size=20, num_objects=5, min_row_gap=3, max_row_gap=6):
        super(CatchEnv, self).__init__()
        
        self.time_limit = TIME_LIMIT
        self.grid_size = grid_size
        self.num_objects = num_objects
        self.min_row_gap = min_row_gap
        self.max_row_gap = max_row_gap

        self.action_space = spaces.Discrete(3)  # 0 (LEFT), 1 (STAY), 2 (RIGHT)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(grid_size, grid_size), dtype=np.float32
        )

        self.state = None
        self.window = None
        self.clock = None
        self.spawn_probability = 0.1  # Random chance to spawn a new fruit each frame

        self.fruit_cols = []
        self.fruit_rows = []
        self.basket_pos = self.grid_size // 2

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.time_limit = TIME_LIMIT
        
        # Inizializziamo il basket al centro
        self.basket_pos = self.grid_size // 2
        
        # Svuotiamo le liste di frutti:
        self.fruit_cols = []
        self.fruit_rows = []

        # Ritorniamo l’osservazione iniziale
        return self.observe(), {}


    def step(self, action):
        self.time_limit -= 1
        
        # Se abbiamo finito il time limit, done
        if self.time_limit <= 0:
            return self.observe(), 0, True, False, {}

        # Aggiorniamo la posizione del basket e dei frutti
        self._update_state(action)

        # Calcoliamo ricompensa
        reward = self._get_reward()

        # Proviamo a spawnare nuovi frutti
        self._spawn_new_fruits()

        return self.observe(), reward, False, False, {}

    def _update_state(self, action):
        # Decodifica azione in movimento
        move = {0: LEFT, 1: STAY, 2: RIGHT}.get(action, STAY)
        self.basket_pos = min(max(1, self.basket_pos + move), self.grid_size - 2)

        # I frutti scendono di 1 riga
        self.fruit_rows = [row + 1 for row in self.fruit_rows]

    def _get_reward(self):
        reward = 0
        to_remove = []

        # Se un frutto arriva all'ultima riga
        for i in range(len(self.fruit_rows)):
            if self.fruit_rows[i] == self.grid_size - 1:
                # Controlla se il basket lo prende
                if abs(self.fruit_cols[i] - self.basket_pos) <= 1:
                    reward += 1
                else:
                    reward -= 1
                to_remove.append(i)

        # Qualunque frutto sia sceso OLTRE l'ultima riga, lo rimuoviamo comunque.
        # Se vuoi dare reward negativo in questi casi, fallo pure qui.
        for i in range(len(self.fruit_rows)):
            if self.fruit_rows[i] >= self.grid_size:
                # Questo frutto è "sparito" fuori dallo schermo
                to_remove.append(i)

        # Rimuoviamo i frutti processati
        for index in sorted(set(to_remove), reverse=True):
            del self.fruit_rows[index]
            del self.fruit_cols[index]

        return reward


    # =========================================================================
    #                       NUOVA LOGICA DI SPAWN
    # =========================================================================
    def _spawn_new_fruits(self):
        """
        Tenta di spawnare un frutto (al massimo uno per step) se:
        - Siamo sotto il numero massimo di frutti desiderato (self.num_objects)
        - self.spawn_probability lo consente
        - Troviamo una (row,col) valida rispettando il min_row_gap
        """
        if len(self.fruit_rows) < self.num_objects and np.random.rand() < self.spawn_probability:
            new_row = self._generate_unique_row()
            if new_row is None:
                # Non si è trovato uno slot valido: skip
                return

            new_col = self._generate_unique_column()
            if new_col is None:
                # Stesso discorso per la colonna (se vuoi limitare le collisioni in colonna)
                return

            # Se è tutto ok, aggiungiamo
            self.fruit_rows.append(new_row)
            self.fruit_cols.append(new_col)

    def _generate_unique_row(self):
        """
        Genera una row NEGATIVA, lontana almeno self.min_row_gap
        da ogni frutto già presente 'sopra' la griglia (row < 0).
        Ritorna None se non trova nulla dopo un certo numero di tentativi.
        """
        max_tries = 100
        for _ in range(max_tries):
            candidate_row = -np.random.randint(self.min_row_gap, self.max_row_gap+1)

            # Controlliamo la distanza solo tra frutti che NON sono ancora entrati nella griglia
            # (cioè row < 0). Se vuoi controllare anche quelli già nella griglia, puoi togliere `if r < 0`.
            if all(abs(candidate_row - r) >= self.min_row_gap for r in self.fruit_rows if r < 0):
                return candidate_row

        # Se non troviamo nulla di valido, restituiamo None
        return None

    def _generate_unique_column(self):
        """
        Genera una colonna. Se vuoi evitare che due frutti stiano
        sulla stessa colonna, controlla 'not in'.
        Se non ti interessa, puoi semplicemente restituire un random.
        """
        max_tries = 100
        for _ in range(max_tries):
            candidate_col = np.random.randint(0, self.grid_size)

            # Se vuoi EVITARE la stessa colonna, controlla così:
            if candidate_col not in self.fruit_cols:
                return candidate_col

            # Se invece vuoi PERMETTERE la stessa colonna,
            # commenta la riga sopra e fai direttamente return candidate_col.

        return None
    # =========================================================================

    def observe(self):
        canvas = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        for fruit_row, fruit_col in zip(self.fruit_rows, self.fruit_cols):
            if 0 <= fruit_row < self.grid_size:
                canvas[fruit_row, fruit_col] = 1
        # Basket: 3 celle verdi sulla riga finale (grid_size - 1)
        canvas[-1, self.basket_pos - 1 : self.basket_pos + 2] = 1
        return canvas

    def render(self, mode="human"):
        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode((500, 500))
            pygame.display.set_caption("Catch Game with Continuous Objects")
            self.clock = pygame.time.Clock()

        for event in pygame.event.get():
        # Se chiudono la finestra, facciamo una quit pulita
            if event.type == pygame.QUIT:
                pygame.quit()
                self.window = None
                return

        cell_size = 500 // self.grid_size

        self.window.fill((0, 0, 0))  # Black background
        for fruit_row, fruit_col in zip(self.fruit_rows, self.fruit_cols):
            if 0 <= fruit_row < self.grid_size:
                fruit_rect = pygame.Rect(
                    fruit_col * cell_size, fruit_row * cell_size, cell_size, cell_size
                )
                pygame.draw.ellipse(self.window, (255, 0, 0), fruit_rect)  # Red fruit

        basket_rect = pygame.Rect(
            (self.basket_pos - 1) * cell_size, (self.grid_size - 1) * cell_size,
            cell_size * 3, cell_size
        )
        pygame.draw.rect(self.window, (0, 255, 0), basket_rect)  # Green basket

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.window is not None:
            pygame.quit()
            self.window = None


if __name__ == "__main__":
    env = CatchEnv(grid_size=20)
    obs, info = env.reset()
    done = False
    while not done:
        env.render()
        # Muoviamo a caso per la demo
        obs, reward, done, truncated, info = env.step(env.action_space.sample())
    env.close()
