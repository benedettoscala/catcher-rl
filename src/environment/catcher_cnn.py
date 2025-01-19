import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame

LEFT = -1
STAY = 0
RIGHT = 1
TIME_LIMIT = 300

class CatchEnv(gym.Env):
    """
    Gymnasium Environment for the Catch game with variable falling speeds and malicious objects.
    Ora include un secondo canale nell'osservazione per mostrare la velocità 
    dell'oggetto in quella cella (se esiste).
    """

    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(
        self,
        grid_size=20,
        num_objects=6,
        min_row_gap=3,
        max_row_gap=6,
        spawn_probability=0.1,
        malicious_probability=0.4,  # Probabilità di spawnare un oggetto malevolo
        min_speed=0.5,
        max_speed=1.5
    ):
        super(CatchEnv, self).__init__()

        self.time_limit = TIME_LIMIT
        self.grid_size = grid_size
        self.num_objects = num_objects
        self.min_row_gap = min_row_gap
        self.max_row_gap = max_row_gap

        # Probabilità di spawnare un frutto (o bomba)
        self.spawn_probability = spawn_probability
        self.malicious_probability = malicious_probability

        # Range velocità oggetti
        self.min_speed = min_speed
        self.max_speed = max_speed

        # Conteggio delle bombe prese
        self.malicious_catches = 0
        self.max_malicious_catches = 5

        # Definizione spazi Gym
        self.action_space = spaces.Discrete(3)  # 0 (LEFT), 1 (STAY), 2 (RIGHT)
        
        # =========================
        # MODIFICA OBSERVATION SPACE
        # =========================
        # Ora l'osservazione sarà un tensore [2, grid_size, grid_size]:
        #    - Primo canale (0): tipo di cella (vuota=0, fruit=1, bomb=2, basket=3)
        #    - Secondo canale (1): velocità (0 se non c'è oggetto, altrimenti speed).
        self.observation_space = spaces.Box(
            low=0.0,
            high=max_speed,                     # La velocità non dovrebbe superare max_speed
            shape=(2, grid_size, grid_size),
            dtype=np.float32
        )

        # Stato
        self.state = None
        self.window = None
        self.clock = None

        # Liste per tracciare caratteristiche dei frutti/bombe
        self.fruit_cols = []
        self.fruit_rows = []
        self.fruit_speeds = []         # Velocità verticale di caduta
        self.fruit_is_malicious = []   # Indica se l'oggetto è malevolo (bomba)

        # Posizione del basket
        self.basket_pos = self.grid_size // 2


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.time_limit = TIME_LIMIT
        
        # Reset basket e conteggi
        self.basket_pos = self.grid_size // 2
        self.malicious_catches = 0

        # Svuotiamo le liste degli oggetti
        self.fruit_cols.clear()
        self.fruit_rows.clear()
        self.fruit_speeds.clear()
        self.fruit_is_malicious.clear()

        # Ritorniamo l’osservazione iniziale
        return self.observe(), {}


    def step(self, action):
        self.time_limit -= 1

        # Se abbiamo finito il time limit, done
        if self.time_limit <= 0:
            return self.observe(), 0, True, False, {}

        # Aggiorniamo la posizione del basket e lo stato
        self._update_state(action)

        # Calcoliamo ricompensa
        reward = self._get_reward()

        # Controllo se abbiamo raccolto troppi oggetti malevoli
        #if self.malicious_catches >= self.max_malicious_catches:
            # Hai preso troppe bombe: si perde.
            #return self.observe(), reward, True, False, {}

        # Proviamo a spawnare nuovi frutti/bombe
        self._spawn_new_fruits()

        return self.observe(), reward, False, False, {}


    def _update_state(self, action):
        # Decodifica azione in movimento
        move = {0: LEFT, 1: STAY, 2: RIGHT}.get(action, STAY)
        self.basket_pos = min(max(1, self.basket_pos + move), self.grid_size - 2)

        # Aggiorniamo la posizione di ogni frutto/bomba in base alla sua velocità
        for i in range(len(self.fruit_rows)):
            self.fruit_rows[i] += self.fruit_speeds[i]


    def _get_reward(self):
        """
        Calcola il reward considerando frutti e bombe catturate.
        - +3 se prendi un frutto non malevolo
        - -4 se prendi un frutto malevolo
        - -1 se un frutto non malevolo cade oltre l'ultima riga (mancato)
        -  0 se un frutto malevolo cade oltre l'ultima riga (mancato)
        """
        reward = 0
        to_remove = []

        for i in range(len(self.fruit_rows)):
            # Se un oggetto arriva all'ultima riga (o la supera)
            if self.fruit_rows[i] >= self.grid_size - 1:
                # Verifichiamo se il basket lo prende
                caught = (abs(self.fruit_cols[i] - self.basket_pos) <= 1)
                if caught:
                    # Controlla se è malevolo
                    if self.fruit_is_malicious[i]:
                        reward -= 4
                        self.malicious_catches += 1
                    else:
                        reward += 3
                    to_remove.append(i)
                else:
                    # Oggetto perso
                    if self.fruit_is_malicious[i]:
                        reward -= 0
                    else:
                        reward -= 1
                    to_remove.append(i)

        # Rimuoviamo i frutti/bombe processati
        for index in sorted(set(to_remove), reverse=True):
            del self.fruit_rows[index]
            del self.fruit_cols[index]
            del self.fruit_speeds[index]
            del self.fruit_is_malicious[index]

        return reward


    # =========================================================================
    #                          LOGICA DI SPAWN
    # =========================================================================
    def _spawn_new_fruits(self):
        """
        Tenta di spawnare un oggetto (max uno per step) se:
        - Siamo sotto il numero massimo di oggetti desiderato (self.num_objects)
        - self.spawn_probability lo consente
        - Troviamo una (row,col) valida rispettando il min_row_gap
        """
        if len(self.fruit_rows) < self.num_objects and np.random.rand() < self.spawn_probability:
            new_row = self._generate_unique_row()
            if new_row is None:
                return

            new_col = self._generate_unique_column()
            if new_col is None:
                return

            # Generiamo la velocità e decidiamo se è malevolo
            speed = np.random.uniform(self.min_speed, self.max_speed)
            is_malicious = (np.random.rand() < self.malicious_probability)

            self.fruit_rows.append(new_row)
            self.fruit_cols.append(new_col)
            self.fruit_speeds.append(speed)
            self.fruit_is_malicious.append(is_malicious)


    def _generate_unique_row(self):
        """
        Genera una row NEGATIVA, lontana almeno self.min_row_gap
        da ogni frutto già presente 'sopra' la griglia (row < 0).
        Ritorna None se non trova nulla dopo un certo numero di tentativi.
        """
        max_tries = 100
        for _ in range(max_tries):
            candidate_row = -np.random.randint(self.min_row_gap, self.max_row_gap + 1)
            if all(abs(candidate_row - r) >= self.min_row_gap for r in self.fruit_rows if r < 0):
                return candidate_row
        return None


    def _generate_unique_column(self):
        """
        Genera una colonna evitando di collidere con le colonne già occupate.
        Se vuoi permettere più frutti sulla stessa colonna, rimuovi il check.
        """
        max_tries = 100
        for _ in range(max_tries):
            candidate_col = np.random.randint(0, self.grid_size)
            if candidate_col not in self.fruit_cols:
                return candidate_col
        return None
    # =========================================================================


    def observe(self):
        """
        Restituisce una matrice di dimensione (2, grid_size, grid_size).
        
        Primo canale (0): tipo di cella
             0: cella vuota
             1: frutto (benevolo)
             2: bomba (malevolo)
             3: cestino (basket)
        
        Secondo canale (1): velocità dell'oggetto in quella cella
             0 se non c'è oggetto
             >0 se c'è frutto/bomba (velocità in [min_speed, max_speed])
        """
        # Inizializziamo 2 canali: uno per il "tipo" e uno per la "velocità"
        canvas = np.zeros((2, self.grid_size, self.grid_size), dtype=np.float32)

        # Riempio il primo canale (tipo) e il secondo (velocità) per ogni oggetto
        for fruit_row, fruit_col, is_malicious, speed in zip(self.fruit_rows, 
                                                             self.fruit_cols, 
                                                             self.fruit_is_malicious,
                                                             self.fruit_speeds):
            row_int = int(round(fruit_row))
            if 0 <= row_int < self.grid_size:
                canvas[0, row_int, fruit_col] = 2.0 if is_malicious else 1.0
                canvas[1, row_int, fruit_col] = speed  # Salvo la velocità

        # Disegno il basket: 3 celle
        basket_row = self.grid_size - 1
        canvas[0, basket_row, self.basket_pos - 1 : self.basket_pos + 2] = 3.0
        # Nel secondo canale, il cestino non ha velocità, lascio 0

        return canvas


    def render(self, mode="human"):
        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode((500, 500))
            pygame.display.set_caption("Catch Game with Continuous Objects and Malicious Items")
            self.clock = pygame.time.Clock()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                self.window = None
                return

        cell_size = 500 // self.grid_size
        self.window.fill((0, 0, 0))  # Sfondo nero

        # Disegniamo i frutti/bombe
        for row, col, malicious in zip(self.fruit_rows, self.fruit_cols, self.fruit_is_malicious):
            row_int = int(round(row))
            if 0 <= row_int < self.grid_size:
                fruit_rect = pygame.Rect(col * cell_size, row_int * cell_size, cell_size, cell_size)
                color = (255, 0, 0) if malicious else (255, 255, 0)
                pygame.draw.ellipse(self.window, color, fruit_rect)

        # Disegniamo il basket (3 celle)
        basket_rect = pygame.Rect(
            (self.basket_pos - 1) * cell_size,
            (self.grid_size - 1) * cell_size,
            cell_size * 3, cell_size
        )
        pygame.draw.rect(self.window, (0, 255, 0), basket_rect)

         # disegno il punteggio, il tempo e le vite in alto a destra
        font = pygame.font.Font(None, 36)
        text = font.render(f"Malicious catches: {self.malicious_catches}", True, (255, 255, 255))
        self.window.blit(text, (10, 10))

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])


    def close(self):
        if self.window is not None:
            pygame.quit()
            self.window = None
