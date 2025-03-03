import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from abc import ABC, abstractmethod

# Costanti per le azioni
LEFT = -1
STAY = 0
RIGHT = 1
TIME_LIMIT = 500

class CatchEnvBase(gym.Env, ABC):
    """
    Classe base astratta per l'ambiente Catch.
    Contiene funzionalità condivise e definisce metodi astratti per le differenze specifiche delle sottoclassi.
    """

    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(
        self,
        grid_size=10,
        num_objects=4,
        max_objects_in_state=2,   # Numero massimo di oggetti considerati nello stato
        min_row_gap=3,
        max_row_gap=6,
        spawn_probability=0.1,
        malicious_probability=0.4,  # Probabilità di spawnare una bomba
        min_speed=0.5,
        max_speed=1.5,
        speed_bins=3,               # Numero di bin per discretizzare la velocità (solo per osservazioni discrete)
        render_mode="human",
        basket_size=3
    ):
        super(CatchEnvBase, self).__init__()

        # Parametri di base
        self.grid_size = grid_size
        self.num_objects = num_objects
        self.min_row_gap = min_row_gap
        self.max_row_gap = max_row_gap
        self.spawn_probability = spawn_probability
        self.malicious_probability = malicious_probability
        self.render_mode = render_mode
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.speed_bins = speed_bins
        self.basket_size = basket_size

        # Carica le immagini personalizzate
        self.background_image = pygame.image.load("assets/background.jpg")
        self.fruit_image = pygame.image.load("assets/fruit.png")
        self.bomb_image = pygame.image.load("assets/bomb.png")
        
        # Ridimensiona le immagini per adattarsi alle celle della griglia
        self.cell_size = 500 // self.grid_size
        self.fruit_image = pygame.transform.smoothscale(
            self.fruit_image,
            (int(self.fruit_image.get_width() * (self.cell_size / self.fruit_image.get_height())), self.cell_size)
        )
        self.bomb_image = pygame.transform.smoothscale(
            self.bomb_image,
            (int(self.bomb_image.get_width() * (self.cell_size / self.bomb_image.get_height())), self.cell_size)
        )
        
        self.time_limit = TIME_LIMIT
        self.max_objects_in_state = max_objects_in_state

        # Numero massimo di bombe catturate prima di terminare l'episodio
        self.max_malicious_catches = 5
        self.malicious_catches = 0

        # Nuove variabili per informazioni extra
        self.lives = 5              # Vite iniziali
        self.missed_objects = 0     # Oggetti mancati
        self.caught_objects = 0     # Oggetti presi

        # Spazio azioni: 0 (LEFT), 1 (STAY), 2 (RIGHT)
        self.action_space = spaces.Discrete(3)

        # Passo di discretizzazione della velocità (solo per osservazioni discrete)
        self.speed_step = (self.max_speed - self.min_speed) / self.speed_bins

        # Definizione dello spazio di osservazione (da implementare nelle sottoclassi)
        self.observation_space = None  # Placeholder

        # Variabili dell'ambiente
        self.window = None
        self.clock = None

        # Liste per tracciare le proprietà degli oggetti
        self.fruit_rows = []
        self.fruit_cols = []
        self.fruit_speeds = []
        self.fruit_is_malicious = []

        # Posizione del cestino
        self.basket_pos = self.grid_size // 2

        # Offset per la discretizzazione delle righe (solo per osservazioni discrete)
        self.offset = max_row_gap + 1  # Assicura indicizzazione positiva
        self.max_row_after_offset = self.grid_size - 1 + self.offset

        # Inizializza lo spazio di osservazione
        self._initialize_observation_space()

    @abstractmethod
    def _initialize_observation_space(self):
        """
        Metodo astratto per inizializzare lo spazio di osservazione.
        Deve essere implementato dalle sottoclassi.
        """
        pass

    @abstractmethod
    def _get_observation(self):
        """
        Metodo astratto per ottenere l'osservazione corrente.
        Deve essere implementato dalle sottoclassi.
        """
        pass

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.time_limit = TIME_LIMIT

        # Resetta la posizione del cestino e i contatori
        self.basket_pos = self.grid_size // 2
        self.malicious_catches = 0

        # Resetta le nuove variabili di stato
        self.lives = 5
        self.missed_objects = 0
        self.caught_objects = 0

        # Svuota le liste degli oggetti
        self.fruit_rows.clear()
        self.fruit_cols.clear()
        self.fruit_speeds.clear()
        self.fruit_is_malicious.clear()

        return self._get_observation(), {}

    def step(self, action):
        self.time_limit -= 1

        # Applica l'azione
        move = {0: LEFT, 1: STAY, 2: RIGHT}.get(action, STAY)
        self.basket_pos = np.clip(self.basket_pos + move, 0, self.grid_size - 1)

        # Aggiorna la posizione degli oggetti
        self._update_objects()

        # Calcola la ricompensa e gestisce le catture/mancati
        reward = self._get_reward()

        # Controlla se l'episodio deve terminare (aggiunta la condizione sulle vite)
        done = False
        if self.time_limit <= 0 or self.malicious_catches >= self.max_malicious_catches or self.lives <= 0:
            done = True

        # Tenta di spawnare nuovi oggetti
        if not done:
            self._spawn_new_fruit()

        return self._get_observation(), reward, done, False, {}

    def _update_objects(self):
        """
        Aggiorna le posizioni di tutti gli oggetti in caduta.
        Può essere esteso dalle sottoclassi se necessario.
        """
        for i in range(len(self.fruit_rows)):
            self.fruit_rows[i] += self.fruit_speeds[i]
            # Le sottoclassi possono estendere questo metodo per includere ulteriori comportamenti

    def _get_reward(self):
        """
        Calcola la ricompensa basata sugli oggetti catturati o mancati.
        Aggiorna inoltre i contatori di vite, oggetti presi e mancati.
        """
        reward = 0
        to_remove = []

        for i in range(len(self.fruit_rows)):
            row = self.fruit_rows[i]
            col = self.fruit_cols[i]
            malicious = self.fruit_is_malicious[i]

            # Se l'oggetto raggiunge l'ultima riga
            if row >= (self.grid_size - 1):
                # Controlla se il cestino lo cattura
                if abs(round(col) - self.basket_pos) <= self.basket_size // 2:
                    # Oggetto catturato
                    if malicious:
                        reward -= 4
                        self.malicious_catches += 1
                        self.lives -= 1  # Penalità per aver catturato una bomba
                    else:
                        reward += 3
                        self.caught_objects += 1
                else:
                    # Oggetto mancato
                    if malicious:
                        reward += 0  # Nessuna penalità per bombe mancati
                    else:
                        reward -= 1
                        self.missed_objects += 1
                to_remove.append(i)

        # Rimuove gli oggetti processati
        for index in sorted(to_remove, reverse=True):
            del self.fruit_rows[index]
            del self.fruit_cols[index]
            del self.fruit_speeds[index]
            del self.fruit_is_malicious[index]

        return reward

    def _spawn_new_fruit(self):
        """
        Tenta di spawnare un nuovo oggetto con certe probabilità,
        se non si è superato il numero massimo di oggetti.
        """
        if len(self.fruit_rows) < self.num_objects and np.random.rand() < self.spawn_probability:
            new_row = self._generate_unique_row()
            if new_row is None:
                return

            new_col = self._generate_unique_column()
            if new_col is None:
                return

            speed = np.random.uniform(self.min_speed, self.max_speed)
            is_malicious = (np.random.rand() < self.malicious_probability)

            self.fruit_rows.append(new_row)
            self.fruit_cols.append(new_col)
            self.fruit_speeds.append(speed)
            self.fruit_is_malicious.append(is_malicious)

    def _generate_unique_row(self):
        """
        Genera una riga negativa in modo che l'oggetto parta sopra la griglia.
        Assicura che non sia troppo vicina ad altri oggetti già spawnati sopra la griglia.
        """
        max_tries = 100
        for _ in range(max_tries):
            candidate_row = -np.random.randint(self.min_row_gap, self.max_row_gap + 1)
            # Controlla che sia sufficientemente lontano dagli altri oggetti sopra la griglia
            too_close = False
            for r in self.fruit_rows:
                if r < 0 and abs(candidate_row - r) < self.min_row_gap:
                    too_close = True
                    break
            if not too_close:
                return candidate_row
        return None

    def _generate_unique_column(self):
        """
        Genera una colonna evitando collisioni con colonne già occupate.
        Se si desidera permettere più oggetti nella stessa colonna, rimuovere il controllo.
        """
        max_tries = 100
        for _ in range(max_tries):
            candidate_col = np.random.randint(0, self.grid_size)
            if candidate_col not in self.fruit_cols:
                return candidate_col
        return None

    def render(self, mode="human"):
        """
        Modifica UI:
          - Finestra allargata: scoreboard a sinistra, gioco (griglia) al centro/destra
          - I dati testuali (vite, presi, mancati, ecc.) sono ora elencati nella sezione di sinistra
        """
        # Dimensioni della finestra
        WINDOW_WIDTH = 700
        WINDOW_HEIGHT = 500
        # Larghezza della zona scoreboard
        SCOREBOARD_WIDTH = 200

        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
            pygame.display.set_caption("Catch Game")
            self.clock = pygame.time.Clock()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                self.window = None
                return

        # Disegna lo sfondo su tutta la finestra
        scaled_bg = pygame.transform.scale(self.background_image, (WINDOW_WIDTH, WINDOW_HEIGHT))
        self.window.blit(scaled_bg, (0, 0))

        # -- Disegno della “barra” scoreboard a sinistra --
        scoreboard_rect = pygame.Rect(0, 0, SCOREBOARD_WIDTH, WINDOW_HEIGHT)
        pygame.draw.rect(self.window, (20, 20, 20), scoreboard_rect)  # rettangolo scuro

        # Stampa delle info di gioco nel riquadro a sinistra
        font = pygame.font.Font(None, 28)
        info_lines = [
            f"Vite: {self.lives}",
            f"Presi: {self.caught_objects}",
            f"Mancati: {self.missed_objects}",
            f"Malicious: {self.malicious_catches}",
            f"Tempo: {self.time_limit}"
        ]
        # Partiamo dall'alto e lasciamo una piccola spaziatura fra le righe
        text_y = 30
        for line in info_lines:
            text_surf = font.render(line, True, (255, 255, 255))
            self.window.blit(text_surf, (20, text_y))
            text_y += 35

        # -- Zona di gioco: centriamo la griglia a destra partendo da SCOREBOARD_WIDTH --
        board_x_offset = SCOREBOARD_WIDTH
        board_y_offset = 0

        # Disegna gli oggetti (frutti/bombe)
        for row, col, malicious in zip(self.fruit_rows, self.fruit_cols, self.fruit_is_malicious):
            row_int = int(round(row))
            col_int = int(round(col))
            if 0 <= row_int < self.grid_size and 0 <= col_int < self.grid_size:
                fruit_pos = (board_x_offset + col_int * self.cell_size,
                             board_y_offset + row_int * self.cell_size)
                if malicious:
                    self.window.blit(self.bomb_image, fruit_pos)
                else:
                    self.window.blit(self.fruit_image, fruit_pos)

        # Disegna il cestino (rettangolo verde) nella zona di gioco
        basket_rect = pygame.Rect(
            board_x_offset + (self.basket_pos - self.basket_size // 2) * self.cell_size,
            board_y_offset + (self.grid_size - 1) * self.cell_size,
            self.basket_size * self.cell_size,
            self.cell_size
        )
        pygame.draw.rect(self.window, (0, 255, 0), basket_rect)

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        """
        Chiude la finestra di rendering.
        """
        if self.window is not None:
            pygame.quit()
            self.window = None
