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
    Gymnasium Environment per il gioco "Catch" con velocità variabili e oggetti malevoli (bombe).
    Adatto all'uso con un algoritmo SARSA/tabulare grazie allo spazio di osservazione discreto.
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
        speed_bins=3,               # Numero di bin per discretizzare la velocità
        render_mode="human",
        basket_size=3
    ):
        super(CatchEnv, self).__init__()
        
        # Parametri base
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
        
        self.time_limit = TIME_LIMIT
        
        self.max_objects_in_state = max_objects_in_state

        # Massimo numero di bombe prese prima di concludere l'episodio
        self.max_malicious_catches = 5
        self.malicious_catches = 0
        
        # Azioni: 0 (LEFT), 1 (STAY), 2 (RIGHT)
        self.action_space = spaces.Discrete(3)
        
        # -------------- DISCRETIZZAZIONE VELOCITÀ --------------
        # suddividiamo [min_speed, max_speed] in speed_bins intervalli
        self.speed_step = (self.max_speed - self.min_speed) / self.speed_bins
        
        # -------------- DEFINIZIONE SPAZIO OSSERVAZIONE --------------
        # Lo stato è un vettore con:
        #   1) basket_pos: [0..(grid_size-1)]
        #   2) Per max_objects_in_state oggetti: (row, col, type, speed_bin)
        #      - row: in teoria può essere < 0 (sopra la griglia).
        #        Per semplificare, trasformiamo row "reale" in row discreta "positiva":
        #            row_discreta = row_int + OFFSET
        #        dove OFFSET = max_row_gap (o un valore che assicuri > 0).
        #        Oppure, fissiamo un limite massimo di row_negativa e scartiamo oggetti troppo in alto.
        #
        #   - col: [0..(grid_size-1)]
        #   - type: {0: frutto, 1: bomba}
        #   - speed_bin: [0..(speed_bins-1)]
        #
        # Se un oggetto non esiste, mettiamo row = grid_size (o un valore sentinel), col=0, type=0, speed_bin=0
        #
        # Esempio dimensione:
        #   [grid_size] + [grid_size+some_offset, grid_size, 2, speed_bins] * max_objects_in_state
        
        # Definiamo un OFFSET in modo da spostare le row negative
        # Se un oggetto fosse a row = -6, lo mappiamo in uno "0" se offset=6
        # e se row = 9, lo mappiamo in 9+6 = 15
        self.offset = 10  # offset per righe negative (scegli un valore >= max_row_gap)
        # Limite massimo per la row dopo l'offset
        max_row_after_offset = self.grid_size - 1 + self.offset
        
        # Spazio MultiDiscrete
        # 1) basket_pos: range [0, grid_size-1]
        # 2) per ogni oggetto:
        #    row: [0, max_row_after_offset]
        #    col: [0, grid_size-1]
        #    type: [0, 1]
        #    speed_bin: [0, speed_bins-1]
        low_high_per_object = [
            (max_row_after_offset + 1),  # row
            self.grid_size,             # col
            2,                          # type (0 o 1)
            self.speed_bins             # speed_bin
        ]
        
        # Componiamo la lista MultiDiscrete:
        # - primo elemento: basket_pos
        # - poi i 4 parametri per ciascun oggetto
        self.observation_space = spaces.MultiDiscrete(
            [self.grid_size] + low_high_per_object * self.max_objects_in_state
        )

        # -------------- VARIABILI DELL'AMBIENTE --------------
        self.window = None
        self.clock = None

        # Liste per tracciare caratteristiche dei frutti/bombe
        self.fruit_rows = []
        self.fruit_cols = []
        self.fruit_speeds = []
        self.fruit_is_malicious = []

        # Posizione del cestino
        self.basket_pos = self.grid_size // 2

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.time_limit = TIME_LIMIT

        # Reset basket e conteggi
        self.basket_pos = self.grid_size // 2
        self.malicious_catches = 0

        # Svuota le liste degli oggetti
        self.fruit_rows.clear()
        self.fruit_cols.clear()
        self.fruit_speeds.clear()
        self.fruit_is_malicious.clear()

        return self._get_observation(), {}

    def step(self, action):
        self.time_limit -= 1
        
        # Applica azione
        move = {0: LEFT, 1: STAY, 2: RIGHT}[action]
        self.basket_pos = np.clip(self.basket_pos + move, 0, self.grid_size - 1)
        
        # Aggiorna la posizione degli oggetti
        for i in range(len(self.fruit_rows)):
            self.fruit_rows[i] += self.fruit_speeds[i]

        # Calcola la ricompensa e rimuovi oggetti se necessario
        reward = self._get_reward()

        # Controllo fine episodio
        done = False
        # 1) se finisce il time limit
        if self.time_limit <= 0:
            done = True
        # 2) se ho preso troppe bombe
        if self.malicious_catches >= self.max_malicious_catches:
            done = True

        # Prova a spawnare nuovi oggetti
        if not done:
            self._spawn_new_fruit()

        return self._get_observation(), reward, done, False, {}

    def _get_reward(self):
        """
        Calcola il reward considerando frutti e bombe catturate.
        - +3 se prendi un frutto non malevolo
        - -4 se prendi una bomba
        - -1 se un frutto cade oltre l'ultima riga (mancato)
        -  0 se una bomba cade oltre l'ultima riga (mancata)
        """
        reward = 0
        to_remove = []

        for i in range(len(self.fruit_rows)):
            row = self.fruit_rows[i]
            col = self.fruit_cols[i]
            malicious = self.fruit_is_malicious[i]

            # Se l'oggetto raggiunge (o supera) l'ultima riga della griglia
            if row >= (self.grid_size - 1):
                # Controlliamo se il cestino lo prende
                # (distanza in colonna <= 1)
                if abs(round(col) - self.basket_pos) <= 1:
                    # Oggetto catturato
                    if malicious:
                        reward -= 4
                        
                        self.malicious_catches += 1
                    else:
                        reward += 3
                else:
                    # Oggetto perso
                    if malicious:
                        reward += 0
                    else:
                        reward -= 1
                to_remove.append(i)

        # Rimuove gli oggetti "consumati"
        for index in sorted(to_remove, reverse=True):
            del self.fruit_rows[index]
            del self.fruit_cols[index]
            del self.fruit_speeds[index]
            del self.fruit_is_malicious[index]

        return reward

    def _spawn_new_fruit(self):
        """
        Tenta di spawnare un nuovo oggetto con certe probabilità, 
        se non abbiamo superato il numero max di oggetti.
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
        Assicuriamoci che non sia troppo vicina ad altri oggetti già spawnati (sopra la griglia).
        """
        max_tries = 100
        for _ in range(max_tries):
            candidate_row = -np.random.randint(self.min_row_gap, self.max_row_gap+1)
            # Controlla che rispetti la distanza minima
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
        Genera una colonna evitando collisioni con le colonne già occupate (facoltativo).
        Se vuoi permettere più frutti sulla stessa colonna, rimuovi il check.
        """
        max_tries = 100
        for _ in range(max_tries):
            candidate_col = np.random.randint(0, self.grid_size)
            if candidate_col not in self.fruit_cols:
                return candidate_col
        return None

    def _get_observation(self):
        """
        Restituisce uno stato discreto così composto:
          [ basket_pos,  (row, col, type, speed_bin) x max_objects_in_state ]

        - Se ci sono meno oggetti, riempiamo con valori "sentinel".
        """
        # 1) Posizione del basket
        obs = [int(self.basket_pos)]
        
        # 2) Ordiniamo gli oggetti (ad es. dal più "basso" al più "alto") 
        #    o in base a quanto sono vicini all’ultima riga, in modo da prendere
        #    i max_objects_in_state più "importanti".
        if len(self.fruit_rows) > 0:
            indices_sorted = sorted(
                range(len(self.fruit_rows)),
                key=lambda i: self.fruit_rows[i],
                reverse=True
            )
        else:
            indices_sorted = []
        
        # Prendiamo i primi max_objects_in_state
        indices_sorted = indices_sorted[:self.max_objects_in_state]

        # 3) Costruiamo la parte di stato per ogni oggetto
        for i in indices_sorted:
            row_float = self.fruit_rows[i]
            col_float = self.fruit_cols[i]
            malicious = self.fruit_is_malicious[i]
            speed = self.fruit_speeds[i]

            # (a) Discretizziamo la riga: offsettiamo per renderla >= 0
            # row_neg può essere es. -5, con offset 10 diventa 5
            row_discrete = int(round(row_float)) + self.offset
            # tagliamo a [0, max_row_after_offset]
            row_discrete = np.clip(row_discrete, 0, self.grid_size - 1 + self.offset)

            # (b) Discretizziamo la colonna
            col_discrete = int(round(col_float))
            col_discrete = np.clip(col_discrete, 0, self.grid_size - 1)

            # (c) Tipo (0 frutto, 1 bomba)
            obj_type = 1 if malicious else 0

            # (d) Speed bin
            speed_bin = int((speed - self.min_speed) // self.speed_step)
            speed_bin = np.clip(speed_bin, 0, self.speed_bins - 1)

            obs.extend([row_discrete, col_discrete, obj_type, speed_bin])

        # 4) Se abbiamo meno oggetti di max_objects_in_state, riempiamo con default
        #    Mettiamo una row >= grid_size + offset per segnalare "oggetto inesistente" 
        for _ in range(self.max_objects_in_state - len(indices_sorted)):
            # Esempio: row = grid_size + offset = "fuori dal mondo"
            row_dummy = self.grid_size - 1 + self.offset
            col_dummy = 0
            type_dummy = 0
            speed_dummy = 0
            obs.extend([row_dummy, col_dummy, type_dummy, speed_dummy])

        return np.array(obs, dtype=int)

    # ----------------- RENDER -----------------
    def render(self, mode="human"):
        if mode != "human":
            raise ValueError(f"Unsupported render mode: {mode}. Supported mode is 'human'.")

        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode((500, 500))
            pygame.display.set_caption("Catch Game - Discrete State Env")
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
                fruit_rect = pygame.Rect(
                    col * cell_size,
                    row_int * cell_size,
                    cell_size, cell_size
                )
                color = (255, 0, 0) if malicious else (255, 255, 0)  # Rosso per bomba, giallo per frutto
                pygame.draw.ellipse(self.window, color, fruit_rect)

        # Disegniamo il basket (larghezza 1 cella o 3 celle, a tua scelta)
        # Qui lo facciamo di 3 per restare simili all'originale
        x_left = (self.basket_pos - 1) * cell_size
        x_left = max(0, x_left)  # Evito di uscire a sinistra
        basket_rect = pygame.Rect(
            x_left,
            (self.grid_size - 1) * cell_size,
            cell_size * self.basket_size,
            cell_size
        )
        pygame.draw.rect(self.window, (0, 255, 0), basket_rect)

        # Mostra info
        font = pygame.font.Font(None, 24)
        text_surf = font.render(
            f"Malicious catches: {self.malicious_catches} | Time left: {self.time_limit}",
            True,
            (255, 255, 255)
        )
        self.window.blit(text_surf, (10, 10))

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

        return


    def close(self):
        if self.window is not None:
            pygame.quit()
            self.window = None
