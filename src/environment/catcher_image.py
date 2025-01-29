import numpy as np
from gymnasium import spaces
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from environment.core.catch_base import CatchEnvBase


class CatchEnvImage(CatchEnvBase):
    """
    Ambiente Catch con spazio di osservazione continuo (Box).
    Adatto per algoritmi basati su reti neurali convoluzionali (CNN).
    """

    def __init__(self, in_channels=2, **kwargs):
        self.in_channels = in_channels
        super(CatchEnvImage, self).__init__(**kwargs)

    def _initialize_observation_space(self):
        """
        Definisce lo spazio di osservazione come Box per input adatti a CNN.
        """
        # Definisce lo spazio Box con 2 canali:
        # 1) Tipo di cella: vuoto=0, frutto=1, bomba=2, basket=3
        # 2) Velocità verticale: 0 se nessun oggetto, altrimenti velocità [min_speed, max_speed]

        self.observation_space = spaces.Box(
            low=0.0,
            high=self.max_speed,  # La velocità non dovrebbe superare max_speed
            shape=(2, self.grid_size, self.grid_size),
            dtype=np.float32
        )

    def _get_observation(self):
        """
        Restituisce una matrice di dimensione (2, grid_size, grid_size).
        Primo canale: tipo di cella (0: vuoto, 1: frutto, 2: bomba, 3: basket)
        Secondo canale: velocità verticale dell'oggetto in quella cella (0 se nessun oggetto, altrimenti speed)
        """
        # Inizializza 2 canali: tipo e velocità verticale
        canvas = np.zeros((2, self.grid_size, self.grid_size), dtype=np.float32)

        # Riempie il primo canale (tipo) e il secondo canale (velocità verticale) per ogni oggetto
        for fruit_row, fruit_col, is_malicious, speed in zip(
            self.fruit_rows,
            self.fruit_cols,
            self.fruit_is_malicious,
            self.fruit_speeds
        ):
            row_int = int(round(fruit_row))
            col_int = int(round(fruit_col))
            if 0 <= row_int < self.grid_size and 0 <= col_int < self.grid_size:
                canvas[0, row_int, col_int] = 2.0 if is_malicious else 1.0
                canvas[1, row_int, col_int] = speed  # Salva la velocità verticale

        # Disegna il cestino: 3 celle
        basket_row = self.grid_size - 1
        basket_left = int(round(self.basket_pos - 1))
        basket_right = int(round(self.basket_pos + 1))
        basket_left = max(0, basket_left)
        basket_right = min(self.grid_size - 1, basket_right)
        canvas[0, basket_row, basket_left : basket_right + 1] = 3.0
        # Nel secondo canale, il cestino non ha velocità, lascio 0

        return canvas


class CatchEnvImageChangeDirection(CatchEnvImage):
    """
    Ambiente Catch con spazio di osservazione continuo e movimento orizzontale degli oggetti.
    Adatto per algoritmi basati su reti neurali convoluzionali (CNN).
    """

    def __init__(
        self,
        min_h_speed=-0.5,  # Velocità orizzontale minima
        max_h_speed=0.5,   # Velocità orizzontale massima
        in_channels=3,
        **kwargs
    ):
        self.in_channels = in_channels
        self.min_h_speed = min_h_speed
        self.max_h_speed = max_h_speed
        self.fruit_h_speeds = []
        self.fruit_direction_changed = []  # Lista per tracciare i cambi di direzione

        super(CatchEnvImageChangeDirection, self).__init__(**kwargs, in_channels=self.in_channels)

    def _initialize_observation_space(self):
        """
        Definisce lo spazio di osservazione come Box per input adatti a CNN, includendo un terzo canale per i cambi di direzione.
        """
        # Definisce lo spazio Box con 3 canali:
        # 1) Tipo di cella
        # 2) Velocità verticale
        # 3) Cambiamento di direzione (0: no, 1: sì)

        self.observation_space = spaces.Box(
            low=0.0,
            high=np.maximum(self.max_speed, 1.0),  # Assicurati che il terzo canale possa essere 1
            shape=(3, self.grid_size, self.grid_size),
            dtype=np.float32
        )

    def _spawn_new_fruit(self):
        """
        Tenta di spawnare un nuovo oggetto includendo la velocità orizzontale e inizializza il flag di cambiamento di direzione.
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

            # Assegna una velocità orizzontale
            h_speed = np.random.uniform(self.min_h_speed, self.max_h_speed)

            self.fruit_rows.append(new_row)
            self.fruit_cols.append(new_col)
            self.fruit_speeds.append(speed)
            self.fruit_is_malicious.append(is_malicious)
            self.fruit_h_speeds.append(h_speed)
            self.fruit_direction_changed.append(0)  # Inizialmente, nessun cambiamento di direzione

    def _update_objects(self):
        """
        Aggiorna le posizioni verticali e orizzontali degli oggetti, tracciando i cambi di direzione orizzontale.
        """
        for i in range(len(self.fruit_rows)):
            # Aggiorna la posizione verticale
            self.fruit_rows[i] += self.fruit_speeds[i]

            # Aggiorna la posizione orizzontale
            old_h_speed = self.fruit_h_speeds[i]
            self.fruit_cols[i] += self.fruit_h_speeds[i]

            # Reset del flag di cambiamento di direzione
            self.fruit_direction_changed[i] = 0

            # Gestisci le collisioni con i bordi orizzontali
            if self.fruit_cols[i] < 0:
                self.fruit_cols[i] = 0
                self.fruit_h_speeds[i] *= -1  # Inverte direzione
                self.fruit_direction_changed[i] = 1  # Ha cambiato direzione
            elif self.fruit_cols[i] > self.grid_size - 1:
                self.fruit_cols[i] = self.grid_size - 1
                self.fruit_h_speeds[i] *= -1  # Inverte direzione
                self.fruit_direction_changed[i] = 1  # Ha cambiato direzione

            # Cambia casualmente la direzione orizzontale con una certa probabilità
            change_dir_prob = 0.05  # 5% di probabilità
            if np.random.rand() < change_dir_prob:
                #value can be -0.1 ,-0.05, -0.02 , 0 ,0.02, 0.05, 0.1
                self.fruit_h_speeds[i] = np.random.choice([-0.1, -0.05, -0.02, 0, 0.02, 0.05, 0.1])
                # Limita la velocità orizzontale ai limiti specificati
                self.fruit_h_speeds[i] = np.clip(
                    self.fruit_h_speeds[i],
                    self.min_h_speed,
                    self.max_h_speed
                )
                if self.fruit_h_speeds[i] != old_h_speed:
                    self.fruit_direction_changed[i] = 1  # Ha cambiato direzione

    def _get_observation(self):
        """
        Restituisce una matrice di dimensione (3, grid_size, grid_size).
        Primo canale: tipo di cella (0: vuoto, 1: frutto, 2: bomba, 3: basket)
        Secondo canale: velocità verticale dell'oggetto in quella cella (0 se nessun oggetto, altrimenti speed)
        Terzo canale: cambiamento di direzione (0: no, 1: sì)
        """
        # Inizializza 3 canali: tipo, velocità verticale e cambiamento di direzione
        canvas = np.zeros((3, self.grid_size, self.grid_size), dtype=np.float32)

        # Riempie i canali per ogni oggetto
        for fruit_row, fruit_col, is_malicious, speed, h_speed in zip(
            self.fruit_rows,
            self.fruit_cols,
            self.fruit_is_malicious,
            self.fruit_speeds,
            self.fruit_h_speeds,
        ):
            row_int = int(round(fruit_row))
            col_int = int(round(fruit_col))
            if 0 <= row_int < self.grid_size and 0 <= col_int < self.grid_size:
                # Tipo di cella
                canvas[0, row_int, col_int] = 2.0 if is_malicious else 1.0
                # Velocità verticale
                canvas[1, row_int, col_int] = speed
                # velcità orizzontale
                canvas[2, row_int, col_int] = h_speed

        # Disegna il cestino: 3 celle
        basket_row = self.grid_size - 1
        basket_left = int(round(self.basket_pos - 1))
        basket_right = int(round(self.basket_pos + 1))
        basket_left = max(0, basket_left)
        basket_right = min(self.grid_size - 1, basket_right)
        canvas[0, basket_row, basket_left : basket_right + 1] = 3.0
        # Nel secondo e terzo canale, il cestino non ha velocità né cambiamento di direzione

        return canvas


# Test per vedere se l'ambiente funziona renderizzato per umano
if __name__ == "__main__":
    env = CatchEnvImageChangeDirection(grid_size=15, render_mode="human")
    env.reset()
    done = False
    while not done:
        env.render()
        _, _, done, _, _ = env.step(2)  # Esempio di azione fissa
    env.close()
