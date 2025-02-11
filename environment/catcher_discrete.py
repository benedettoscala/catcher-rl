import numpy as np
from gymnasium import spaces
import sys, os
import pygame

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from environment.core.catch_base import CatchEnvBase


class CatchEnv(CatchEnvBase):
    """
    Ambiente Catch con spazio di osservazione discreto.
    Adatto per algoritmi tabulari come SARSA.
    """
    
    def __init__(self,direction = False, **kwargs):

        self.direction = direction
        super(CatchEnv, self).__init__(**kwargs)
    
    def _initialize_observation_space(self):
        """
        Definisce lo spazio di osservazione come MultiDiscrete per rappresentazioni discrete dello stato.
        """
        # Definisce lo spazio MultiDiscrete
        # 1) basket_pos: [0, grid_size-1]
        # 2) Per ogni oggetto:
        #    row: [0, max_row_after_offset]
        #    col: [0, grid_size-1]
        #    type: [0, 1]
        #    speed_bin: [0, speed_bins-1]
        
        low_high_per_object = [
            self.max_row_after_offset + 1,  # row
            self.grid_size,                # col
            2,                             # type (0 o 1)
            self.speed_bins                # speed_bin
        ]
        
        # Combina in una singola lista
        observation_space = [self.grid_size] + [dim for obj in range(self.max_objects_in_state) for dim in low_high_per_object]
        
        self.observation_space = spaces.MultiDiscrete(observation_space)
    
    def _get_observation(self):
        """
        Restituisce uno stato discreto composto da:
          [ basket_pos, (row, col, type, speed_bin) x max_objects_in_state ]
        """
        # 1) Posizione del cestino
        obs = [int(self.basket_pos)]
        
        # 2) Ordina gli oggetti per vicinanza al fondo (righe decrescenti)
        if len(self.fruit_rows) > 0:
            indices_sorted = sorted(
                range(len(self.fruit_rows)),
                key=lambda i: self.fruit_rows[i],
                reverse=True
            )
        else:
            indices_sorted = []
        
        # Prendi i primi max_objects_in_state oggetti
        indices_sorted = indices_sorted[:self.max_objects_in_state]
        
        # 3) Costruisci la parte dello stato per ogni oggetto
        for i in indices_sorted:
            row_float = self.fruit_rows[i]
            col_float = self.fruit_cols[i]
            malicious = self.fruit_is_malicious[i]
            speed = self.fruit_speeds[i]
            
            # (a) Discretizza la riga: offset per renderla positiva
            row_discrete = int(round(row_float)) + self.offset
            row_discrete = np.clip(row_discrete, 0, self.max_row_after_offset)
            
            # (b) Discretizza la colonna
            col_discrete = int(round(col_float))
            col_discrete = np.clip(col_discrete, 0, self.grid_size - 1)
            
            # (c) Tipo: 1 se malevolo, altrimenti 0
            obj_type = 1 if malicious else 0
            
            # (d) Speed bin
            speed_bin = int((speed - self.min_speed) // self.speed_step)
            speed_bin = np.clip(speed_bin, 0, self.speed_bins - 1)
            
            obs.extend([row_discrete, col_discrete, obj_type, speed_bin])
        
        # 4) Se ci sono meno oggetti, riempi con valori sentinel
        for _ in range(self.max_objects_in_state - len(indices_sorted)):
            # Esempio: row = grid_size - 1 + offset indica "oggetto inesistente"
            row_dummy = self.grid_size - 1 + self.offset
            col_dummy = 0
            type_dummy = 0
            speed_dummy = 0
            obs.extend([row_dummy, col_dummy, type_dummy, speed_dummy])
        
        return np.array(obs, dtype=int)


class CatchEnvChangeDirection(CatchEnv):
    """
    Ambiente Catch con movimento orizzontale degli oggetti.
    Adatto per scenari più dinamici.
    """
    
    def __init__(self, 
                 h_speed_bins=3,
                 min_h_speed=-1.0,
                 max_h_speed=1.0,
                 direction=True,
                 **kwargs):
        self.direction=direction
        self.h_speed_bins = h_speed_bins
        self.min_h_speed = min_h_speed
        self.max_h_speed = max_h_speed
        self.h_speed_step = (self.max_h_speed - self.min_h_speed) / self.h_speed_bins
        self.fruit_h_speeds = []
        self.fruit_direction_changed = []  # Lista per tracciare i cambi di direzione
        super(CatchEnvChangeDirection, self).__init__(**kwargs, direction=direction)
        print(self.direction)
    
    def _initialize_observation_space(self):
        """
        Definisce lo spazio di osservazione come MultiDiscrete includendo i bin di velocità orizzontale e cambi di direzione.
        """
        # Definisce lo spazio MultiDiscrete
        # 1) basket_pos: [0, grid_size-1]
        # 2) Per ogni oggetto:
        #    row: [0, max_row_after_offset]
        #    col: [0, grid_size-1]
        #    type: [0, 1]
        #    v_speed_bin: [0, speed_bins-1]
        #    h_speed_bin: [0, h_speed_bins-1]
        #    direction: [1, 2,3]
        
        low_high_per_object = [
            self.max_row_after_offset + 1,  # row
            self.grid_size,                # col
            2,                             # type (0 o 1)
            self.speed_bins,               # v_speed_bin
            self.h_speed_bins,             # h_speed_bin
            3,                     # direction_change
        ]
        
        # Combina in una singola lista
        observation_space = [self.grid_size] + [dim for obj in range(self.max_objects_in_state) for dim in low_high_per_object]
        
        self.observation_space = spaces.MultiDiscrete(observation_space)
    
    def _get_observation(self):
        """
        Restituisce uno stato discreto composto da:
          [ basket_pos, (row, col, type, v_speed_bin, h_speed_bin) x max_objects_in_state ]
        """
        # 1) Posizione del cestino
        obs = [int(self.basket_pos)]
        
        # 2) Ordina gli oggetti per vicinanza al fondo (righe decrescenti)
        if len(self.fruit_rows) > 0:
            indices_sorted = sorted(
                range(len(self.fruit_rows)),
                key=lambda i: self.fruit_rows[i],
                reverse=True
            )
        else:
            indices_sorted = []
        
        # Prendi i primi max_objects_in_state oggetti
        indices_sorted = indices_sorted[:self.max_objects_in_state]
        
        # 3) Costruisci la parte dello stato per ogni oggetto
        for i in indices_sorted:
            row_float = self.fruit_rows[i]
            col_float = self.fruit_cols[i]
            malicious = self.fruit_is_malicious[i]
            v_speed = self.fruit_speeds[i]
            # Recupera la velocità orizzontale se disponibile
            if hasattr(self, 'fruit_h_speeds') and len(self.fruit_h_speeds) > i:
                h_speed = self.fruit_h_speeds[i]
            else:
                h_speed = 0.0  # Valore di default se non definito
            
            # Recupera il flag di cambiamento di direzione
            if hasattr(self, 'fruit_direction_changed') and len(self.fruit_direction_changed) > i:
                direction_changed = self.fruit_direction_changed[i]
            else:
                direction_changed = 0  # Default
            
            # (a) Discretizza la riga: offset per renderla positiva
            row_discrete = int(round(row_float)) + self.offset
            row_discrete = np.clip(row_discrete, 0, self.max_row_after_offset)
            
            # (b) Discretizza la colonna
            col_discrete = int(round(col_float))
            col_discrete = np.clip(col_discrete, 0, self.grid_size - 1)
            
            # (c) Tipo: 1 se malevolo, altrimenti 0
            obj_type = 1 if malicious else 0
            
            # (d) Speed bin verticale
            v_speed_bin = int((v_speed - self.min_speed) // self.speed_step)
            v_speed_bin = np.clip(v_speed_bin, 0, self.speed_bins - 1)
            
            #direction, se lo h speed è negativo allora 1, se è positivo allora 2, se è 0 allora 3
            direction = 0
            if h_speed < 0:
                direction = 1
            elif h_speed > 0:
                direction = 2
            else:
                direction = 3

            # (e) Speed bin orizzontale
            h_speed_bin = (h_speed - self.min_h_speed) // self.h_speed_step
            h_speed_bin = int(h_speed_bin)
            h_speed_bin = np.clip(h_speed_bin, 0, self.h_speed_bins - 1)
            
            obs.extend([row_discrete, col_discrete, obj_type, v_speed_bin, h_speed_bin, direction])
        
        # 4) Se ci sono meno oggetti, riempi con valori sentinel
        for _ in range(self.max_objects_in_state - len(indices_sorted)):
            # Esempio: row = grid_size - 1 + offset indica "oggetto inesistente"
            row_dummy = self.grid_size - 1 + self.offset
            col_dummy = 0
            type_dummy = 0
            v_speed_dummy = 0
            h_speed_dummy = 0
            direction_change_dummy = 0
            obs.extend([row_dummy, col_dummy, type_dummy, v_speed_dummy, h_speed_dummy, direction_change_dummy])
        
        return np.array(obs, dtype=int)
    
    def _update_objects(self):
        """
        Aggiorna le posizioni degli oggetti includendo il movimento orizzontale e tracciando i cambi di direzione.
        """
        for i in range(len(self.fruit_rows)):
            # Aggiorna la posizione verticale
            self.fruit_rows[i] += self.fruit_speeds[i]
            
            # Aggiorna la posizione orizzontale
            if hasattr(self, 'fruit_h_speeds') and len(self.fruit_h_speeds) > i:
                old_h_speed = self.fruit_h_speeds[i]
                self.fruit_cols[i] += self.fruit_h_speeds[i]
                
                # Reset del flag di cambiamento di direzione
                if hasattr(self, 'fruit_direction_changed') and len(self.fruit_direction_changed) > i:
                    self.fruit_direction_changed[i] = 0
                else:
                    # Inizializza se non presente
                    self.fruit_direction_changed = [0] * len(self.fruit_rows)
                
                # Gestisci le collisioni con i bordi orizzontali
                if self.fruit_cols[i] < 0:
                    self.fruit_cols[i] = 0
                    self.fruit_h_speeds[i] *= -1  # Inverte direzione
                    if hasattr(self, 'fruit_direction_changed') and len(self.fruit_direction_changed) > i:
                        self.fruit_direction_changed[i] = 1  # Ha cambiato direzione
                elif self.fruit_cols[i] > self.grid_size - 1:
                    self.fruit_cols[i] = self.grid_size - 1
                    self.fruit_h_speeds[i] *= -1  # Inverte direzione
                    if hasattr(self, 'fruit_direction_changed') and len(self.fruit_direction_changed) > i:
                        self.fruit_direction_changed[i] = 1  # Ha cambiato direzione
                
                # Cambia casualmente la direzione orizzontale con una certa probabilità
                change_dir_prob = 0.05  # 5% di probabilità
                if np.random.rand() < change_dir_prob:
                    #le direzioni possono essere 0.05, 0.1, 0.2, -0.05, -0.1, -0.2
                    self.fruit_h_speeds[i] = np.random.choice([-0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2])
                    # Limita la velocità orizzontale ai limiti specificati
                    self.fruit_h_speeds[i] = np.clip(
                        self.fruit_h_speeds[i],
                        self.min_h_speed,
                        self.max_h_speed
                    )
                    # Verifica se la velocità è cambiata
                    if self.fruit_h_speeds[i] != old_h_speed:
                        if hasattr(self, 'fruit_direction_changed') and len(self.fruit_direction_changed) > i:
                            self.fruit_direction_changed[i] = 1  # Ha cambiato direzione
    
    def _spawn_new_fruit(self):
        """
        Tenta di spawnare un nuovo oggetto includendo la velocità orizzontale.
        """
        if len(self.fruit_rows) < self.num_objects and np.random.rand() < self.spawn_probability:
            new_row = self._generate_unique_row()
            if new_row is None:
                return

            new_col = self._generate_unique_column()
            if new_col is None:
                return

            v_speed = np.random.uniform(self.min_speed, self.max_speed)
            is_malicious = (np.random.rand() < self.malicious_probability)
            
            # Assegna una velocità orizzontale
            h_speed = np.random.uniform(self.min_h_speed, self.max_h_speed)
            
            self.fruit_rows.append(new_row)
            self.fruit_cols.append(new_col)
            self.fruit_speeds.append(v_speed)
            self.fruit_is_malicious.append(is_malicious)
            
            # Inizializza la lista della velocità orizzontale se non presente
            if not hasattr(self, 'fruit_h_speeds'):
                self.fruit_h_speeds = []
            self.fruit_h_speeds.append(h_speed)
            
            # Inizializza la lista dei cambi di direzione
            if not hasattr(self, 'fruit_direction_changed'):
                self.fruit_direction_changed = []
            self.fruit_direction_changed.append(0)  # Inizialmente, nessun cambiamento di direzione
    

# Test per vedere se l'ambiente funziona renderizzato per umano
if __name__ == "__main__":
    env = CatchEnvChangeDirection(grid_size=15, render_mode="human")
    env.reset()
    done = False
    while not done:
        env.render()
        _, _, done, _, _ = env.step(2)  # Esempio di azione fissa
    env.close()
