# Catcher RL

## Descrizione del Progetto
Questo progetto implementa agenti di Reinforcement Learning (RL) per il gioco "Catcher" utilizzando tecniche come Deep Q-Network (DQN) e SARSA. Il codice include ambienti personalizzati, reti neurali e strumenti di valutazione per l'addestramento e la validazione degli agenti.

## Struttura del Progetto

- **`agent/`**: Contiene implementazioni di agenti RL, inclusi DQN e SARSA.
  - `dqn_cnn.py`: Implementazione di un agente DQN con CNN.
  - `sarsa.py`: Implementazione dell'algoritmo SARSA.
  - `sarsa_approximated.py`: Variante di SARSA con approssimazione.
  - `play_*`: Script per testare gli agenti.

- **`environment/`**: Definisce l'ambiente di gioco "Catcher".
  - `catcher_discrete.py`: Versione con azioni discrete.
  - `catcher_image.py`: Versione basata su immagini.
  - `core/catch_base.py`: Classe base per l'ambiente.

- **`networks/`**: Contiene modelli di reti neurali.
  - `QNetwork.py`: Rete neurale per Q-learning.

- **`metrics/`**: Grafici delle metriche di addestramento (PDF).

- **`tensorboard_runs/`**: Log per TensorBoard.

- **`tests/`**: Contiene test.
  - `test_dqn_cnn.py`: Test per DQN.
  - `test_q_sarsa_approximated.py`: Test per SARSA.

- **`assets/`**: Contiene immagini di gioco (sfondo, frutti, bombe).

## Requisiti

Le dipendenze del progetto sono elencate in `requirements.txt`. Per installarle:
```sh
pip install -r requirements.txt
```
è necessario installare torch.

## Utilizzo

### Eseguire un agente DQN
```sh
python agent/play_dqn_agent.py
```

### Eseguire un agente SARSA
```sh
python agent/play_sarsa.py
```

## Autori
Questo progetto è stato sviluppato per esplorare tecniche di Reinforcement Learning su ambienti personalizzati.


