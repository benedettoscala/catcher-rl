# Catcher-RL

Catcher-RL is a Reinforcement Learning project for the "Catcher" game implemented with various learning techniques, including DQN (Deep Q-Network) and SARSA (State-Action-Reward-State-Action). The game involves an agent that must catch falling objects on the screen, improving its behavior through reinforcement learning.

## Project Structure

The project is structured into several folders:

- `agent/` : Contains reinforcement learning algorithms
  - `dqn_cnn.py` : Implementation of DQN with CNN, train dqn with this
  - `sarsa.py` : Classic SARSA implementation, train q-table with this
  - `sarsa_approximated.py` : SARSA with function approximation, train sarsa with nn with this

- `environment/` : Contains the game simulation environment
  - `catcher_discrete.py` : Implementation with discrete states
  - `catcher_image.py` : Implementation with image input
  - `core/catch_base.py` : Base of the game environment
    
- `play/` : Contains the game simulation environment
  - `play_dqn_agent.py` : Play the dqn agent
  - `play_q_sarsa_approximated.py` : Play the sarsa with neural network agent
  - `play_q_table_sarsa.py` : Play the sarsa with q table agent

- `assets/` : Contains background images and game objects

- `metrics/` : Contains graphs and experiment metrics

- `networks/` : Contains the neural network definitions used for DQN
  - `QNetwork.py` : Network model for the DQN algorithm

- `tensorboard_runs/` : Logs of training sessions for TensorBoard

- `tests/` : Contains testing scripts to measure model performance

## Installation

To install the project dependencies, run:

```bash
pip install -r requirements.txt
```

## Execution

### Train an agent with DQN using CNN
```bash
python -m agent.dqn_cnn
```

### Train an agent with SARSA
```bash
python -m agent.sarsa
```

### Train an agent with SARSA approximated with a neural network
```bash
python -m agent.sarsa-approximated
```

## Play

### Play an agent with DQN using CNN
```bash
python -m play.play_dqn_agent
```

### Play an agent with SARSA
```bash
python -m play.play_q_sarsa
```

### Train an agent with SARSA approximated with a neural network
```bash
python -m play.play_q_approximated
```

