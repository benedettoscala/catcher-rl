# Catcher-RL

Catcher-RL is a Reinforcement Learning project for the "Catcher" game implemented with various learning techniques, including DQN (Deep Q-Network) and SARSA (State-Action-Reward-State-Action). The game involves an agent that must catch falling objects on the screen, improving its behavior through reinforcement learning.

## Project Structure

The project is structured into several folders:

- `agent/` : Contains reinforcement learning algorithms
  - `dqn_cnn.py` : Implementation of DQN with CNN
  - `play_dqn_agent.py` : Script to run a trained DQN agent
  - `play_dqn_agent_cnn.py` : CNN version of DQN
  - `play_sarsa.py` : Execution of the SARSA agent
  - `play_sarsa_nn.py` : Neural network version for SARSA
  - `sarsa.py` : Classic SARSA implementation
  - `sarsa_approximated.py` : SARSA with function approximation

- `environment/` : Contains the game simulation environment
  - `catcher_discrete.py` : Implementation with discrete states
  - `catcher_image.py` : Implementation with image input
  - `core/catch_base.py` : Base of the game environment

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
python agent/dqn_cnn.py
```

### Train an agent with SARSA
```bash
python agent/sarsa.py
```

### Train an agent with SARSA approximated with a neural network
```bash
python agent/sarsa.py
```
