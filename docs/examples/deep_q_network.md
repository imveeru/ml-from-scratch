# Deep Q-Network (DQN) Example

This example trains an agent to play CartPole-v1 using DQN.

## Description
-   **Environment**: OpenAI Gym `CartPole-v1`.
-   **Task**: Reinforcement Learning (Balance pole on cart).
-   **Model**: Deep Q-Network (2-layer MLP).
-   **Training**: Runs for 500 epochs. Dictionary `memory` stores experience.
-   **Play**: Demonstrates the trained agent for 100 epochs.

## Usage
Run the script from the root directory:
```bash
python -m mlfromscratch.examples.deep_q_network
```

## Output
-   Prints training progress (Loss, Reward, Epsilon).
-   Renders the environment during play phase (requires a display or suitable backend).
