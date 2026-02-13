# Deep Q Network (DQN)

## 1. Pseudocode

```text
Initialize Replay Memory M
Initialize Deep Q Network Q with random parameters
Initialize Epsilon

Train(n_epochs):
    For epoch = 1 to n_epochs:
        State = Env.Reset()
        
        Loop:
            # 1. Action Selection (Epsilon Greedy)
            If Random < Epsilon:
                Action = Random
            Else:
                Action = Argmax(Q(State))
                
            # 2. Step
            Next_State, Reward, Done = Env.Step(Action)
            
            # 3. Store
            Store (State, Action, Reward, Next_State, Done) in Memory
            
            # 4. Replay
            Batch = Sample Randomly from Memory
            X_train = []
            y_train = []
            
            For (s, a, r, s_next, done) in Batch:
                Target = Q(s)
                If done:
                    Target[a] = r
                Else:
                    Target[a] = r + Gamma * Max(Q(s_next))
                    
                X_train.append(s)
                y_train.append(Target)
                
            # 5. Train
            Model.Train(X_train, y_train)
            
            State = Next_State
            If Done: Break
            
        Decay Epsilon
```

## 2. Algorithm Explanation

**Deep Q-Learning (DQN)** combines Q-Learning with Deep Neural Networks.

In standard Q-Learning, a Q-table is used to store Q-values for every state-action pair. However, for environments with large state spaces (like pixels in a game), a table is infeasible.
DQN uses a Neural Network to **approximate** the Q-function: $Q(s, a; \theta) \approx Q^*(s, a)$.

Key components:
1.  **Experience Replay**: Stores transitions and samples them randomly to break correlations between consecutive samples, stabilizing training.
2.  **Target Network**: (Note: Not explicitly implemented in this basic version, but standard in full DQN) Uses a separate, slowly updating network to calculate targets, further stabilizing training.

## 3. Math Formulas

**Bellman Equation (for Q-value):**
$$ Q(s, a) = r + \gamma \max_{a'} Q(s', a') $$

**Loss Function (MSE):**
$$ L = \mathbb{E} \left[ (r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta))^2 \right] $$

## 4. Inputs Required

-   **env_name**: OpenAI Gym environment name (e.g., 'CartPole-v1').
-   **epsilon**: Exploration rate.
-   **gamma**: Discount factor.
-   **decay_rate**: Rate at which epsilon decays.

## 5. Usage Guidelines

### Industry Best Practices:
-   **Double DQN**: Mitigates overestimation bias of Q-values.
-   **Dueling DQN**: Splits Q-network into Value $V(s)$and Advantage$A(s, a)$ streams.
-   **Prioritized Experience Replay**: Replay important transitions more often.
-   **Hyperparameters**: Extremely sensitive. Requires careful tuning of learning rate, batch size, and buffer size.

## 6. Concurrency, Parallelism, Memory Management

-   **Memory**: Replay Buffer can grow large. Limited by `memory_size`.
-   **Concurrency**: Env stepping and Training can be parallelized (e.g., A3C algorithm), but standard DQN is sequential.

## 7. Underlying Data Structure

-   **Deque**: Used for Experience Replay memory (efficient append/pop).
-   **Numpy Arrays**: Batch processing.
