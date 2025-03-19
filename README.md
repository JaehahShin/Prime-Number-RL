# Prime Number Explorer

An interactive reinforcement learning environment for exploring prime numbers using Gymnasium and Stable-Baselines3.

## Overview

This project creates a custom Gymnasium environment where an agent learns to navigate the number line in search of prime numbers. The environment, `PrimeExplorerEnv`, allows an agent to move along the number line with different step sizes and rewards it for landing on prime numbers.

## Features

- **Custom Gymnasium Environment**: A specialized environment for prime number exploration
- **Real-time Visualization**: Detailed, multi-panel visualization of the agent's behavior
- **Reinforcement Learning Integration**: Includes DQN (Deep Q-Network) agent implementation using Stable-Baselines3
- **Prime Number Mechanics**: Uses sympy for accurate prime number verification
- **Performance Tracking**: Monitors and visualizes prime discovery rate, action distribution, and more

## Requirements

- Python 3.7+
- PyTorch
- Gymnasium
- Stable-Baselines3
- Matplotlib
- Sympy
- NumPy
- IPython (for visualization)

## Installation

```bash
# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install gymnasium stable-baselines3 matplotlib sympy numpy ipython
```

## Usage

### Basic Usage

```python
from prime_explorer import PrimeExplorerEnv
from stable_baselines3 import DQN

# Create environment
env = PrimeExplorerEnv(start=2, max_num=1000, max_steps=100, render_mode="visualization")

# Initialize and train a DQN agent
model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

# Test the trained agent
obs, _ = env.reset()
for _ in range(100):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    
    if terminated or truncated:
        break
```

### Environment Parameters

- `start`: Starting number (default: 2)
- `max_num`: Maximum number to reach (default: 10000)
- `max_steps`: Maximum steps per episode (default: 200)
- `render_mode`: Visualization mode ("human" or "visualization")

### Actions

The agent can choose from 4 actions:
- **0**: Move +1 step
- **1**: Move +2 steps
- **2**: Move +5 steps
- **3**: Move +10 steps

### Rewards

- **+1.0**: Landing on a prime number
- **-0.1**: Landing on a non-prime number

## Visualization

When using `render_mode="visualization"`, the environment provides a rich multi-panel display:

1. **Position vs Step**: Shows the agent's path with prime numbers highlighted
2. **Cumulative Reward**: Tracks total reward and success rate over time
3. **Action Distribution**: Visualizes which actions the agent prefers
4. **Prime Density Heatmap**: Shows where prime numbers are more concentrated
5. **Recent Actions Timeline**: Displays recent movements and their outcomes

## Advanced Usage

See the main script for examples of:
- Periodic visualization during training
- GPU acceleration with CUDA
- Custom training loops

## How It Works

The environment simulates an agent navigating the number line. At each step:

1. The agent chooses how far to move (1, 2, 5, or 10 steps)
2. The environment checks if the new position is a prime number
3. The agent receives a reward (+1 for primes, -0.1 otherwise)
4. The visualization updates to show the agent's progress

Over time, the agent learns patterns that lead to higher concentrations of prime numbers.

## License

[MIT License](LICENSE)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
