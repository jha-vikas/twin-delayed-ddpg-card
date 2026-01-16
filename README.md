# TD3 Car Navigation - Continuous Steering Control

A Twin Delayed Deep Deterministic Policy Gradient (TD3) implementation for autonomous car navigation on city maps using continuous action space reinforcement learning.

## Overview

This project implements a self-driving car agent using **TD3**, an advanced actor-critic reinforcement learning algorithm. Unlike discrete action spaces (turn left/right/straight), this implementation uses **continuous actions** (steering angle from -1.0 to +1.0), enabling smooth and realistic car control.

### Key Features

- **Continuous Action Space** - Smooth steering control (-1.0 to +1.0) instead of discrete choices
- **TD3 Algorithm** - Twin critics reduce overestimation bias, delayed policy updates improve stability
- **Multi-Target Navigation** - Sequential target reaching with progressive rewards (200, 300, 400, 500...)
- **Real-time Visualization** - PyQt6 GUI with sensor visualization and reward charts
- **Inference Mode** - Record clean demos without exploration noise (frozen weights, deterministic)
- **Multiple Maps** - Support for various city map configurations (map0.jpg - map5.jpg)

## Video Demo

**Watch the full demonstration:** [YouTube Video](https://youtu.be/ck3ebGH_vQk)

The video showcases TD3 training process, continuous steering control, multi-target navigation, and smooth pathfinding through complex city maps.

## Algorithm Comparison: DQN vs TD3

| Aspect | DQN (ass16) | TD3 (ass17) |
|--------|-------------|-------------|
| **Action Space** | Discrete (5 choices) | Continuous (-1.0 to +1.0) |
| **Network** | Single Q-Network | Actor + Twin Critics |
| **Exploration** | Epsilon-greedy | Gaussian noise |
| **Updates** | Every step | Delayed (every 2 steps) |
| **Control** | Step-wise turns | Smooth steering |

## Requirements

```bash
pip install -r requirements.txt
# Or manually:
pip install torch numpy PyQt6
```

## Quick Start

### 1. Run the Application

```bash
cd ass17
python city_assignment_part2.py
```

### 2. Setup Navigation

1. **Left-click** on map to place car starting position
2. **Left-click** multiple times to add sequential targets (1, 2, 3, ...)
3. **Right-click** to confirm target placement
4. Press **SPACE** to start training

### 3. Watch the Car Learn!

The car will initially move randomly, then gradually learn to:
- Stay on roads (bright pixels)
- Avoid obstacles (dark pixels)
- Navigate to all targets in sequence using smooth steering

### 4. Inference Mode (Demo Recording)

1. Train until satisfied with performance
2. Click **💾 SAVE MODEL** to save weights
3. Click **🎬 INFERENCE MODE** (button turns green)
4. Start screen recording
5. Press SPACE to watch smooth, deterministic behavior

**Inference Mode Features:**
- No exploration noise (smooth driving)
- Frozen weights (no learning)
- Perfect for demo videos

## File Structure

```
ass17/
├── city_assignment_part2.py    # Main TD3 implementation
├── CHANGES_TD3.md              # Detailed change documentation
├── README.md                   # This file
├── maps/                       # City map images
│   ├── map0.jpg
│   ├── map1.jpg
│   ├── map2.jpg
│   ├── map3.jpg              # Default map (Paris-style)
│   ├── map4.jpg
│   └── map5.jpg
└── models/                     # Saved model weights
    ├── td3_car_actor.pth
    └── td3_car_critic.pth
```

## Parameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `SENSOR_DIST` | 50 | Sensor range in pixels |
| `SENSOR_ANGLE` | 15° | Angle between sensors (150° total FOV) |
| `SPEED` | 2 | Movement speed per step |
| `MAX_TURN` | 25° | Maximum steering angle |
| `BATCH_SIZE` | 100 | Training batch size |
| `GAMMA` | 0.99 | Future reward discount |
| `TAU` | 0.005 | Soft target update rate |
| `POLICY_NOISE` | 0.2 | Target policy smoothing noise |
| `NOISE_CLIP` | 0.5 | Target noise clipping range |
| `POLICY_FREQ` | 2 | Delayed policy update frequency |
| `EXPL_NOISE` | 0.2 | Exploration noise std |
| `START_TIMESTEPS` | 500 | Random exploration period |
| `REPLAY_BUFFER_SIZE` | 200k | Experience replay buffer size |

## Neural Network Architecture

### Actor Network (Policy)
```
Input(9) → FC(400) → ReLU → FC(300) → ReLU → FC(1) → tanh × max_action
```
Maps state directly to continuous steering action.

### Twin Critic Networks (Q-Functions)
```
Input(9+1) → FC(400) → ReLU → FC(300) → ReLU → FC(1)
```
Two independent Q-networks to reduce overestimation bias.

**Input**: 7 sensor values + angle to target + distance to target  
**Output**: Continuous steering angle (-1.0 to +1.0)

## State and Action Spaces

### State Space (9 dimensions)
- 7 sensor readings (normalized distance to obstacles, 0-1)
- 1 angle to target (normalized -1 to 1, where -1 = 180° left, +1 = 180° right)
- 1 distance to target (normalized 0 to 1)

### Action Space (1 dimension)
- Continuous steering angle: -1.0 (full left) to +1.0 (full right)
- Scaled to -25° to +25° turn angle per step

## Reward Structure

The reward system encourages efficient navigation:

- **Target Reaching**: Progressive bonuses
  - Target 1: +200
  - Target 2: +300
  - Target 3: +400
  - Target 4: +500
  - All Complete: +500 bonus
- **Distance Progress**: +5 × progress (getting closer)
- **Road Following**: +0.5 for clear center path
- **Step Penalty**: -0.5 per step (discourages loops)
- **Crash Penalty**: -50
- **Direction Penalty**: -1 if facing >90° away from target

## TD3 Algorithm Details

### Key Innovations

1. **Twin Critics**: Two Q-networks reduce overestimation bias
2. **Delayed Policy Updates**: Actor updated every 2 critic updates (reduces variance)
3. **Target Policy Smoothing**: Noise added to target actions (smooths Q-function)
4. **Clipped Double Q-Learning**: Uses minimum of twin Q-values

### Training Process

```
For each iteration:
    1. Sample batch from replay buffer
    2. Compute target Q-values with noise smoothing
    3. Update both critics with MSE loss
    4. Every 2 iterations (POLICY_FREQ):
       - Update actor (maximize Q1)
       - Soft update target networks (Polyak averaging)
```

## Available Maps

Maps are located in the `maps/` folder:

| Map | Description | Difficulty |
|-----|-------------|------------|
| map0.jpg | Basic city layout | Easy |
| map1.jpg | Realistic aerial city view | Medium |
| map2.jpg | Grid city with waterways | Easy |
| map3.jpg | Paris-style with river (default) | Medium |
| map4.jpg | Night aerial (stunning visuals) | Medium |
| map5.jpg | River city, organic layout | Medium |

## Documentation

- **[CHANGES_TD3.md](CHANGES_TD3.md)** - Detailed documentation of all modifications from DQN to TD3

## Results

The agent successfully learns to:
- Navigate complex city maps
- Avoid obstacles using sensor data
- Reach multiple targets sequentially
- Use smooth, continuous steering control

**Example Performance:**
- Episode 151: Completed all 4 targets in 244 steps
- Consistent target reaching after sufficient training
- Smooth navigation in inference mode

## Troubleshooting

### Car Crashes Immediately
- Ensure car is placed on bright/white road pixels
- Check map contrast (dark = obstacles, bright = roads)

### Agent Not Learning
- Wait for initial exploration period (500 steps)
- Check reward chart for upward trend
- Verify targets are reachable

### Application Won't Start
```bash
# Verify dependencies
python -c "import torch, numpy, PyQt6"

# Check Python version
python --version  # Should be 3.8+
```

### Map Not Loading
- Ensure `maps/` folder exists in `ass17/`
- Use "📂 LOAD MAP" button to manually select map

### Agent Stuck in Loops
- Increase step penalty (already -0.5)
- Check if targets are too far apart
- Verify reward structure is encouraging progress

## References

1. **TD3 Paper**: [Addressing Function Approximation Error in Actor-Critic Methods](https://arxiv.org/abs/1802.09477)
   - Fujimoto et al., 2018

2. **DDPG Paper**: [Continuous Control with Deep Reinforcement Learning](https://arxiv.org/abs/1509.02971)
   - Lillicrap et al., 2015

## License

MIT License - Educational implementation of TD3 for autonomous navigation.

## Author

Assignment 17 - TD3 Car Navigation Implementation

---

**Built with PyTorch, PyQt6, and TD3**
