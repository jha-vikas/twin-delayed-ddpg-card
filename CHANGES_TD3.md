# TD3 Implementation Changes for Car Navigation

This document details all modifications made to convert the DQN-based car navigation from `ass16/citymap_assignment.py` to use TD3 (Twin Delayed DDPG) algorithm from `ERA1S25.ipynb`.

---

## Summary

The original DQN implementation used **discrete actions** (5 choices: left, straight, right, sharp left, sharp right). The new TD3 implementation uses **continuous actions** (steering angle from -1.0 to +1.0), which is the fundamental characteristic of actor-critic methods like DDPG and TD3.

---

## 1. Algorithm Change: DQN → TD3

### What Changed

| Aspect | DQN (Original) | TD3 (New) |
|--------|----------------|-----------|
| **Action Space** | Discrete (5 choices) | Continuous (-1.0 to +1.0) |
| **Network Architecture** | Single Q-Network | Actor + Twin Critics |
| **Exploration** | Epsilon-greedy | Gaussian noise |
| **Target Networks** | Single target net | Actor target + Critic target |
| **Policy Updates** | Every step | Delayed (every 2 steps) |
| **Value Estimation** | Single Q-value | Minimum of twin Q-values |

### Why TD3?

TD3 addresses three key issues in DDPG:
1. **Overestimation Bias** - Using twin critics and taking the minimum reduces Q-value overestimation
2. **Policy Instability** - Delayed policy updates reduce variance in actor learning
3. **Target Smoothing** - Adding noise to target actions smooths the Q-function estimate

---

## 2. Neural Network Architecture

### Original DQN
```
Input(9) → FC(128) → ReLU → FC(256) → ReLU → FC(512) → ReLU → FC(256) → ReLU → FC(128) → ReLU → Output(5)
```
Single network outputting Q-values for 5 discrete actions.

### New TD3

**Actor Network (Policy):**
```
Input(9) → FC(400) → ReLU → FC(300) → ReLU → FC(1) → tanh × max_action
```
Maps state directly to continuous steering action.

**Twin Critic Networks (Q-Functions):**
```
Input(9+1) → FC(400) → ReLU → FC(300) → ReLU → FC(1)
```
Two independent networks, each estimating Q(state, action).

### Why These Sizes?

The 400-300 hidden layer sizes match the original TD3 paper and the ERA1S25.ipynb implementation. This architecture has been proven effective for continuous control tasks.

---

## 3. Action Space Conversion

### Original (Discrete)
```python
# 5 discrete actions
if action == 0:   turn = -TURN_SPEED    # Left
elif action == 1: turn = 0               # Straight
elif action == 2: turn = TURN_SPEED      # Right
elif action == 3: turn = -SHARP_TURN     # Sharp left
elif action == 4: turn = SHARP_TURN      # Sharp right
```

### New (Continuous)
```python
# Action is continuous value from -1.0 to +1.0
action = np.clip(action, -1.0, 1.0)
turn = action * MAX_TURN  # Scale to degrees
```

### Advantages of Continuous Actions

1. **Smoother Control** - Car can make any turn angle, not just 5 fixed options
2. **Better Learning** - Gradient-based optimization can fine-tune actions
3. **More Realistic** - Mimics real car steering behavior

---

## 4. Exploration Strategy

### Original (Epsilon-Greedy)
```python
if random.random() < self.epsilon:
    action = random.randint(0, 4)  # Random discrete action
else:
    action = policy_net(state).argmax()
```

### New (Gaussian Noise)
```python
if total_timesteps < START_TIMESTEPS:
    action = np.random.uniform(-1, 1)  # Pure random exploration
else:
    action = actor(state) + np.random.normal(0, EXPL_NOISE)
    action = np.clip(action, -1, 1)
```

### Why Gaussian Noise?

- Continuous action spaces require continuous exploration
- Gaussian noise centered at the policy action enables targeted exploration
- Initial random phase (START_TIMESTEPS) ensures diverse experience collection

---

## 5. Replay Buffer

### Original (Prioritized)
```python
# Separate buffers for successful/failed episodes
self.memory = deque(maxlen=10000)
self.priority_memory = deque(maxlen=3000)
```

### New (Uniform)
```python
class ReplayBuffer:
    def __init__(self, max_size=1e6):
        self.storage = []
        self.max_size = int(max_size)
        self.ptr = 0
```

### Change Rationale

The TD3 paper uses uniform replay. While prioritized experience replay can help, the continuous action space with Gaussian exploration already provides good coverage. Keeping it simple follows the original TD3 implementation.

---

## 6. Training Process

### TD3 Training Steps (from train() method)

```
For each training iteration:
    1. Sample batch from replay buffer
    2. Compute next actions using actor_target
    3. Add clipped Gaussian noise to next actions (target policy smoothing)
    4. Compute target Q-values: Q_target = min(Q1_target, Q2_target)
    5. Compute TD target: y = r + γ * (1-done) * Q_target
    6. Update both critics with MSE loss
    
    Every 2 iterations (delayed updates):
        7. Update actor by maximizing Q1(s, actor(s))
        8. Soft update all target networks (Polyak averaging)
```

### Key Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `BATCH_SIZE` | 100 | Batch size for training |
| `GAMMA` | 0.99 | Discount factor |
| `TAU` | 0.005 | Soft update rate |
| `POLICY_NOISE` | 0.2 | Noise added to target actions |
| `NOISE_CLIP` | 0.5 | Clipping range for target noise |
| `POLICY_FREQ` | 2 | Actor update frequency |
| `EXPL_NOISE` | 0.1 | Exploration noise std |
| `START_TIMESTEPS` | 1000 | Random exploration period |

---

## 7. GUI Modifications

### New UI Elements

1. **Action Display** - Shows current continuous action value (-1.0 to +1.0)
2. **Timesteps Counter** - Shows total environment steps
3. **Save Model Button** - Saves actor and critic weights

### Updated Stats Panel

```
Timesteps: [total steps]
Episode: [episode number]
Score: [current episode score]
Action: [current steering value]
```

### Changed Labels

- Title: "TD3 CONTROLS" (was "CONTROLS (V2 - Safety)")
- Window: "NeuralNav TD3: Continuous Steering Control"

---

## 8. Removed Features

### Safety Override System

The original `safety_override()` method was removed. Reasons:
1. TD3 should learn obstacle avoidance through rewards
2. Safety override would interfere with continuous action learning
3. The agent needs to experience crashes to learn avoidance

### Prioritized Replay

Simplified to uniform replay to match standard TD3.

### Epsilon Decay

Replaced with fixed exploration noise, as is standard for actor-critic methods.

---

## 9. File Structure

```
ass17/
├── city_assignment_part2.py    # TD3 implementation (NEW)
├── CHANGES_TD3.md              # This documentation (NEW)
├── ERA1S25.ipynb              # Reference TD3 notebook
└── Archive/                    # Previous DQN versions
```

---

## 10. Running the Application

### Prerequisites
```bash
pip install torch numpy PyQt6
```

### Launch
```bash
cd ass17
python city_assignment_part2.py
```

### Training Steps
1. Click on map to place the car (starting position)
2. Left-click to add targets (can add multiple)
3. Right-click to finish setup
4. Press SPACE or click START to begin training
5. Watch the car learn continuous steering control!

### Tips for Training
- First 1000 timesteps are pure random exploration
- Learning starts after the replay buffer has enough samples
- Expect crashes early - the agent is learning!
- Check the reward chart for learning progress

---

## 11. Video Recording

For assignment submission, record:
1. **Setup Phase** - Placing car and targets
2. **Early Training** - Random behavior, crashes
3. **Learning Progress** - Improving navigation
4. **Final Performance** - Smooth steering to targets

### Recording Tools
- **macOS**: QuickTime Player (File → New Screen Recording)
- **Windows**: OBS Studio or Xbox Game Bar
- **Linux**: OBS Studio

---

## References

1. [TD3 Paper](https://arxiv.org/abs/1802.09477): "Addressing Function Approximation Error in Actor-Critic Methods"
2. ERA1S25.ipynb: Course notebook with AntBulletEnv-v0 implementation
3. ass16/citymap_assignment.py: Original DQN car navigation

---

*Created for Assignment 17 - TD3 Car Navigation*

