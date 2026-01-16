"""
===============================================================================
ASSIGNMENT: AUTONOMOUS CAR NAVIGATION WITH TD3 (Twin Delayed DDPG)
===============================================================================

This implementation uses TD3 (Twin Delayed Deep Deterministic Policy Gradient)
algorithm for continuous steering control of an autonomous car.

Key Features:
1. Continuous Action Space - Steering angle from -1.0 to +1.0
2. Actor-Critic Architecture - Separate networks for policy and value
3. Twin Critics - Two Q-networks to reduce overestimation bias
4. Delayed Policy Updates - Actor updated every 2 critic updates
5. Target Policy Smoothing - Noise added to target actions

Based on:
- TD3 Paper: "Addressing Function Approximation Error in Actor-Critic Methods"
- Reference Implementation: ERA1S25.ipynb (AntBulletEnv-v0)
- Environment: ass16/citymap_assignment.py

===============================================================================
"""

import sys
import os
import math
import numpy as np
import random
from collections import deque

# --- PYTORCH ---
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# --- PYQT ---
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QGraphicsScene, 
                             QGraphicsView, QGraphicsItem, QFrame, QFileDialog,
                             QTextEdit, QGridLayout)
from PyQt6.QtGui import (QImage, QPixmap, QColor, QPen, QBrush, QPainter, 
                         QPolygonF, QFont, QPainterPath)
from PyQt6.QtCore import Qt, QTimer, QPointF, QRectF

# ==========================================
# 1. CONFIGURATION & THEME
# ==========================================
# Nordic Theme
C_BG_DARK   = QColor("#2E3440") 
C_PANEL     = QColor("#3B4252")
C_INFO_BG   = QColor("#4C566A") 
C_ACCENT    = QColor("#88C0D0") 
C_TEXT      = QColor("#ECEFF4") 
C_SUCCESS   = QColor("#A3BE8C") 
C_FAILURE   = QColor("#BF616A") 
C_SENSOR_ON = QColor("#A3BE8C")  # Green
C_SENSOR_OFF= QColor("#BF616A")  # Red

# ==========================================
# PHYSICS PARAMETERS
# ==========================================
CAR_WIDTH = 14     
CAR_HEIGHT = 8   
SENSOR_DIST = 50   # Sensor range for obstacle detection
SENSOR_ANGLE = 15  # Angle between sensors (degrees)
SPEED = 2          # Forward speed (pixels/step)
MAX_TURN = 25      # Maximum turn angle (degrees) - continuous action scaled to this

# ==========================================
# TD3 HYPERPARAMETERS (from ERA1S25.ipynb)
# ==========================================
BATCH_SIZE = 100        # Batch size for training
GAMMA = 0.99            # Discount factor
TAU = 0.005             # Soft target update rate
ACTOR_LR = 0.001        # Actor learning rate
CRITIC_LR = 0.001       # Critic learning rate
POLICY_NOISE = 0.2      # Noise added to target policy
NOISE_CLIP = 0.5        # Range to clip target policy noise
POLICY_FREQ = 2         # Frequency of delayed policy updates
EXPL_NOISE = 0.2        # Exploration noise std (increased for better exploration)
START_TIMESTEPS = 500   # Timesteps before using policy (random exploration)
MAX_CONSECUTIVE_CRASHES = 5

# Target Colors (for multiple targets)
TARGET_COLORS = [
    QColor(0, 255, 255),      # Cyan
    QColor(255, 100, 255),    # Magenta
    QColor(0, 255, 100),      # Green
    QColor(255, 150, 0),      # Orange
    QColor(100, 150, 255),    # Blue
    QColor(255, 50, 150),     # Pink
    QColor(150, 255, 50),     # Lime
    QColor(255, 255, 0),      # Yellow
]

# Device selection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. EXPERIENCE REPLAY BUFFER
# ==========================================
class ReplayBuffer:
    """
    Experience Replay Buffer for TD3.
    Stores (state, next_state, action, reward, done) transitions.
    """
    def __init__(self, max_size=1e6):
        self.storage = []
        self.max_size = int(max_size)
        self.ptr = 0

    def add(self, transition):
        if len(self.storage) == self.max_size:
            self.storage[self.ptr] = transition
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(transition)
            self.ptr = (self.ptr + 1) % self.max_size

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = [], [], [], [], []
        for i in ind:
            state, next_state, action, reward, done = self.storage[i]
            batch_states.append(np.asarray(state))
            batch_next_states.append(np.asarray(next_state))
            batch_actions.append(np.asarray(action))
            batch_rewards.append(np.asarray(reward))
            batch_dones.append(np.asarray(done))
        return (
            np.array(batch_states), 
            np.array(batch_next_states), 
            np.array(batch_actions), 
            np.array(batch_rewards).reshape(-1, 1), 
            np.array(batch_dones).reshape(-1, 1)
        )
    
    def __len__(self):
        return len(self.storage)

# ==========================================
# 3. ACTOR NETWORK (Policy)
# ==========================================
class Actor(nn.Module):
    """
    Actor Network for TD3.
    Maps state to continuous action (steering angle).
    
    Architecture: Input(9) -> FC(400) -> ReLU -> FC(300) -> ReLU -> FC(1) -> tanh * max_action
    """
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.layer_1 = nn.Linear(state_dim, 400)
        self.layer_2 = nn.Linear(400, 300)
        self.layer_3 = nn.Linear(300, action_dim)
        self.max_action = max_action

    def forward(self, x):
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        x = self.max_action * torch.tanh(self.layer_3(x))
        return x

# ==========================================
# 4. CRITIC NETWORK (Twin Q-Networks)
# ==========================================
class Critic(nn.Module):
    """
    Twin Critic Networks for TD3.
    Two independent Q-networks to reduce overestimation bias.
    Each takes (state, action) and outputs Q-value.
    
    Architecture: Input(9+1) -> FC(400) -> ReLU -> FC(300) -> ReLU -> FC(1)
    """
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # First Critic Network
        self.layer_1 = nn.Linear(state_dim + action_dim, 400)
        self.layer_2 = nn.Linear(400, 300)
        self.layer_3 = nn.Linear(300, 1)
        # Second Critic Network
        self.layer_4 = nn.Linear(state_dim + action_dim, 400)
        self.layer_5 = nn.Linear(400, 300)
        self.layer_6 = nn.Linear(300, 1)

    def forward(self, x, u):
        xu = torch.cat([x, u], 1)
        # First Critic
        x1 = F.relu(self.layer_1(xu))
        x1 = F.relu(self.layer_2(x1))
        x1 = self.layer_3(x1)
        # Second Critic
        x2 = F.relu(self.layer_4(xu))
        x2 = F.relu(self.layer_5(x2))
        x2 = self.layer_6(x2)
        return x1, x2

    def Q1(self, x, u):
        """Return only Q1 value (used for actor loss)"""
        xu = torch.cat([x, u], 1)
        x1 = F.relu(self.layer_1(xu))
        x1 = F.relu(self.layer_2(x1))
        x1 = self.layer_3(x1)
        return x1

# ==========================================
# 5. TD3 AGENT
# ==========================================
class TD3:
    """
    TD3 (Twin Delayed DDPG) Agent.
    
    Key Innovations:
    1. Twin Critics - Use minimum of two Q-values to reduce overestimation
    2. Delayed Policy Updates - Update actor less frequently than critics
    3. Target Policy Smoothing - Add noise to target actions
    """
    def __init__(self, state_dim, action_dim, max_action):
        # Actor networks
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=ACTOR_LR)
        
        # Critic networks
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=CRITIC_LR)
        
        self.max_action = max_action
        self.total_it = 0  # Total training iterations

    def select_action(self, state):
        """Select action using actor network (deterministic)"""
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, iterations):
        """
        Train TD3 agent for given number of iterations.
        
        TD3 Training Process:
        1. Sample batch from replay buffer
        2. Compute target Q-values with noise (target policy smoothing)
        3. Update critics using MSE loss
        4. Every policy_freq iterations, update actor and target networks
        """
        if len(replay_buffer) < BATCH_SIZE:
            return 0.0
        
        total_critic_loss = 0.0
        
        for it in range(iterations):
            self.total_it += 1
            
            # Step 1: Sample batch
            batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = replay_buffer.sample(BATCH_SIZE)
            state = torch.FloatTensor(batch_states).to(device)
            next_state = torch.FloatTensor(batch_next_states).to(device)
            action = torch.FloatTensor(batch_actions).to(device)
            reward = torch.FloatTensor(batch_rewards).to(device)
            done = torch.FloatTensor(batch_dones).to(device)

            # Step 2: Get next action from actor target
            next_action = self.actor_target(next_state)

            # Step 3: Add clipped noise to next action (target policy smoothing)
            noise = torch.FloatTensor(batch_actions).data.normal_(0, POLICY_NOISE).to(device)
            noise = noise.clamp(-NOISE_CLIP, NOISE_CLIP)
            next_action = (next_action + noise).clamp(-self.max_action, self.max_action)

            # Step 4: Compute target Q-values using twin critics
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            
            # Step 5: Take minimum of two Q-values (reduces overestimation)
            target_Q = torch.min(target_Q1, target_Q2)
            
            # Step 6: Compute TD target
            target_Q = reward + ((1 - done) * GAMMA * target_Q).detach()

            # Step 7: Get current Q-values
            current_Q1, current_Q2 = self.critic(state, action)

            # Step 8: Compute critic loss
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
            total_critic_loss += critic_loss.item()

            # Step 9: Update critics
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Step 10: Delayed policy updates
            if self.total_it % POLICY_FREQ == 0:
                # Compute actor loss (maximize Q1)
                actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
                
                # Update actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Step 11: Soft update target networks (Polyak averaging)
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
        
        return total_critic_loss / max(iterations, 1)

    def save(self, filename, directory):
        """Save model weights"""
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save(self.actor.state_dict(), f'{directory}/{filename}_actor.pth')
        torch.save(self.critic.state_dict(), f'{directory}/{filename}_critic.pth')

    def load(self, filename, directory):
        """Load model weights"""
        self.actor.load_state_dict(torch.load(f'{directory}/{filename}_actor.pth'))
        self.critic.load_state_dict(torch.load(f'{directory}/{filename}_critic.pth'))

# ==========================================
# 6. CAR ENVIRONMENT (Physics & State)
# ==========================================
class CarEnvironment:
    """
    Car Navigation Environment with continuous steering control.
    
    State Space: 9 dimensions
        - 7 sensor readings (normalized distance to obstacles)
        - 1 angle to target (normalized -1 to 1)
        - 1 distance to target (normalized 0 to 1)
    
    Action Space: 1 dimension (continuous)
        - Steering angle from -1.0 (full left) to +1.0 (full right)
        - Scaled to -MAX_TURN to +MAX_TURN degrees
    """
    def __init__(self, map_image: QImage):
        self.map = map_image
        self.w, self.h = map_image.width(), map_image.height()
        self.map_diagonal = math.sqrt(self.w**2 + self.h**2)
        
        # State/Action dimensions
        self.state_dim = 9  # 7 sensors + angle_to_target + distance_to_target
        self.action_dim = 1  # Continuous steering angle
        self.max_action = 1.0  # Action range: -1 to +1
        
        # TD3 Agent
        self.agent = TD3(self.state_dim, self.action_dim, self.max_action)
        self.replay_buffer = ReplayBuffer(max_size=200000)  # Larger for complex maps
        
        # Training stats
        self.total_timesteps = 0
        self.episode_num = 0
        self.consecutive_crashes = 0
        
        # Locations
        self.start_pos = QPointF(100, 100) 
        self.car_pos = QPointF(100, 100)   
        self.car_angle = 0
        self.target_pos = QPointF(200, 200)
        
        # Multiple Targets Support
        self.targets = []
        self.current_target_idx = 0
        self.targets_reached = 0
        
        self.alive = True
        self.score = 0
        self.sensor_coords = [] 
        self.prev_dist = None
        self.episode_steps = 0

    def set_start_pos(self, point):
        self.start_pos = point
        self.car_pos = point

    def reset(self):
        """Reset environment for new episode"""
        self.alive = True
        self.score = 0
        self.episode_steps = 0
        self.car_pos = QPointF(self.start_pos.x(), self.start_pos.y())
        self.car_angle = random.randint(0, 360)
        self.current_target_idx = 0
        self.targets_reached = 0
        if len(self.targets) > 0:
            self.target_pos = self.targets[0]
        state, dist = self.get_state()
        self.prev_dist = dist
        return state
    
    def add_target(self, point):
        """Add a target to the sequence"""
        self.targets.append(QPointF(point.x(), point.y()))
        if len(self.targets) == 1:
            self.target_pos = self.targets[0]
            self.current_target_idx = 0
    
    def switch_to_next_target(self):
        """Move to next target in sequence"""
        if self.current_target_idx < len(self.targets) - 1:
            self.current_target_idx += 1
            self.target_pos = self.targets[self.current_target_idx]
            self.targets_reached += 1
            return True
        return False

    def get_state(self):
        """
        Get current state observation.
        
        Returns:
            state: numpy array of shape (9,)
            distance: raw distance to target
        """
        sensor_vals = []
        self.sensor_coords = []
        # 7 sensors with SENSOR_ANGLE spacing
        angles = [(i - 3) * SENSOR_ANGLE for i in range(7)]
        
        for a in angles:
            rad = math.radians(self.car_angle + a)
            
            # Ray-casting to find obstacle distance
            obstacle_dist = SENSOR_DIST
            step_size = 5
            
            for dist in range(step_size, SENSOR_DIST + 1, step_size):
                check_x = self.car_pos.x() + math.cos(rad) * dist
                check_y = self.car_pos.y() + math.sin(rad) * dist
                
                if not (0 <= check_x < self.w and 0 <= check_y < self.h):
                    obstacle_dist = dist
                    break
                    
                c = QColor(self.map.pixel(int(check_x), int(check_y)))
                brightness = (c.red() + c.green() + c.blue()) / 3.0 / 255.0
                
                if brightness < 0.4:
                    obstacle_dist = dist
                    break
            
            # Store endpoint for visualization
            sx = self.car_pos.x() + math.cos(rad) * obstacle_dist
            sy = self.car_pos.y() + math.sin(rad) * obstacle_dist
            self.sensor_coords.append(QPointF(sx, sy))
            
            # Normalized sensor value
            val = obstacle_dist / SENSOR_DIST
            sensor_vals.append(val)
            
        # Distance and angle to target
        dx = self.target_pos.x() - self.car_pos.x()
        dy = self.target_pos.y() - self.car_pos.y()
        dist = math.sqrt(dx*dx + dy*dy)
        
        rad_to_target = math.atan2(dy, dx)
        angle_to_target = math.degrees(rad_to_target)
        
        angle_diff = (angle_to_target - self.car_angle) % 360
        if angle_diff > 180: 
            angle_diff -= 360
        
        norm_dist = min(dist / self.map_diagonal, 1.0)
        norm_angle = angle_diff / 180.0
        
        state = sensor_vals + [norm_angle, norm_dist]
        return np.array(state, dtype=np.float32), dist

    def step(self, action):
        """
        Execute action in environment.
        
        Args:
            action: Continuous steering value from -1.0 to +1.0
        
        Returns:
            next_state, reward, done
        """
        # Convert continuous action to steering angle
        if isinstance(action, np.ndarray):
            action = action[0]
        
        # Clip action to valid range
        action = np.clip(action, -1.0, 1.0)
        
        # Scale action to turn angle
        turn = action * MAX_TURN
        
        # Update car angle and position
        self.car_angle += turn
        rad = math.radians(self.car_angle)
        
        new_x = self.car_pos.x() + math.cos(rad) * SPEED
        new_y = self.car_pos.y() + math.sin(rad) * SPEED
        self.car_pos = QPointF(new_x, new_y)
        
        # Increment step counter
        self.episode_steps += 1
        
        # Get new state
        next_state, dist = self.get_state()
        
        # Compute reward - FOCUS ON REACHING TARGET
        reward = -0.5  # Larger step penalty to discourage going in circles
        done = False
        
        # Check for collision
        car_center_val = self.check_pixel(self.car_pos.x(), self.car_pos.y())
        
        if car_center_val < 0.4:
            # Crashed into obstacle
            reward = -50
            done = True
            self.alive = False
        elif dist < 25:
            # Reached target - PROGRESSIVE BONUS (later targets worth more!)
            target_bonus = 200 + (self.current_target_idx * 100)  # 200, 300, 400, 500...
            reward = target_bonus
            has_next = self.switch_to_next_target()
            if has_next:
                done = False
                _, new_dist = self.get_state()
                self.prev_dist = new_dist
                # Slight angle randomization to encourage different approaches
                self.car_angle += random.uniform(-15, 15)
            else:
                # Completed all targets - HUGE BONUS!
                reward += 500
                done = True
        else:
            # STRONG distance-based reward (most important signal)
            if self.prev_dist is not None:
                progress = self.prev_dist - dist  # Positive if getting closer
                reward += progress * 5  # Scale up the progress reward
                
                # Bonus for significant progress
                if progress > 1:
                    reward += 2
            
            # Small reward for being on road (but not dominant)
            center_sensor = next_state[3]
            if center_sensor > 0.5:
                reward += 0.5  # Minor bonus for clear path
            
            # Penalize being far from target direction
            angle_to_target = next_state[7]  # Normalized angle
            if abs(angle_to_target) > 0.5:  # More than 90 degrees off
                reward -= 1
            
            self.prev_dist = dist
        
        # Timeout for very long episodes (anti-circle measure)
        if self.episode_steps > 2000:
            reward -= 50
            done = True
            
        self.score += reward
        return next_state, reward, done

    def check_pixel(self, x, y):
        """Check brightness at pixel location"""
        if 0 <= x < self.w and 0 <= y < self.h:
            c = QColor(self.map.pixel(int(x), int(y)))
            return ((c.red() + c.green() + c.blue()) / 3.0) / 255.0
        return 0.0

# ==========================================
# 7. CUSTOM WIDGETS (VISUALS)
# ==========================================
class RewardChart(QWidget):
    """Widget for displaying reward history"""
    def __init__(self):
        super().__init__()
        self.setMinimumHeight(150)
        self.setStyleSheet(f"background-color: {C_PANEL.name()}; border-radius: 5px;")
        self.scores = []
        self.max_points = 50

    def update_chart(self, new_score):
        self.scores.append(new_score)
        if len(self.scores) > self.max_points:
            self.scores.pop(0)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        w = self.width()
        h = self.height()
        
        painter.fillRect(0, 0, w, h, C_PANEL)
        
        if len(self.scores) < 2:
            return

        min_val = min(self.scores)
        max_val = max(self.scores)
        if max_val == min_val: 
            max_val += 1
        
        points = []
        step_x = w / (self.max_points - 1)
        
        for i, score in enumerate(self.scores):
            x = i * step_x
            ratio = (score - min_val) / (max_val - min_val)
            y = h - (ratio * (h * 0.8) + (h * 0.1))
            points.append(QPointF(x, y))

        # Draw raw scores
        path = QPainterPath()
        path.moveTo(points[0])
        for p in points[1:]:
            path.lineTo(p)
            
        pen = QPen(C_ACCENT, 2)
        painter.setPen(pen)
        painter.drawPath(path)
        
        # Draw moving average
        if len(self.scores) >= 2:
            avg_points = []
            window_size = 10
            
            for i in range(len(self.scores)):
                start_idx = max(0, i - window_size + 1)
                avg_score = sum(self.scores[start_idx:i+1]) / (i - start_idx + 1)
                
                x = i * step_x
                ratio = (avg_score - min_val) / (max_val - min_val)
                y = h - (ratio * (h * 0.8) + (h * 0.1))
                avg_points.append(QPointF(x, y))
            
            if len(avg_points) > 1:
                avg_path = QPainterPath()
                avg_path.moveTo(avg_points[0])
                for p in avg_points[1:]:
                    avg_path.lineTo(p)
                
                avg_pen = QPen(QColor(255, 215, 0), 3)
                painter.setPen(avg_pen)
                painter.drawPath(avg_path)
        
        # Draw zero line
        if min_val < 0 and max_val > 0:
            zero_ratio = (0 - min_val) / (max_val - min_val)
            y_zero = h - (zero_ratio * (h * 0.8) + (h * 0.1))
            painter.setPen(QPen(QColor(255, 255, 255, 50), 1, Qt.PenStyle.DashLine))
            painter.drawLine(0, int(y_zero), w, int(y_zero))
        
        # Draw legend
        legend_x = 10
        legend_y = 15
        
        painter.setPen(QPen(C_ACCENT, 2))
        painter.drawLine(legend_x, legend_y, legend_x + 20, legend_y)
        painter.setPen(QPen(QColor(200, 200, 200)))
        painter.setFont(QFont("Segoe UI", 9))
        painter.drawText(legend_x + 25, legend_y + 4, "Raw")
        
        painter.setPen(QPen(QColor(255, 215, 0), 3))
        painter.drawLine(legend_x + 60, legend_y, legend_x + 80, legend_y)
        painter.setPen(QPen(QColor(200, 200, 200)))
        painter.drawText(legend_x + 85, legend_y + 4, "Avg (10)")


class SensorItem(QGraphicsItem):
    """Animated sensor dot"""
    def __init__(self):
        super().__init__()
        self.setZValue(90)
        self.pulse = 0
        self.pulse_speed = 0.3
        self.is_detecting = True
        
    def set_detecting(self, detecting):
        self.is_detecting = detecting
        self.update()
    
    def boundingRect(self):
        return QRectF(-4, -4, 8, 8)
    
    def paint(self, painter, option, widget):
        self.pulse += self.pulse_speed
        if self.pulse > 1.0:
            self.pulse = 0
        
        if self.is_detecting:
            color = C_SENSOR_ON
            outer_alpha = int(150 * (1 - self.pulse))
        else:
            color = C_SENSOR_OFF
            outer_alpha = int(200 * (1 - self.pulse))
        
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        outer_size = 3 + (2 * self.pulse)
        outer_color = QColor(color)
        outer_color.setAlpha(outer_alpha)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(outer_color))
        painter.drawEllipse(QPointF(0, 0), outer_size, outer_size)
        
        painter.setBrush(QBrush(color))
        painter.drawEllipse(QPointF(0, 0), 2, 2)


class CarItem(QGraphicsItem):
    """Car visualization item"""
    def __init__(self):
        super().__init__()
        self.setZValue(100)
        self.brush = QBrush(C_ACCENT)
        self.pen = QPen(Qt.GlobalColor.white, 1)

    def boundingRect(self):
        return QRectF(-CAR_WIDTH/2, -CAR_HEIGHT/2, CAR_WIDTH, CAR_HEIGHT)

    def paint(self, painter, option, widget):
        painter.setBrush(self.brush)
        painter.setPen(self.pen)
        painter.drawRoundedRect(self.boundingRect(), 2, 2)
        painter.setBrush(Qt.GlobalColor.white)
        painter.drawRect(int(CAR_WIDTH/2)-2, -3, 2, 6)


class TargetItem(QGraphicsItem):
    """Target visualization item"""
    def __init__(self, color=None, is_active=True, number=1):
        super().__init__()
        self.setZValue(50)
        self.pulse = 0
        self.growing = True
        self.color = color if color else QColor(0, 255, 255)
        self.is_active = is_active
        self.number = number

    def set_active(self, active):
        self.is_active = active
        self.update()
    
    def set_color(self, color):
        self.color = color
        self.update()

    def boundingRect(self):
        return QRectF(-20, -20, 40, 40)

    def paint(self, painter, option, widget):
        if self.is_active:
            if self.growing:
                self.pulse += 0.5
                if self.pulse > 10: 
                    self.growing = False
            else:
                self.pulse -= 0.5
                if self.pulse < 0: 
                    self.growing = True
            
            r = 10 + self.pulse
            painter.setPen(Qt.PenStyle.NoPen)
            outer_color = QColor(self.color)
            outer_color.setAlpha(100)
            painter.setBrush(QBrush(outer_color)) 
            painter.drawEllipse(QPointF(0,0), r, r)
            painter.setBrush(QBrush(self.color)) 
            painter.setPen(QPen(Qt.GlobalColor.white, 2))
            painter.drawEllipse(QPointF(0,0), 8, 8)
        else:
            dimmed_color = QColor(self.color)
            dimmed_color.setAlpha(120)
            painter.setPen(QPen(Qt.GlobalColor.white, 1))
            painter.setBrush(QBrush(dimmed_color))
            painter.drawEllipse(QPointF(0,0), 6, 6)
        
        painter.setPen(QPen(Qt.GlobalColor.white))
        painter.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        painter.drawText(QRectF(-10, -10, 20, 20), Qt.AlignmentFlag.AlignCenter, str(self.number))

# ==========================================
# 8. MAIN APPLICATION
# ==========================================
class TD3NavApp(QMainWindow):
    """
    Main Application Window for TD3 Car Navigation.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NeuralNav TD3: Continuous Steering Control")
        self.resize(1300, 850)
        self.setStyleSheet(f"""
            QMainWindow {{ background-color: {C_BG_DARK.name()}; }}
            QLabel {{ color: {C_TEXT.name()}; font-family: Segoe UI; font-size: 13px; }}
            QPushButton {{ background-color: {C_PANEL.name()}; color: white; border: 1px solid {C_INFO_BG.name()}; padding: 8px; border-radius: 4px; }}
            QPushButton:hover {{ background-color: {C_INFO_BG.name()}; }}
            QPushButton:checked {{ background-color: {C_ACCENT.name()}; color: black; }}
            QTextEdit {{ background-color: {C_PANEL.name()}; color: #D8DEE9; border: none; font-family: Consolas; font-size: 11px; }}
            QFrame {{ border: none; }}
        """)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # LEFT PANEL
        panel = QFrame()
        panel.setFixedWidth(280)
        panel.setStyleSheet(f"background-color: {C_BG_DARK.name()};")
        vbox = QVBoxLayout(panel)
        vbox.setSpacing(10)
        
        lbl_title = QLabel("TD3 CONTROLS")
        lbl_title.setStyleSheet("font-weight: bold; font-size: 14px; margin-bottom: 5px;")
        vbox.addWidget(lbl_title)
        
        self.lbl_status = QLabel("1. Click Map -> CAR\n2. Click Map -> TARGET(S)\n   (Multiple clicks for sequence)")
        self.lbl_status.setStyleSheet(f"background-color: {C_INFO_BG.name()}; padding: 10px; border-radius: 5px; color: #E5E9F0;")
        vbox.addWidget(self.lbl_status)

        self.btn_run = QPushButton("▶ START (Space)")
        self.btn_run.setCheckable(True)
        self.btn_run.setEnabled(False) 
        self.btn_run.clicked.connect(self.toggle_training)
        vbox.addWidget(self.btn_run)
        
        self.btn_reset = QPushButton("↺ RESET ALL")
        self.btn_reset.clicked.connect(self.full_reset)
        vbox.addWidget(self.btn_reset)
        
        self.btn_load = QPushButton("📂 LOAD MAP")
        self.btn_load.clicked.connect(self.load_map_dialog)
        vbox.addWidget(self.btn_load)
        
        self.btn_save = QPushButton("💾 SAVE MODEL")
        self.btn_save.clicked.connect(self.save_model)
        vbox.addWidget(self.btn_save)
        
        self.btn_inference = QPushButton("🎬 INFERENCE MODE")
        self.btn_inference.setCheckable(True)
        self.btn_inference.clicked.connect(self.toggle_inference)
        self.btn_inference.setStyleSheet(f"""
            QPushButton {{ background-color: {C_PANEL.name()}; color: white; border: 1px solid {C_INFO_BG.name()}; padding: 8px; border-radius: 4px; }}
            QPushButton:hover {{ background-color: {C_INFO_BG.name()}; }}
            QPushButton:checked {{ background-color: #A3BE8C; color: black; font-weight: bold; }}
        """)
        vbox.addWidget(self.btn_inference)
        
        self.inference_mode = False

        vbox.addSpacing(15)
        vbox.addWidget(QLabel("REWARD HISTORY"))
        self.chart = RewardChart()
        vbox.addWidget(self.chart)

        stats_frame = QFrame()
        stats_frame.setStyleSheet(f"background-color: {C_PANEL.name()}; border-radius: 5px;")
        sf_layout = QGridLayout(stats_frame)
        sf_layout.setContentsMargins(10, 10, 10, 10)
        
        self.val_timesteps = QLabel("0")
        self.val_timesteps.setStyleSheet(f"color: {C_ACCENT.name()}; font-weight: bold;")
        sf_layout.addWidget(QLabel("Timesteps:"), 0, 0)
        sf_layout.addWidget(self.val_timesteps, 0, 1)
        
        self.val_episode = QLabel("0")
        self.val_episode.setStyleSheet(f"color: {C_ACCENT.name()}; font-weight: bold;")
        sf_layout.addWidget(QLabel("Episode:"), 1, 0)
        sf_layout.addWidget(self.val_episode, 1, 1)
        
        self.val_rew = QLabel("0")
        self.val_rew.setStyleSheet(f"color: {C_ACCENT.name()}; font-weight: bold;")
        sf_layout.addWidget(QLabel("Score:"), 2, 0)
        sf_layout.addWidget(self.val_rew, 2, 1)
        
        self.val_action = QLabel("0.00")
        self.val_action.setStyleSheet(f"color: {C_ACCENT.name()}; font-weight: bold;")
        sf_layout.addWidget(QLabel("Action:"), 3, 0)
        sf_layout.addWidget(self.val_action, 3, 1)
        
        vbox.addWidget(stats_frame)

        vbox.addWidget(QLabel("LOGS"))
        self.log_console = QTextEdit()
        self.log_console.setReadOnly(True)
        vbox.addWidget(self.log_console)

        main_layout.addWidget(panel)

        # RIGHT PANEL
        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        self.view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.view.setStyleSheet(f"border: 2px solid {C_PANEL.name()}; background-color: {C_BG_DARK.name()}")
        self.view.mousePressEvent = self.on_scene_click
        main_layout.addWidget(self.view)

        # Logic - Load default map
        self.setup_map("maps/map3.jpg") 
        self.setup_state = 0 
        self.sim_timer = QTimer()
        self.sim_timer.timeout.connect(self.game_loop)
        
        self.car_item = CarItem()
        self.target_items = []
        self.sensor_items = []
        for _ in range(7):
            si = SensorItem()
            self.scene.addItem(si)
            self.sensor_items.append(si)
        
        # Episode tracking
        self.episode_reward = 0
        self.episode_timesteps = 0
        self.current_state = None
        self.last_action = 0.0

    def log(self, msg):
        self.log_console.append(msg)
        sb = self.log_console.verticalScrollBar()
        sb.setValue(sb.maximum())

    def setup_map(self, path):
        if not os.path.exists(path):
            self.create_dummy_map(path)
        self.map_img = QImage(path).convertToFormat(QImage.Format.Format_RGB32)
        self.scene.clear()
        self.scene.addPixmap(QPixmap.fromImage(self.map_img))
        self.env = CarEnvironment(self.map_img)
        self.log(f"Map Loaded. Using TD3 with continuous steering.")
        self.log(f"Device: {device}")

    def create_dummy_map(self, path):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        img = QImage(1000, 800, QImage.Format.Format_RGB32)
        img.fill(C_BG_DARK)
        p = QPainter(img)
        p.setBrush(Qt.GlobalColor.white)
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(100, 100, 800, 600)
        p.setBrush(C_BG_DARK)
        p.drawEllipse(250, 250, 500, 300)
        p.end()
        img.save(path)

    def load_map_dialog(self):
        f, _ = QFileDialog.getOpenFileName(self, "Load Map", "", "Images (*.png *.jpg)")
        if f: 
            self.full_reset()
            self.setup_map(f)

    def save_model(self):
        self.env.agent.save("td3_car", "./models")
        self.log("<font color='#A3BE8C'>Model saved to ./models/</font>")

    def toggle_inference(self):
        self.inference_mode = self.btn_inference.isChecked()
        if self.inference_mode:
            self.log("<font color='#A3BE8C'><b>🎬 INFERENCE MODE ON - No noise, no training, weights frozen</b></font>")
            self.btn_inference.setText("🎬 INFERENCE ON")
        else:
            self.log("<font color='#88C0D0'><b>📚 TRAINING MODE ON - Exploration + learning resumed</b></font>")
            self.btn_inference.setText("🎬 INFERENCE MODE")

    def on_scene_click(self, event):
        pt = self.view.mapToScene(event.pos())
        if self.setup_state == 0:
            self.env.set_start_pos(pt) 
            self.scene.addItem(self.car_item)
            self.car_item.setPos(pt)
            self.setup_state = 1
            self.lbl_status.setText("Click Map -> TARGET(S)\nRight-click when done")
        elif self.setup_state == 1:
            if event.button() == Qt.MouseButton.LeftButton:
                self.env.add_target(pt)
                target_idx = len(self.env.targets) - 1
                color = TARGET_COLORS[target_idx % len(TARGET_COLORS)]
                is_active = (target_idx == 0)
                num_targets = len(self.env.targets)
                
                target_item = TargetItem(color, is_active, num_targets)
                target_item.setPos(pt)
                self.scene.addItem(target_item)
                self.target_items.append(target_item)
                
                self.lbl_status.setText(f"Targets: {num_targets}\nRight-click to finish setup")
                self.log(f"Target #{num_targets} added at ({pt.x():.0f}, {pt.y():.0f})")
            
            elif event.button() == Qt.MouseButton.RightButton:
                if len(self.env.targets) > 0:
                    self.setup_state = 2
                    self.lbl_status.setText(f"READY. {len(self.env.targets)} target(s). Press SPACE.")
                    self.lbl_status.setStyleSheet(f"background-color: {C_SUCCESS.name()}; color: #2E3440; font-weight: bold; padding: 10px; border-radius: 5px;")
                    self.btn_run.setEnabled(True)
                    # Initialize first state
                    self.current_state = self.env.reset()
                    self.update_visuals()

    def full_reset(self):
        self.sim_timer.stop()
        self.btn_run.setChecked(False)
        self.btn_run.setEnabled(False)
        self.setup_state = 0
        self.scene.removeItem(self.car_item)
        for target_item in self.target_items:
            self.scene.removeItem(target_item)
        self.target_items = []
        self.env.targets = []
        self.env.current_target_idx = 0
        self.env.targets_reached = 0
        self.env.total_timesteps = 0
        self.env.episode_num = 0
        
        for s in self.sensor_items: 
            if s.scene() == self.scene: 
                self.scene.removeItem(s)
        self.lbl_status.setText("1. Click Map -> CAR\n2. Click Map -> TARGET(S)")
        self.lbl_status.setStyleSheet(f"background-color: {C_INFO_BG.name()}; color: white; padding: 10px; border-radius: 5px;")
        self.log("--- RESET ---")
        self.chart.scores = []
        self.chart.update()

    def toggle_training(self):
        if self.btn_run.isChecked():
            self.sim_timer.start(16)
            self.btn_run.setText("⏸ PAUSE")
        else:
            self.sim_timer.stop()
            self.btn_run.setText("▶ RESUME")

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Space and self.setup_state == 2:
            self.btn_run.click()

    def game_loop(self):
        """Main training loop - one step per call"""
        if self.setup_state != 2: 
            return

        # Get current state
        if self.current_state is None:
            self.current_state = self.env.reset()
        
        state = self.current_state
        prev_target_idx = self.env.current_target_idx
        
        # Select action
        if self.inference_mode:
            # INFERENCE MODE: Pure policy, no noise
            action = self.env.agent.select_action(state)
        elif self.env.total_timesteps < START_TIMESTEPS:
            # Random exploration at the start
            action = np.random.uniform(-1, 1, size=(1,))
        else:
            # Use policy with exploration noise
            action = self.env.agent.select_action(state)
            action = action + np.random.normal(0, EXPL_NOISE, size=action.shape)
            action = np.clip(action, -1, 1)
        
        self.last_action = action[0] if isinstance(action, np.ndarray) else action

        # Execute action
        next_state, reward, done = self.env.step(action)
        
        # Store transition (skip in inference mode)
        if not self.inference_mode:
            done_bool = float(done) if self.episode_timesteps + 1 < 1000 else 0
            self.env.replay_buffer.add((state, next_state, action, reward, done_bool))
        
        # Update state
        self.current_state = next_state
        self.episode_reward += reward
        self.episode_timesteps += 1
        if not self.inference_mode:
            self.env.total_timesteps += 1
        
        # Train agent (after initial exploration, skip in inference mode)
        if not self.inference_mode and self.env.total_timesteps >= START_TIMESTEPS:
            self.env.agent.train(self.env.replay_buffer, 1)
        
        # Check if target changed
        if self.env.current_target_idx != prev_target_idx:
            target_num = self.env.current_target_idx + 1
            total = len(self.env.targets)
            self.log(f"<font color='#88C0D0'>🎯 Target {prev_target_idx + 1} reached! Moving to target {target_num}/{total}</font>")
        
        # Episode done
        if done:
            self.env.episode_num += 1
            
            # Track crashes
            if not self.env.alive:
                self.env.consecutive_crashes += 1
                txt = f"CRASH ({self.env.consecutive_crashes}/{MAX_CONSECUTIVE_CRASHES})"
                col = "#BF616A"
            else:
                self.env.consecutive_crashes = 0
                if self.env.targets_reached == len(self.env.targets) - 1:
                    txt = f"ALL {len(self.env.targets)} TARGETS COMPLETED!"
                else:
                    txt = "GOAL"
                col = "#A3BE8C"
            
            # Log episode
            self.log(f"<font color='{col}'>{txt} | Ep: {self.env.episode_num} | Steps: {self.episode_timesteps} | Score: {self.episode_reward:.0f} | Buffer: {len(self.env.replay_buffer)}</font>")
            
            # Update chart
            self.chart.update_chart(self.episode_reward)
            
            # Reset for next episode
            if self.env.consecutive_crashes >= MAX_CONSECUTIVE_CRASHES:
                self.log(f"<font color='#BF616A'><b>⚠️ {MAX_CONSECUTIVE_CRASHES} consecutive crashes! Resetting...</b></font>")
                self.env.consecutive_crashes = 0
            
            self.current_state = self.env.reset()
            self.episode_reward = 0
            self.episode_timesteps = 0

        # Update visuals
        self.update_visuals()
        self.val_timesteps.setText(str(self.env.total_timesteps))
        self.val_episode.setText(str(self.env.episode_num))
        self.val_rew.setText(f"{self.env.score:.0f}")
        self.val_action.setText(f"{self.last_action:.2f}")

    def update_visuals(self):
        self.car_item.setPos(self.env.car_pos)
        self.car_item.setRotation(self.env.car_angle)
        
        for i, target_item in enumerate(self.target_items):
            is_active = (i == self.env.current_target_idx)
            target_item.set_active(is_active)
        
        self.scene.update() 
        
        for i, coord in enumerate(self.env.sensor_coords):
            self.sensor_items[i].setPos(coord)
            state, _ = self.env.get_state()
            s_val = state[i]
            self.sensor_items[i].set_detecting(s_val > 0.5)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = TD3NavApp()
    win.show()
    sys.exit(app.exec())

