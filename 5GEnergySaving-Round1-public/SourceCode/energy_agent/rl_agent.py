import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import logging
import os
from datetime import datetime

from .transition import Transition, TransitionBuffer
from .models import Actor
from .models import Critic
from .state_normalizer import StateNormalizer


class RLAgent:
    def __init__(self, n_cells, n_ues, max_time, log_file='rl_agent.log', use_gpu=False):
        """
        Initialize PPO agent for 5G energy saving
        
        Args:
            n_cells (int): Number of cells to control
            n_ues (int): Number of UEs in network
            max_time (int): Maximum simulation time steps
            log_file (str): Path to log file
            use_gpu (bool): Whether to use GPU acceleration
        """
        print("Initializing RL Agent")
        self.n_cells = n_cells
        self.n_ues = n_ues
        self.max_time = max_time
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_gpu else 'cpu')
        
        # State dimensions: 17 simulation features + 14 network features + (n_cells * 12) cell features
        self.state_dim = 17 + 14 + (n_cells * 12)
        self.action_dim = n_cells  # Power ratio for each cell
        
        # Normalization parameters - learned from data
        self.state_normalizer = StateNormalizer(self.state_dim, n_cells=n_cells)
        
        self.actor = Actor(self.state_dim, self.action_dim, hidden_dim=256).to(self.device)
        self.critic = Critic(self.state_dim, hidden_dim=256).to(self.device)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4, eps=1e-5)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3, eps=1e-5)
        
        # PPO hyperparameters - Optimized for 5G energy saving
        self.gamma = 0.998  # Higher discount for long-term energy efficiency
        self.lambda_gae = 0.97  # Better bias-variance trade-off
        self.clip_epsilon = 0.15  # More conservative clipping for stability
        self.ppo_epochs = 12  # More updates per batch for better learning
        self.batch_size = 128  # Larger batch for stable gradients
        self.buffer_size = 4096  # Larger buffer for diverse experiences
        
        
        # Experience buffer
        self.buffer = TransitionBuffer(self.buffer_size)
        
        self.training_mode = True
        self.total_episodes = 0
        self.total_steps = 0
        self.episode_rewards = deque(maxlen=100)
        self.episode_steps = 0
        self.current_episode_reward = 0.0
        
        self.setup_logging(log_file)
        
        self.logger.info(f"PPO Agent initialized: {n_cells} cells, {n_ues} UEs")
        self.logger.info(f"State dim: {self.state_dim}, Action dim: {self.action_dim}")
        self.logger.info(f"Device: {self.device}")
    
    def normalize_state(self, state):
        """Normalize state vector to [0, 1] range"""
        return self.state_normalizer.normalize(state)
    
    def setup_logging(self, log_file):
        """Setup logging configuration"""
        self.logger = logging.getLogger('PPOAgent')
        self.logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def start_scenario(self):
        self.total_episodes += 1
        self.episode_steps = 0
        self.current_episode_reward = 0.0
        self.logger.info(f"Starting episode {self.total_episodes}")
    
    def end_scenario(self):
        self.episode_rewards.append(self.current_episode_reward)
        avg_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0.0
        
        self.logger.info(f"Episode {self.total_episodes} ended: "
                        f"Steps={self.episode_steps}, "
                        f"Reward={self.current_episode_reward:.2f}, "
                        f"Avg100={avg_reward:.2f}")
        
        # Train if buffer has enough experiences
        if self.training_mode and len(self.buffer) >= self.batch_size:
            self.train()
    
    # NOT REMOVED FOR INTERACTING WITH SIMULATION (CAN BE MODIFIED)
    def get_action(self, state):
        """
        Get action from policy network
        
        Args:
            state: State vector from MATLAB interface
            
        Returns:
            action: Power ratios for each cell [0, 1]
        """
        state = self.normalize_state(np.array(state).flatten())  # make sure it’s 1D
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_mean, action_logstd = self.actor(state_tensor)
            
            if self.training_mode:
                # Sample from policy during training
                action_std = torch.exp(action_logstd)
                dist = torch.distributions.Normal(action_mean, action_std)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(-1)
            else:
                # Use mean during evaluation
                action = action_mean
                log_prob = torch.zeros(1).to(self.device)
        
        # Clamp actions to [0, 1] range
        action = torch.clamp(action, 0.0, 1.0)
        
        # Store for experience replay
        self.last_state = state_tensor.cpu().numpy().flatten()
        self.last_action = action.cpu().numpy().flatten()
        self.last_log_prob = log_prob.cpu().numpy().flatten()
        
        return action.cpu().numpy().flatten()
    
    ## OPTIONAL: Modify reward calculation as needed
    def calculate_reward(self, prev_state, action, current_state):
        """Calculate reward based on energy savings and KPI constraints"""
        if prev_state is None:
            return 0.0
        
        # Convert to numpy arrays for consistent indexing
        prev_state = np.array(prev_state).flatten()
        current_state = np.array(current_state).flatten()
        
        # Extract state components
        current_simulation_start = 0
        current_network_start = 17  # After simulation features
        drop_call_threshold = current_state[11]  # From simulation features
        latency_threshold = current_state[12]    # From simulation features
        
        # Current state metrics
        current_energy = current_state[current_network_start + 0]
        current_connected_ues = current_state[current_simulation_start + 5]
        current_drop_rate = current_state[current_simulation_start + 2]
        current_latency = current_state[current_simulation_start + 3]
        current_cpu_violations = current_state[current_network_start + 6]
        current_prb_violations = current_state[current_network_start + 7]
        
        # Previous state metrics for comparison
        prev_energy = prev_state[current_network_start + 0]
        prev_connected_ues = prev_state[current_simulation_start + 5]
        prev_drop_rate = prev_state[current_simulation_start + 2]
        prev_latency = prev_state[current_simulation_start + 3]
        
        # === 1. ENERGY EFFICIENCY (Primary objective) ===
        # Relative energy improvement with diminishing returns
        if prev_energy > 0:
            energy_efficiency = (prev_energy - current_energy) / prev_energy
            energy_reward = np.tanh(energy_efficiency * 6) * 10  # Smooth scaling
        else:
            energy_reward = 0
        
        # === 2. HARD CONSTRAINTS (Critical penalties) ===
        # Exponential penalties for violations - ensure compliance
        drop_violation = max(0, current_drop_rate - drop_call_threshold)
        drop_penalty = -50 * (drop_violation / max(drop_call_threshold, 1e-6)) ** 2 if drop_violation > 0 else 0
        
        latency_violation = max(0, current_latency - latency_threshold)
        latency_penalty = -30 * (latency_violation / max(latency_threshold, 1e-6)) ** 2 if latency_violation > 0 else 0
        
        # Resource violations - severe penalties
        cpu_penalty = -100 * current_cpu_violations if current_cpu_violations > 0 else 0
        prb_penalty = -100 * current_prb_violations if current_prb_violations > 0 else 0
        
        # === 3. QoS STABILITY (Smooth operations) ===
        # Penalize large connection fluctuations
        connection_stability = -abs(current_connected_ues - prev_connected_ues) * 0.2
        
        # Reward steady improvements in performance
        drop_improvement = np.clip((prev_drop_rate - current_drop_rate) * 5, -2, 5)
        latency_improvement = np.clip((prev_latency - current_latency) * 0.2, -2, 3)
        
        # === 4. ACTION REGULARIZATION ===
        # Encourage smooth, efficient actions
        action_smoothness = -np.sum(np.abs(np.diff(action))) * 0.1  # Penalize abrupt changes
        action_efficiency = np.sum(action) * 0.1  # Small bonus for conservative power usage
        
        # === 5. ADAPTIVE WEIGHTING ===
        # Adjust weights based on current QoS status
        qos_health = 1.0
        if current_drop_rate > drop_call_threshold * 0.8:  # Near violation
            qos_health *= 0.5  # Reduce energy priority
        if current_latency > latency_threshold * 0.8:
            qos_health *= 0.5
            
        # === TOTAL REWARD CALCULATION ===
        constraint_penalty = drop_penalty + latency_penalty + cpu_penalty + prb_penalty
        performance_reward = (drop_improvement + latency_improvement + connection_stability)
        action_reward = action_smoothness + action_efficiency
        
        # Weighted combination with adaptive scaling
        total_reward = (
            energy_reward * qos_health * 0.4 +  # Primary objective
            constraint_penalty +                 # Hard constraints
            performance_reward * 0.3 +          # QoS improvements
            action_reward * 0.1                 # Action regularization
        )
        
        # Enhanced logging for debugging
        if self.total_steps % 50 == 0:  # Log every 50 steps
            self.logger.info(f"Reward breakdown - Energy: {energy_reward:.2f}, "
                           f"Constraints: {constraint_penalty:.2f}, Performance: {performance_reward:.2f}, "
                           f"Actions: {action_reward:.2f}, QoS_health: {qos_health:.2f}")
        
        # Smooth clipping for better gradient flow
        return float(np.tanh(total_reward / 20) * 15)
    
    # NOT REMOVED FOR INTERACTING WITH SIMULATION (CAN BE MODIFIED)
    def update(self, state, action, next_state, done):
        """
        Update agent with experience
        
        Args:
            state: Previous state
            action: Action taken
            next_state: Next state
            done: Whether episode is done
        """
        if not self.training_mode:
            return
        
        # Calculate actual reward using state as prev_state and next_state as current
        actual_reward = self.calculate_reward(state, action, next_state)

        self.episode_steps += 1
        self.total_steps += 1
        self.current_episode_reward += actual_reward
        
        # Convert inputs to numpy if needed
        if hasattr(state, 'numpy'):
            state = state.numpy()
        if hasattr(action, 'numpy'):
            action = action.numpy()
        if hasattr(next_state, 'numpy'):
            next_state = next_state.numpy()
        
        # Ensure proper shapes
        state = self.normalize_state(np.array(state).flatten())
        action = np.array(action).flatten()
        next_state = self.normalize_state(np.array(next_state).flatten())
        
        # Get value estimates
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            value = self.critic(state_tensor).cpu().numpy().flatten()[0]
            next_value = self.critic(next_state_tensor).cpu().numpy().flatten()[0]
        
        # Create transition
        transition = Transition(
            state=state,
            action=action,
            reward=actual_reward,
            next_state=next_state,
            done=done,
            log_prob=getattr(self, 'last_log_prob', np.array([0.0]))[0],
            value=value
        )
        
        self.buffer.add(transition)
    
    def compute_gae(self, rewards, values, next_values, dones):
        """Compute Generalized Advantage Estimation"""
        advantages = np.zeros_like(rewards)
        last_advantage = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_value = next_values[t]
            else:
                next_non_terminal = 1.0 - dones[t]
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            advantages[t] = last_advantage = delta + self.gamma * self.lambda_gae * next_non_terminal * last_advantage
        
        returns = advantages + values
        return advantages, returns
    
    def train(self):
        """Train the PPO agent"""
        if len(self.buffer) < self.batch_size:
            return
        
        # Get all transitions
        transitions = self.buffer.get_all()
        
        states = np.array([t.state for t in transitions])
        actions = np.array([t.action for t in transitions])
        rewards = np.array([t.reward for t in transitions])
        next_states = np.array([t.next_state for t in transitions])
        dones = np.array([t.done for t in transitions])
        old_log_probs = np.array([t.log_prob for t in transitions])
        values = np.array([t.value for t in transitions])
        
        # Compute next values
        next_states_tensor = torch.FloatTensor(next_states).to(self.device)
        with torch.no_grad():
            next_values = self.critic(next_states_tensor).cpu().numpy().flatten()
        
        # Compute advantages and returns
        advantages, returns = self.compute_gae(rewards, values, next_values, dones)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Convert to tensors
        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.FloatTensor(actions).to(self.device)
        old_log_probs_tensor = torch.FloatTensor(old_log_probs).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        
        # PPO training loop
        for epoch in range(self.ppo_epochs):
            # Shuffle data
            indices = torch.randperm(len(states))
            
            for start_idx in range(0, len(states), self.batch_size):
                end_idx = min(start_idx + self.batch_size, len(states))
                batch_indices = indices[start_idx:end_idx]
                
                batch_states = states_tensor[batch_indices]
                batch_actions = actions_tensor[batch_indices]
                batch_old_log_probs = old_log_probs_tensor[batch_indices]
                batch_advantages = advantages_tensor[batch_indices]
                batch_returns = returns_tensor[batch_indices]
                
                # Compute current policy
                action_mean, action_logstd = self.actor(batch_states)
                action_std = torch.exp(action_logstd)
                dist = torch.distributions.Normal(action_mean, action_std)
                new_log_probs = dist.log_prob(batch_actions).sum(-1)
                
                # Compute ratio
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                # Compute surrogate losses
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Critic loss
                current_values = self.critic(batch_states).squeeze()
                critic_loss = nn.MSELoss()(current_values, batch_returns)
                
                # Update actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.actor_optimizer.step()
                
                # Update critic
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.critic_optimizer.step()
        
        # Clear buffer after training
        self.buffer.clear()
        
        self.logger.info(f"Training completed: Actor loss={actor_loss:.4f}, "
                        f"Critic loss={critic_loss:.4f}")
    
    def save_model(self, filepath=None):
        """Save model parameters"""
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"ppo_model_{timestamp}.pth"
        
        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'total_episodes': self.total_episodes,
            'total_steps': self.total_steps
        }
        
        torch.save(checkpoint, filepath)
        self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load model parameters"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        
        self.total_episodes = checkpoint.get('total_episodes', 0)
        self.total_steps = checkpoint.get('total_steps', 0)
        
        self.logger.info(f"Model loaded from {filepath}")
    
    def set_training_mode(self, training):
        """Set training mode"""
        self.training_mode = training
        self.actor.train(training)
        self.critic.train(training)
        self.logger.info(f"Training mode set to {training}")
    
    def get_stats(self):
        """Get training statistics"""
        avg_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0.0
        
        return {
            'total_episodes': self.total_episodes,
            'total_steps': self.total_steps,
            'avg_reward': avg_reward,
            'buffer_size': len(self.buffer),
            'training_mode': self.training_mode,
            'episode_steps': self.episode_steps,
            'current_episode_reward': self.current_episode_reward
        }