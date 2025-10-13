"""
DQN (Deep Q-Network) Agent
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random


class QNetwork(nn.Module):
    """
    Neural network for Q-function approximation.
    
    Q-function Q(s, a) returns the expected total return after taking action a in state s.
    """
    
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        """
        Initialize the Q-network.
        
        Args:
            state_dim: State space dimension
            action_dim: Number of actions
            hidden_dim: Hidden layer dimension
        """
        super(QNetwork, self).__init__()
        
        # Fully connected neural network
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, state):
        """
        Forward pass through the network.
        
        Args:
            state: Environment state
            
        Returns:
            Q-values for all actions
        """
        return self.network(state)


class ReplayBuffer:
    """
    Replay buffer for storing experiences.
    
    DQN uses experience replay for:
    - Breaking correlation between consecutive samples
    - Better data utilization (each experience is used multiple times)
    """
    
    def __init__(self, capacity=10000):
        """
        Initialize the buffer.
        
        Args:
            capacity: Maximum number of stored experiences
        """
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """
        Add experience to the buffer.
        
        Args:
            state: Current state
            action: Performed action
            reward: Received reward
            next_state: Next state
            done: Whether the episode ended
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """
        Randomly sample a batch of experiences.
        
        Args:
            batch_size: Number of samples
            
        Returns:
            Batch of experiences (states, actions, rewards, next_states, dones)
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )
    
    def __len__(self):
        """Return the current buffer size."""
        return len(self.buffer)


class DQNAgent:
    """
    DQN Agent for learning.
    
    Key DQN components:
    1. Q-network (q_network): Current network for action selection
    2. Target network (target_network): Stable network for target computation
    3. Replay buffer: Memory of experiences
    4. Epsilon-greedy: Exploration vs exploitation
    """
    
    def __init__(
        self,
        state_dim,
        action_dim,
        learning_rate=0.001,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        buffer_capacity=10000,
        batch_size=64,
        target_update_freq=10,
        device=None
    ):
        """
        Initialize DQN agent.
        
        Args:
            state_dim: State space dimension
            action_dim: Number of actions
            learning_rate: Learning rate
            gamma: Discount factor (importance of future rewards)
            epsilon_start: Initial epsilon for exploration
            epsilon_end: Minimum epsilon
            epsilon_decay: Epsilon decay rate
            buffer_capacity: Replay buffer size
            batch_size: Batch size for training
            target_update_freq: How often to update target network
            device: Device for computations (CPU/GPU)
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        # Q-network and target network
        self.q_network = QNetwork(state_dim, action_dim).to(self.device)
        self.target_network = QNetwork(state_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        
        # Counters
        self.steps = 0
        self.episodes = 0
        
        print(f"DQN Agent initialized")
        print(f"  Device: {self.device}")
        print(f"  State dim: {state_dim}")
        print(f"  Action dim: {action_dim}")
        print(f"  Gamma: {gamma}")
        print(f"  Learning rate: {learning_rate}")
    
    def select_action(self, state, training=True):
        """
        Select action using epsilon-greedy strategy.
        
        Args:
            state: Current state
            training: Whether we are in training mode
            
        Returns:
            Selected action
        """
        # Epsilon-greedy exploration
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        
        # Greedy action according to Q-network
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def store_transition(self, state, action, reward, next_state, done):
        """
        Store experience in replay buffer.
        
        Args:
            state: Current state
            action: Performed action
            reward: Received reward
            next_state: Next state
            done: Whether the episode ended
        """
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def update(self):
        """
        Update Q-network using a minibatch from replay buffer.
        
        Returns:
            Loss value or None if not enough data
        """
        # Skip if we don't have enough samples
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Current Q-values: Q(s, a)
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Target Q-values: r + gamma * max_a' Q_target(s', a')
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Loss (MSE between current and target Q)
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        # Update target network
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        return loss.item()
    
    def save(self, path):
        """
        Save the model.
        
        Args:
            path: File path
        """
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps,
            'episodes': self.episodes
        }, path)
        print(f"Model saved to {path}")
    
    def load(self, path):
        """
        Load the model.
        
        Args:
            path: File path
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']
        self.episodes = checkpoint['episodes']
        print(f"Model loaded from {path}")
