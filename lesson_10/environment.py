"""
CartPole Environment for Reinforcement Learning
"""
import gymnasium as gym
import numpy as np


class CartPoleEnvironment:
    """
    Wrapper for CartPole-v1 environment.
    
    CartPole is a classic reinforcement learning problem:
    - Goal: Keep the pole balanced in an upright position as long as possible
    - States (4D): [cart position, cart velocity, pole angle, pole angular velocity]
    - Actions (2D): [0=left, 1=right]
    - Reward: +1 for each step where the pole is upright
    - Episode termination: Pole falls (angle > 15°) or cart moves too far
    """
    
    def __init__(self, render_mode=None):
        """
        Initialize the environment.
        
        Args:
            render_mode: Rendering mode ('human', 'rgb_array', None)
        """
        self.env = gym.make('CartPole-v1', render_mode=render_mode)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        
        # Statistics
        self.episode_count = 0
        self.total_steps = 0
        
    def reset(self, seed=None):
        """
        Reset the environment to initial state.
        
        Args:
            seed: Random seed for reproducibility
            
        Returns:
            observation: Initial state
            info: Additional information
        """
        observation, info = self.env.reset(seed=seed)
        self.episode_count += 1
        return observation, info
    
    def step(self, action):
        """
        Execute an action in the environment.
        
        Args:
            action: Action (0 or 1)
            
        Returns:
            observation: New state
            reward: Reward
            terminated: Whether the episode ended
            truncated: Whether the time limit was reached
            info: Additional information
        """
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.total_steps += 1
        return observation, reward, terminated, truncated, info
    
    def close(self):
        """Close the environment."""
        self.env.close()
    
    def get_state_dim(self):
        """Return the state space dimension."""
        return self.observation_space.shape[0]
    
    def get_action_dim(self):
        """Return the number of possible actions."""
        return self.action_space.n


def test_environment():
    """Test the environment with random actions."""
    print("=" * 60)
    print("CartPole Environment Test")
    print("=" * 60)
    
    env = CartPoleEnvironment()
    
    print(f"\nState dimensions: {env.get_state_dim()}")
    print(f"Number of actions: {env.get_action_dim()}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Test several episodes
    num_episodes = 3
    for episode in range(num_episodes):
        observation, info = env.reset(seed=42 + episode)
        total_reward = 0
        steps = 0
        
        print(f"\n--- Episode {episode + 1} ---")
        print(f"Initial state: {observation}")
        
        while True:
            # Random action
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            steps += 1
            
            if terminated or truncated:
                break
        
        print(f"Total reward: {total_reward}")
        print(f"Number of steps: {steps}")
    
    env.close()
    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)


if __name__ == "__main__":
    test_environment()

