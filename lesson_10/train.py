"""
Training DQN agent on CartPole environment
"""
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import json

from environment import CartPoleEnvironment
from dqn_agent import DQNAgent


def train_dqn(
    num_episodes=500,
    max_steps=500,
    save_dir="models",
    save_freq=50,
    plot_freq=50
):
    """
    Train DQN agent on CartPole environment.
    
    Args:
        num_episodes: Number of training episodes
        max_steps: Maximum number of steps per episode
        save_dir: Directory for saving models
        save_freq: How often to save the model
        plot_freq: How often to plot graphs
    """
    # Create directories
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    
    # Initialize environment and agent
    print("=" * 70)
    print("DQN Training - CartPole")
    print("=" * 70)
    
    env = CartPoleEnvironment()
    agent = DQNAgent(
        state_dim=env.get_state_dim(),
        action_dim=env.get_action_dim(),
        learning_rate=0.001,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        buffer_capacity=10000,
        batch_size=64,
        target_update_freq=10
    )
    
    # Statistics for tracking progress
    episode_rewards = []
    episode_lengths = []
    losses = []
    moving_avg_rewards = []
    epsilons = []
    
    # Best model
    best_reward = -float('inf')
    
    print("\nStarting training...")
    print(f"Number of episodes: {num_episodes}")
    print(f"Max steps per episode: {max_steps}")
    print("=" * 70 + "\n")
    
    # Training loop
    for episode in tqdm(range(num_episodes), desc="Training"):
        state, _ = env.reset()
        episode_reward = 0
        episode_loss = []
        
        for step in range(max_steps):
            # Select and execute action
            action = agent.select_action(state, training=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Store experience
            agent.store_transition(state, action, reward, next_state, done)
            
            # Update agent
            loss = agent.update()
            if loss is not None:
                episode_loss.append(loss)
            
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        # Save statistics
        episode_rewards.append(episode_reward)
        episode_lengths.append(step + 1)
        epsilons.append(agent.epsilon)
        
        if episode_loss:
            losses.append(np.mean(episode_loss))
        
        # Moving average
        window = min(100, episode + 1)
        moving_avg = np.mean(episode_rewards[-window:])
        moving_avg_rewards.append(moving_avg)
        
        # Save best model
        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save(os.path.join(save_dir, "best_model.pth"))
        
        # Regular saving
        if (episode + 1) % save_freq == 0:
            agent.save(os.path.join(save_dir, f"model_episode_{episode + 1}.pth"))
        
        # Print statistics
        if (episode + 1) % 50 == 0:
            print(f"\nEpisode {episode + 1}/{num_episodes}")
            print(f"  Reward: {episode_reward:.2f}")
            print(f"  Moving Avg (100): {moving_avg:.2f}")
            print(f"  Epsilon: {agent.epsilon:.4f}")
            print(f"  Best reward: {best_reward:.2f}")
            if losses:
                print(f"  Loss: {losses[-1]:.4f}")
        
        # Plot graphs
        if (episode + 1) % plot_freq == 0:
            plot_training_progress(
                episode_rewards,
                moving_avg_rewards,
                losses,
                epsilons,
                save_path=f"plots/progress_episode_{episode + 1}.png"
            )
    
    # Final save
    agent.save(os.path.join(save_dir, "final_model.pth"))
    agent.episodes = num_episodes
    
    # Save statistics
    stats = {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'moving_avg_rewards': moving_avg_rewards,
        'losses': losses,
        'epsilons': epsilons,
        'best_reward': best_reward
    }
    
    with open(os.path.join(save_dir, 'training_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Final graph
    plot_training_progress(
        episode_rewards,
        moving_avg_rewards,
        losses,
        epsilons,
        save_path="plots/final_training_progress.png"
    )
    
    print("\n" + "=" * 70)
    print("Training completed!")
    print(f"Best reward: {best_reward:.2f}")
    print(f"Final moving average: {moving_avg_rewards[-1]:.2f}")
    print(f"Model saved in: {save_dir}")
    print(f"Plots saved in: plots/")
    print("=" * 70)
    
    env.close()
    
    return agent, stats


def plot_training_progress(rewards, moving_avg, losses, epsilons, save_path=None):
    """
    Plot training progress.
    
    Args:
        rewards: List of episode rewards
        moving_avg: Moving average of rewards
        losses: List of loss values
        epsilons: List of epsilon values
        save_path: Path to save the graph
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('DQN Training Progress - CartPole', fontsize=16, fontweight='bold')
    
    # 1. Rewards
    ax1 = axes[0, 0]
    ax1.plot(rewards, alpha=0.3, label='Episode Reward')
    ax1.plot(moving_avg, label='Moving Average (100)', linewidth=2)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Rewards during Training')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Loss
    ax2 = axes[0, 1]
    if losses:
        ax2.plot(losses)
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Loss')
        ax2.set_title('Training Loss')
        ax2.grid(True, alpha=0.3)
    
    # 3. Epsilon (exploration)
    ax3 = axes[1, 0]
    ax3.plot(epsilons)
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Epsilon')
    ax3.set_title('Epsilon (exploration) during Training')
    ax3.grid(True, alpha=0.3)
    
    # 4. Reward histogram
    ax4 = axes[1, 1]
    ax4.hist(rewards, bins=30, edgecolor='black', alpha=0.7)
    ax4.axvline(np.mean(rewards), color='red', linestyle='--', 
                linewidth=2, label=f'Mean: {np.mean(rewards):.2f}')
    ax4.set_xlabel('Reward')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Reward Distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Graph saved to: {save_path}")
    
    plt.close()


def evaluate_agent(agent, env, num_episodes=10, render=False):
    """
    Evaluate trained agent.
    
    Args:
        agent: DQN agent
        env: Environment
        num_episodes: Number of evaluation episodes
        render: Whether to render the environment
        
    Returns:
        Average reward
    """
    print("\n" + "=" * 70)
    print("Agent Evaluation")
    print("=" * 70)
    
    total_rewards = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        steps = 0
        
        while True:
            action = agent.select_action(state, training=False)
            state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            steps += 1
            
            if terminated or truncated:
                break
        
        total_rewards.append(episode_reward)
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Steps = {steps}")
    
    avg_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    
    print("\n" + "-" * 70)
    print(f"Average reward: {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"Min reward: {min(total_rewards):.2f}")
    print(f"Max reward: {max(total_rewards):.2f}")
    print("=" * 70 + "\n")
    
    return avg_reward


def main():
    """Main function for running training."""
    # Training
    agent, stats = train_dqn(
        num_episodes=500,
        max_steps=500,
        save_dir="models",
        save_freq=100,
        plot_freq=100
    )
    
    # Evaluation
    env = CartPoleEnvironment()
    evaluate_agent(agent, env, num_episodes=10)
    env.close()


if __name__ == "__main__":
    main()
