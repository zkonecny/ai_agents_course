"""
Visualization of trained DQN agent on CartPole
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

from environment import CartPoleEnvironment
from dqn_agent import DQNAgent


def visualize_episode(agent, env, save_path=None):
    """
    Run and visualize one episode.
    
    Args:
        agent: Trained DQN agent
        env: CartPole environment
        save_path: Path to save the graph
    """
    state, _ = env.reset()
    
    # Store data for visualization
    states = [state]
    actions = []
    rewards = []
    
    episode_reward = 0
    step = 0
    
    print("\nRunning episode...")
    
    while True:
        action = agent.select_action(state, training=False)
        next_state, reward, terminated, truncated, _ = env.step(action)
        
        states.append(next_state)
        actions.append(action)
        rewards.append(reward)
        
        episode_reward += reward
        step += 1
        
        state = next_state
        
        if terminated or truncated:
            break
    
    print(f"Episode completed!")
    print(f"  Total reward: {episode_reward}")
    print(f"  Number of steps: {step}")
    
    # Convert to numpy arrays
    states = np.array(states)
    actions = np.array(actions)
    rewards = np.array(rewards)
    
    # Plot
    fig, axes = plt.subplots(3, 2, figsize=(15, 10))
    fig.suptitle(f'Episode Visualization - Total Reward: {episode_reward}', 
                 fontsize=16, fontweight='bold')
    
    steps = np.arange(len(states) - 1)
    
    # 1. Cart position
    ax1 = axes[0, 0]
    ax1.plot(steps, states[:-1, 0])
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Position')
    ax1.set_title('Cart Position')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    # 2. Cart velocity
    ax2 = axes[0, 1]
    ax2.plot(steps, states[:-1, 1])
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Velocity')
    ax2.set_title('Cart Velocity')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    # 3. Pole angle
    ax3 = axes[1, 0]
    ax3.plot(steps, states[:-1, 2])
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Angle (radians)')
    ax3.set_title('Pole Angle')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    # Mark limits
    ax3.axhline(y=0.209, color='orange', linestyle='--', alpha=0.5, label='Limit')
    ax3.axhline(y=-0.209, color='orange', linestyle='--', alpha=0.5)
    ax3.legend()
    
    # 4. Pole angular velocity
    ax4 = axes[1, 1]
    ax4.plot(steps, states[:-1, 3])
    ax4.set_xlabel('Step')
    ax4.set_ylabel('Angular Velocity')
    ax4.set_title('Pole Angular Velocity')
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    # 5. Actions
    ax5 = axes[2, 0]
    colors = ['blue' if a == 0 else 'red' for a in actions]
    ax5.scatter(steps, actions, c=colors, alpha=0.6)
    ax5.set_xlabel('Step')
    ax5.set_ylabel('Action')
    ax5.set_title('Actions Taken (0=Left, 1=Right)')
    ax5.set_yticks([0, 1])
    ax5.set_yticklabels(['Left', 'Right'])
    ax5.grid(True, alpha=0.3)
    
    # 6. Cumulative reward
    ax6 = axes[2, 1]
    cumulative_rewards = np.cumsum(rewards)
    ax6.plot(steps, cumulative_rewards)
    ax6.set_xlabel('Step')
    ax6.set_ylabel('Cumulative Reward')
    ax6.set_title('Cumulative Reward during Episode')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def compare_models(model_paths, env, num_episodes=10):
    """
    Compare performance of different models.
    
    Args:
        model_paths: List of model paths
        env: CartPole environment
        num_episodes: Number of evaluation episodes per model
    """
    results = {}
    
    for model_path in model_paths:
        print(f"\nEvaluating model: {model_path}")
        
        # Load model
        agent = DQNAgent(
            state_dim=env.get_state_dim(),
            action_dim=env.get_action_dim()
        )
        
        try:
            agent.load(model_path)
        except:
            print(f"Failed to load model: {model_path}")
            continue
        
        # Evaluation
        episode_rewards = []
        for episode in range(num_episodes):
            state, _ = env.reset()
            total_reward = 0
            
            while True:
                action = agent.select_action(state, training=False)
                state, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                
                if terminated or truncated:
                    break
            
            episode_rewards.append(total_reward)
        
        avg_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        
        model_name = os.path.basename(model_path)
        results[model_name] = {
            'mean': avg_reward,
            'std': std_reward,
            'rewards': episode_rewards
        }
        
        print(f"  Average reward: {avg_reward:.2f} ± {std_reward:.2f}")
    
    # Plot comparison
    if results:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Bar chart of average rewards
        names = list(results.keys())
        means = [results[name]['mean'] for name in names]
        stds = [results[name]['std'] for name in names]
        
        ax1.bar(range(len(names)), means, yerr=stds, capsize=5)
        ax1.set_xticks(range(len(names)))
        ax1.set_xticklabels(names, rotation=45, ha='right')
        ax1.set_ylabel('Average Reward')
        ax1.set_title('Model Comparison - Average Reward')
        ax1.grid(True, alpha=0.3)
        
        # Box plot of reward distribution
        all_rewards = [results[name]['rewards'] for name in names]
        ax2.boxplot(all_rewards, tick_labels=names)
        ax2.set_xticklabels(names, rotation=45, ha='right')
        ax2.set_ylabel('Reward')
        ax2.set_title('Model Comparison - Reward Distribution')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('plots/model_comparison.png', dpi=150, bbox_inches='tight')
        print("\nComparison saved to: plots/model_comparison.png")
        plt.close()
    
    return results


def main():
    """Main function for visualization."""
    print("=" * 70)
    print("DQN Agent Visualization - CartPole")
    print("=" * 70)
    
    # Create directory for visualizations
    os.makedirs("plots", exist_ok=True)
    
    # Initialize environment
    env = CartPoleEnvironment()
    
    # Load best model
    model_path = "models/best_model.pth"
    
    if not os.path.exists(model_path):
        print(f"\nModel not found: {model_path}")
        print("First run train.py to train an agent.")
        return
    
    print(f"\nLoading model: {model_path}")
    
    agent = DQNAgent(
        state_dim=env.get_state_dim(),
        action_dim=env.get_action_dim()
    )
    agent.load(model_path)
    
    # Visualize several episodes
    num_visualizations = 3
    
    for i in range(num_visualizations):
        print(f"\n--- Visualization {i + 1}/{num_visualizations} ---")
        visualize_episode(
            agent, 
            env, 
            save_path=f"plots/episode_visualization_{i + 1}.png"
        )
    
    # Compare models if they exist
    model_paths = []
    if os.path.exists("models/best_model.pth"):
        model_paths.append("models/best_model.pth")
    if os.path.exists("models/final_model.pth"):
        model_paths.append("models/final_model.pth")
    
    if len(model_paths) > 1:
        print("\n" + "=" * 70)
        print("Model Comparison")
        print("=" * 70)
        compare_models(model_paths, env, num_episodes=20)
    
    env.close()
    
    print("\n" + "=" * 70)
    print("Visualization completed!")
    print("Results saved in directory: plots/")
    print("=" * 70)


if __name__ == "__main__":
    main()
