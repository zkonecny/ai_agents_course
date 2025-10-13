"""
Quick test to verify everything works
"""
import sys

def test_imports():
    """Test whether all required modules can be imported."""
    print("=" * 70)
    print("Import Test")
    print("=" * 70)
    
    try:
        import gymnasium
        print("✓ gymnasium installed")
    except ImportError:
        print("✗ gymnasium not installed")
        return False
    
    try:
        import numpy
        print("✓ numpy installed")
    except ImportError:
        print("✗ numpy not installed")
        return False
    
    try:
        import torch
        print("✓ torch installed")
        print(f"  CUDA available: {torch.cuda.is_available()}")
    except ImportError:
        print("✗ torch not installed")
        return False
    
    try:
        import matplotlib
        print("✓ matplotlib installed")
    except ImportError:
        print("✗ matplotlib not installed")
        return False
    
    try:
        import tqdm
        print("✓ tqdm installed")
    except ImportError:
        print("✗ tqdm not installed")
        return False
    
    return True


def test_environment():
    """Test the environment."""
    print("\n" + "=" * 70)
    print("Environment Test")
    print("=" * 70)
    
    try:
        from environment import CartPoleEnvironment
        
        env = CartPoleEnvironment()
        print(f"✓ Environment created")
        print(f"  State dimensions: {env.get_state_dim()}")
        print(f"  Number of actions: {env.get_action_dim()}")
        
        # Test reset
        state, info = env.reset(seed=42)
        print(f"✓ Reset works")
        print(f"  Initial state: {state}")
        
        # Test step
        action = env.action_space.sample()
        state, reward, terminated, truncated, info = env.step(action)
        print(f"✓ Step works")
        print(f"  Action: {action}")
        print(f"  Reward: {reward}")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"✗ Error in environment test: {e}")
        return False


def test_agent():
    """Test the agent."""
    print("\n" + "=" * 70)
    print("DQN Agent Test")
    print("=" * 70)
    
    try:
        from environment import CartPoleEnvironment
        from dqn_agent import DQNAgent
        
        env = CartPoleEnvironment()
        agent = DQNAgent(
            state_dim=env.get_state_dim(),
            action_dim=env.get_action_dim()
        )
        
        print("✓ Agent created")
        
        # Test select_action
        state, _ = env.reset(seed=42)
        action = agent.select_action(state, training=False)
        print(f"✓ Select action works")
        print(f"  Selected action: {action}")
        
        # Test store_transition and update
        next_state, reward, terminated, truncated, _ = env.step(action)
        agent.store_transition(state, action, reward, next_state, terminated)
        print(f"✓ Store transition works")
        print(f"  Buffer size: {len(agent.replay_buffer)}")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"✗ Error in agent test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mini_training():
    """Mini training to verify everything works."""
    print("\n" + "=" * 70)
    print("Mini Training Test (5 episodes)")
    print("=" * 70)
    
    try:
        from environment import CartPoleEnvironment
        from dqn_agent import DQNAgent
        
        env = CartPoleEnvironment()
        agent = DQNAgent(
            state_dim=env.get_state_dim(),
            action_dim=env.get_action_dim(),
            batch_size=32
        )
        
        num_episodes = 5
        
        for episode in range(num_episodes):
            state, _ = env.reset()
            episode_reward = 0
            steps = 0
            
            for step in range(100):
                action = agent.select_action(state, training=True)
                next_state, reward, terminated, truncated, _ = env.step(action)
                
                agent.store_transition(state, action, reward, next_state, terminated)
                loss = agent.update()
                
                episode_reward += reward
                steps += 1
                state = next_state
                
                if terminated or truncated:
                    break
            
            print(f"  Episode {episode + 1}: Reward = {episode_reward}, Steps = {steps}")
        
        print("✓ Mini training works!")
        env.close()
        return True
        
    except Exception as e:
        print(f"✗ Error in mini training: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("QUICK TEST - CartPole DQN")
    print("=" * 70 + "\n")
    
    tests = [
        ("Imports", test_imports),
        ("Environment", test_environment),
        ("Agent", test_agent),
        ("Mini training", test_mini_training)
    ]
    
    results = {}
    
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"\n✗ Unexpected error in test '{name}': {e}")
            results[name] = False
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    for name, result in results.items():
        status = "✓ SUCCESS" if result else "✗ FAILED"
        print(f"{name:.<50} {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "=" * 70)
    if all_passed:
        print("✓ ALL TESTS PASSED!")
        print("\nYou can run full training using:")
        print("  python train.py")
    else:
        print("✗ SOME TESTS FAILED")
        print("\nInstall missing dependencies using:")
        print("  pip install -r requirements.txt")
    print("=" * 70 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
