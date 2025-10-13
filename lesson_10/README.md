# Lesson 10: Reinforcement Learning - CartPole with DQN

Implementation of a simple Reinforcement Learning agent to solve the CartPole problem using the DQN (Deep Q-Network) algorithm.

## 📋 Contents

- [Project Description](#project-description)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Results](#results)

## 🎯 Project Description

This project demonstrates **Reinforcement Learning** basics on the classic **CartPole** problem:

### CartPole Problem
- **Goal**: Keep the pole balanced in an upright position as long as possible
- **State**: 4D vector [cart position, cart velocity, pole angle, pole angular velocity]
- **Actions**: 2 options [0=push left, 1=push right]
- **Reward**: +1 for each step where the pole is upright
- **Termination**: Pole falls (angle > 15°) or cart moves too far

### DQN Algorithm

**DQN (Deep Q-Network)** is an algorithm that:
1. **Uses a neural network** to approximate the Q-function Q(s,a)
2. **Experience Replay** - stores experiences in a buffer and learns from random samples
3. **Target Network** - stabilizes learning using a separate target network
4. **Epsilon-Greedy** - balances between exploration (trying new actions) and exploitation (using best known actions)

## 🚀 Installation

### 1. Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### Required Packages:
- `gymnasium` - RL environments
- `numpy` - Numerical computations
- `torch` - Deep learning framework
- `matplotlib` - Visualization
- `tqdm` - Progress bars

## 📖 Usage

### Train Agent

```bash
python train.py
```

This will run:
- DQN agent training for 500 episodes
- Automatic model saving every 100 episodes
- Training progress graph generation
- Trained agent evaluation

### Test Environment

```bash
python environment.py
```

Runs CartPole environment test with random actions.

### Custom Configuration

In `train.py` you can modify training parameters:

```python
agent, stats = train_dqn(
    num_episodes=500,      # Number of training episodes
    max_steps=500,         # Max steps per episode
    save_dir="models",     # Directory for models
    save_freq=100,         # How often to save
    plot_freq=100          # How often to plot
)
```

DQN agent parameters in `train.py`:

```python
agent = DQNAgent(
    state_dim=4,                # State dimension
    action_dim=2,               # Number of actions
    learning_rate=0.001,        # Learning rate
    gamma=0.99,                 # Discount factor
    epsilon_start=1.0,          # Initial exploration
    epsilon_end=0.01,           # Minimum exploration
    epsilon_decay=0.995,        # Exploration decay rate
    buffer_capacity=10000,      # Replay buffer size
    batch_size=64,              # Batch size
    target_update_freq=10       # How often to update target network
)
```

## 📁 Project Structure

```
lesson-10/
│
├── environment.py          # CartPole environment
├── dqn_agent.py           # DQN agent implementation
├── train.py               # Training script
├── requirements.txt       # Dependencies
├── README.md             # Documentation
│
├── models/               # Saved models
│   ├── best_model.pth
│   ├── final_model.pth
│   └── training_stats.json
│
└── plots/                # Graphs and visualizations
    └── final_training_progress.png
```

## 🧠 How It Works

### 1. Q-Learning

Q-learning tries to learn a function Q(s,a) that tells how good action `a` is in state `s`.

**Bellman Equation:**
```
Q(s, a) = r + γ * max_a' Q(s', a')
```
where:
- `r` = immediate reward
- `γ` (gamma) = discount factor (0.99)
- `s'` = next state
- `a'` = possible actions in next state

### 2. Neural Network

DQN uses a neural network to approximate the Q-function:

```
Input (state) → Hidden Layer 1 (128 neurons) → Hidden Layer 2 (128 neurons) → Output (Q-values for actions)
```

### 3. Experience Replay

Instead of learning from each step immediately:
1. Store experiences (s, a, r, s', done) in a buffer
2. Randomly sample batches from experiences
3. Learn from these batches

**Advantages:**
- Breaks correlation between consecutive samples
- Better data utilization

### 4. Target Network

We use two networks:
- **Q-network**: Current network for action selection
- **Target network**: Stable network for target value computation

The target network is updated every N steps, which stabilizes learning.

### 5. Epsilon-Greedy Strategy

```python
if random() < epsilon:
    action = random_action()  # Exploration
else:
    action = argmax(Q(s, a))  # Exploitation
```

Epsilon gradually decreases from 1.0 to 0.01.

## 📊 Results

After training, the following graphs are generated:

1. **Rewards during training** - Shows learning progress
2. **Training Loss** - Shows how well the network is learning
3. **Epsilon (exploration)** - Tracks exploration decay
4. **Reward distribution** - Histogram of final results

### Expected Results:

- **Beginning of training**: Low rewards (~20-50), high exploration
- **Middle of training**: Growing rewards (~100-200), decreasing exploration
- **End of training**: High stable rewards (~200-500), low exploration

CartPole is "solved" when the agent achieves an average reward of **195** over 100 consecutive episodes.

## 🔧 Hyperparameter Tuning

### For faster learning:
- Increase `learning_rate` (e.g., 0.003)
- Decrease `batch_size` (e.g., 32)
- Increase `epsilon_decay` (e.g., 0.999)

### For more stable learning:
- Decrease `learning_rate` (e.g., 0.0005)
- Increase `batch_size` (e.g., 128)
- Decrease `epsilon_decay` (e.g., 0.99)

### For better exploration:
- Increase `epsilon_start` (e.g., 1.0)
- Increase `epsilon_end` (e.g., 0.05)
- Decrease `epsilon_decay` (exploration lasts longer)

## 📚 Next Steps

After mastering CartPole, you can try:

1. **More difficult environments**:
   - LunarLander-v2
   - MountainCar-v0
   - Atari games

2. **Advanced algorithms**:
   - Double DQN
   - Dueling DQN
   - Prioritized Experience Replay
   - Rainbow DQN

3. **Policy-based methods**:
   - REINFORCE
   - Actor-Critic
   - PPO (Proximal Policy Optimization)

## 🎓 Educational Resources

- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Spinning Up in Deep RL (OpenAI)](https://spinningup.openai.com/)
- [DQN Paper (DeepMind, 2015)](https://arxiv.org/abs/1312.5602)

## 📝 Notes

- CartPole is a stochastic environment → results may vary
- Learning may take 5-15 minutes depending on hardware
- CPU training is sufficient, GPU not necessary
- First episodes may have very low rewards - this is normal!
