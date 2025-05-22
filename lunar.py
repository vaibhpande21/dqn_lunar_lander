import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import deque, namedtuple
import random
from IPython.display import clear_output

# Setting random seeds
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Test environment
env = gym.make('LunarLander-v3')
print("Environment created successfully!")
print(f"Observation Space: {env.observation_space}")
print(f"Action Space: {env.action_space}")

# Creating DQN Network
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# ReplayBuffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

# DQL Agent
class DQLAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.policy_net = DQN(state_size, action_size).to(self.device)
        self.target_net = DQN(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        self.memory = ReplayBuffer(10000)

        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.target_update = 10

    def select_action(self, state):
        if random.random() > self.epsilon:
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state)
                return q_values.max(1)[1].item()
        else:
            return random.randrange(self.action_size)

    def train(self):
        if len(self.memory.buffer) < self.batch_size:
            return 0

        transitions = self.memory.sample(self.batch_size)

        # Transposing the batch
        batch = list(zip(*transitions))

        # Converting to torch tensors
        states = torch.FloatTensor(np.array(batch[0])).to(self.device)
        actions = torch.LongTensor(np.array(batch[1])).to(self.device)
        rewards = torch.FloatTensor(np.array(batch[2])).to(self.device)
        next_states = torch.FloatTensor(np.array(batch[3])).to(self.device)
        dones = torch.FloatTensor(np.array(batch[4])).to(self.device)

        # Computing Q values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_net(next_states).max(1)[0].detach()
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Computing loss
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)

        # Optimizing the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

# Training function
def train_agent(num_episodes=1000):
    env = gym.make('LunarLander-v3')  # Changed to v3
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = DQLAgent(state_size, action_size)
    rewards_history = []
    loss_history = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        losses = []

        while True:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Storing the transition
            agent.memory.push(
                state.copy(),  # Ensuring we store a copy of the state
                action,
                reward,
                next_state.copy(),  # Ensuring we store a copy of the next state
                1.0 if done else 0.0  # Storing done as float
            )

            # Train the agent
            loss = agent.train()
            if loss != 0:
                losses.append(loss)

            total_reward += reward
            state = next_state

            if done:
                break

        # Updating target network
        if episode % agent.target_update == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())

        # Decay epsilon
        agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)

        # Storing history
        rewards_history.append(total_reward)
        avg_loss = np.mean(losses) if losses else 0
        loss_history.append(avg_loss)

        # Printing progress
        if episode % 20 == 0:
            clear_output(wait=True)
            print(f"Episode: {episode}")
            print(f"Average Reward (last 100): {np.mean(rewards_history[-100:]):.2f}")
            print(f"Epsilon: {agent.epsilon:.2f}")
            print(f"Buffer size: {len(agent.memory.buffer)}")

            # Plotting the progress
            plt.figure(figsize=(15, 5))

            plt.subplot(1, 2, 1)
            plt.plot(rewards_history)
            plt.title('Training Rewards')
            plt.xlabel('Episode')
            plt.ylabel('Total Reward')

            plt.subplot(1, 2, 2)
            plt.plot(loss_history)
            plt.title('Training Loss')
            plt.xlabel('Episode')
            plt.ylabel('Loss')

            plt.tight_layout()
            plt.show()

    env.close()
    return agent, rewards_history, loss_history

# Training the agent
agent, rewards_history, loss_history = train_agent(num_episodes=1000)

# Plotting final results
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.plot(rewards_history)
plt.title('Final Training Rewards')
plt.xlabel('Episode')
plt.ylabel('Total Reward')

plt.subplot(1, 2, 2)
plt.plot(loss_history)
plt.title('Final Training Loss')
plt.xlabel('Episode')
plt.ylabel('Loss')

plt.tight_layout()
plt.show()

# Saving the trained model
torch.save(agent.policy_net.state_dict(), 'lunar_lander_dqn.pth')
