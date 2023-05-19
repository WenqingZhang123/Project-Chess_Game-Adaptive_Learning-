import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from Chess_env import *
import os
import math

'''
# Novelty
# Define the Q-Network, which maps a given state to Q-values of actions.
'''
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=200):
        super(QNetwork, self).__init__()
        # Define the layers of the neural network
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    # Define the forward pass of the network, applying ReLU activations after each linear layer.
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# Define the agent that will interact with the environment
class Agent:
    def __init__(self, state_size, action_size, hidden_size=200, lr=0.0035, gamma=0.85, epsilon=0.2, beta=0.00005,
                 lr_decay_start=0.3):
        self.state_size = state_size
        self.action_size = action_size
        self.network = QNetwork(state_size, action_size, hidden_size)
        self.initial_lr = lr  # Record the initial learning rate
        self.lr = lr  # Current learning rate
        # Initialize the optimizer with the current learning rate
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.lr)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_0 = epsilon  # Set the initial value of epsilon
        self.beta = beta
        # The proportion at which learning rate decay begins (e.g., 0.3 means it starts after 30% of the training)
        self.lr_decay_start = lr_decay_start
        self.rewards = []
        self.steps = []

    '''
    # Novelty
    # Implement learning rate decay. This is a novelty because it allows for a more nuanced control over the learning
    # rate, potentially leading to improved training dynamics.
    '''
    def decay_lr(self, episode, total_episodes):
        # Start decaying when training exceeds the specified ratio
        if episode >= self.lr_decay_start * total_episodes:
            fraction = (episode - self.lr_decay_start * total_episodes) / (
                        total_episodes - self.lr_decay_start * total_episodes)
            # The learning rate gradually decays as training progresses
            self.lr = self.initial_lr * (1 - fraction)
            # Update the learning rate in the optimizer
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.lr

    # Select an action based on epsilon-greedy strategy.
    def select_action(self, state, allowed_actions, testing=False):
        if not testing and np.random.rand() < self.epsilon:
            return np.random.choice(np.where(allowed_actions.flatten() == 1)[0])
        else:
            with torch.no_grad():
                q_values = self.network(state.float())
            q_values = q_values.numpy()
            q_values[allowed_actions.flatten() == 0] = -np.inf
            return np.argmax(q_values)

    # Update the network based on the observed reward and the next state.
    def update(self, state, action, reward, next_state, done, allowed_actions_next):
        self.optimizer.zero_grad()
        q_values = self.network(state)
        with torch.no_grad():
            if done:
                q_values_next = torch.zeros_like(q_values)
            else:
                q_values_next = self.network(next_state)
                allowed_actions_next = torch.from_numpy(np.array(allowed_actions_next)).float()
                q_values_next[allowed_actions_next.flatten() == 0] = -np.inf
        q_target = reward + (self.gamma * q_values_next.max()) * (1.0 - done)
        loss = (q_values[action] - q_target) ** 2
        loss.backward()
        self.optimizer.step()

    # Decay the epsilon parameter which determines the balance between exploration and exploitation.
    def decay_epsilon(self, episode):
        self.epsilon = self.epsilon_0 / (1 + self.beta * episode)


# Set the board size
size_board = 4
# Initialize the chess environment
env = Chess_Env(size_board)
# Retrieve the state, the state representation, and the allowed actions from the initialized game
S, X, allowed_a = env.Initialise_game()
# Total number of possible actions
N_a = np.shape(allowed_a)[0]
# Input size
N_in = np.shape(X)[0]
# Number of hidden nodes
N_h = 200
# Starting value of epsilon for the epsilon-greedy policy
epsilon_0 = 0.2
# The parameter sets how quickly the value of epsilon is decaying
beta = 0.00005
# beta = 0.0005
# beta = 0.005
# beta = 0.05

# The discount factor
gamma = 0.85
# gamma = 0.95
# gamma = 0.75
# gamma = 0.65
# The learning rate
eta = 0.0035
# The number of games to be played
N_episodes = 100000

# Create arrays to save the total reward and number of moves for each episode
R_save = np.zeros([N_episodes, 1])
N_moves_save = np.zeros([N_episodes, 1])

# Initialize the agent with the specified parameters
agent = Agent(N_in, N_a, N_h, eta, gamma, epsilon_0, beta)

# Create a directory to save the results of Q-learning
if not os.path.exists('result_Q-learning'):
    os.makedirs('result_Q-learning')

# Initialize the best reward and steps outside of the training loop
best_reward = -np.inf
best_steps = np.inf
# Initialize the episode of the best model as 0
best_episode = 0
# Path to save the best model
best_model_path = "result_Q-learning/best_model.pth"

# Start the training loop
for n in range(N_episodes):
    # Novelty: Call the decay function at the start of each episode
    agent.decay_lr(n, N_episodes)
    agent.decay_epsilon(n)

    # Initialize the game
    S, X, allowed_a = env.Initialise_game()
    state = torch.from_numpy(X).float()

    total_reward = 0
    total_steps = 0

    # Add a reward counter
    reward_sum = 0
    # Add a step counter
    steps_without_increase = 0

    Done = 0
    while Done == 0:
        # Select the action
        action = agent.select_action(state, allowed_a, testing=False)
        # Execute the action and get the next state, reward, and whether the game is done
        S_next, X_next, allowed_a_next, R, Done = env.OneStep(action)
        # Transform the next state into a PyTorch tensor
        next_state = torch.from_numpy(np.array(X_next)).float()
        # Update the Q-network
        agent.update(state, action, R, next_state, Done, allowed_a_next)

        # Update the total reward and steps
        total_reward += R
        total_steps += 1

        # Check the consecutive steps and rewards
        if steps_without_increase >= 20:
            # If the sum of rewards in 20 consecutive steps has not increased significantly, terminate the current game
            if reward_sum <= 1:
                Done = 1
            else:  # If the reward has increased, reset the counters
                steps_without_increase = 0
                reward_sum = 0

        # Update the state and allowed actions
        state = next_state
        allowed_a = allowed_a_next

    # Append the total reward and steps to the agent's memory
    agent.rewards.append(total_reward)
    agent.steps.append(total_steps)
    '''
    # Novelty: Check and save the best model, only start saving after 1/2N_episodes
    '''
    if n >= N_episodes // 2:
        if total_reward > best_reward or (total_reward == best_reward and total_steps < best_steps):
            best_reward = total_reward
            best_steps = total_steps
            best_episode = n  # Update the episode of the best model
            # Save the state dictionary of the best model
            torch.save(agent.network.state_dict(), best_model_path)

    # Print the training progress every 100 episodes
    if n % 100 == 0:
        print(f"Episode: {n}, Reward: {total_reward}, Steps: {total_steps}")

# Save the performance of the best model
with open("result_Q-learning/best_performance.txt", "w") as f:
    f.write(f"Best Episode: {best_episode}\n")
    f.write(f"Best Reward: {best_reward}\n")
    f.write(f"Best Steps: {best_steps}\n")

# A function to smooth the data using exponential moving average
def exponential_moving_average(data, alpha=0.1):
    ret = []
    for i in range(len(data)):
        if i == 0:
            ret.append(data[i])
        else:
            ret.append(alpha * data[i] + (1-alpha) * ret[-1])
    return ret

# Plot the reward per game
plt.figure()
plt.plot(exponential_moving_average(agent.rewards))
plt.title("Reward per game vs training time")
plt.xlabel("Game")
plt.ylabel("Reward")
# Save the plot to the 'result_Q-learning' directory
plt.savefig('result_Q-learning/reward_vs_training_time.png')

# Plot the number of moves per game
plt.figure()
plt.plot(exponential_moving_average(agent.steps))
plt.title("Number of moves per game vs training time")
plt.xlabel("Game")
plt.ylabel("Number of moves")
# Save the plot to the 'result_Q-learning' directory
plt.savefig('result_Q-learning/number_of_moves_vs_training_time.png')

# Display the plots
plt.show()

# Create smoothed reward and steps data
ema_rewards = exponential_moving_average(agent.rewards)
ema_steps = exponential_moving_average(agent.steps)

# Create a combined plot with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# Plot the reward per game in the first subplot
ax1.plot(ema_rewards)
ax1.set_title("Reward per game vs training time")
ax1.set_xlabel("Game")
ax1.set_ylabel("Reward")

# Plot the number of moves per game in the second subplot
ax2.plot(ema_steps)
ax2.set_title("Number of moves per game vs training time")
ax2.set_xlabel("Game")
ax2.set_ylabel("Number of moves")

# Add a text to the plot about the parameters
plt.figtext(0.5, 0.01, f"The discount factor γ: {gamma}, the speed β of the decaying trend of ε: {beta}, epsilon_0:{epsilon_0}", ha="center", fontsize=12)
plt.tight_layout()

# Save the combined plot to the 'result_Q-learning' directory
plt.savefig('result_Q-learning/combined_graph.png')
# Display the plot
plt.show()

