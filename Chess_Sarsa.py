
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from Chess_env import *
import os

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

    # The function used to select action. It will select action randomly in training when a random value is less than epsilon.
    # Otherwise, it selects the action with the highest Q value according to the network.
    def select_action(self, state, allowed_actions, testing=False):
        if not testing and np.random.rand() < self.epsilon:
            # With probability epsilon, select a random action. This is the exploration phase of the epsilon-greedy approach.
            return np.random.choice(np.where(allowed_actions.flatten() == 1)[0])
        else:
            # With probability 1-epsilon, select the action with the maximum Q-value. This is the exploitation phase of the epsilon-greedy approach.
            # In testing phase, we rely on what we have learned and only do exploitation (choose the best option we know).
            with torch.no_grad():
                # Generate Q-values from the neural network for each action.
                q_values = self.network(state.float())
            q_values = q_values.numpy()
            # For actions that are not allowed, set their Q-values to negative infinity.
            q_values[allowed_actions.flatten() == 0] = -np.inf
            # Return the action with the highest Q-value.
            return np.argmax(q_values)

    # This function updates the Q values using the Bellman equation. It considers the end game situation by checking the 'done' flag.
    def update(self, state, action, reward, next_state, done, next_action, allowed_actions_next):
        self.optimizer.zero_grad()
        # Predict the Q-values from the neural network for current state.
        q_values = self.network(state)
        with torch.no_grad():
            if done:
                # If the game has ended, the next state does not exist. Therefore, we create a zero Q-value vector.
                q_values_next = torch.zeros_like(q_values)
            else:
                # Predict the Q-values for the next state.
                q_values_next = self.network(next_state)
                # Set Q-values of not allowed actions to negative infinity.
                allowed_actions_next = torch.from_numpy(np.array(allowed_actions_next)).float()
                q_values_next[allowed_actions_next.flatten() == 0] = -np.inf

        if next_action is None:
            # If the game has ended, the Q-value target is just the reward from the last action.
            q_target = reward
        else:
            # Otherwise, the Q-value target is calculated according to the Bellman equation.
            q_target = reward + (self.gamma * q_values_next[next_action]) * (1.0 - done)

        # Calculate the loss as the square difference between the target and predicted Q-values.
        loss = (q_values[action] - q_target) ** 2
        # Perform backpropagation to update the neural network.
        loss.backward()
        # Adjust the weights of the network based on the calculated gradients.
        self.optimizer.step()

    # This function adjusts epsilon based on the episode number, allowing the agent to gradually transition from exploration to exploitation.
    def decay_epsilon(self, episode):
        # Epsilon is decayed over time according to the inverse relation with the episode number.
        # This allows the agent to explore the environment at the beginning of training and exploit its learned knowledge towards the end.
        self.epsilon = self.epsilon_0 / (1 + self.beta * episode)

# Initializing game parameters and agent
size_board = 4
env = Chess_Env(size_board)  # Initialize the game environment.
S, X, allowed_a = env.Initialise_game()  # Initialize the game.
N_a = np.shape(allowed_a)[0]   # Total number of possible actions.

N_in = np.shape(X)[0]    # Input size.
N_h = 200                # Number of hidden nodes.
epsilon_0 = 0.2          # Starting value of epsilon for the epsilon-greedy policy.
beta = 0.00005           # The parameter sets how quickly the value of epsilon is decaying.
gamma = 0.85             # The discount factor.
eta = 0.0035             # The learning rate.
N_episodes = 100000      # The number of games to be played.

# Initializing storage variables
R_save = np.zeros([N_episodes, 1])       # Stores the total reward for each episode.
N_moves_save = np.zeros([N_episodes, 1]) # Stores the number of moves in each episode.

agent = Agent(N_in, N_a, N_h, eta, gamma, epsilon_0, beta) # Initialize the agent.

# Preparing for model storage, this allows saving the best performing model and loading it later for testing or further training.
if not os.path.exists('result_Sarsa'):
    os.makedirs('result_Sarsa')

# Outside of the training loop, initialize the best reward and steps.
best_reward = -np.inf  # Initialize the best reward as negative infinity.
best_steps = np.inf    # Initialize the best steps as infinity.
best_episode = 0       # Initialize the best episode as 0.
best_model_path = "result_Sarsa/best_model.pth"  # Path to save the best model.

for n in range(N_episodes):
    # Novelty: In the beginning of each episode, call the learning rate and epsilon decay functions.
    agent.decay_lr(n, N_episodes)
    agent.decay_epsilon(n)
    Done = 0
    # Initialize game. At the start, checkmate or draw situations cannot occur.
    S, X, allowed_a = env.Initialise_game()
    state = torch.from_numpy(X).float()  # So, the initial state is not empty.

    total_reward = 0  # Initialize total reward for the episode.
    total_steps = 0  # Initialize total steps for the episode.

    start_count = 0

    reward_sum = 0  # Add a reward counter.
    steps_without_increase = 0  # Add a step counter.

    # Step 1: Select the first action in each episode.
    action = agent.select_action(state, allowed_a, testing=False)

    while Done == 0:  # While the game is still ongoing

        # Step 2: Execute the action and observe the result
        # Here, X_next is the next situation, which might be empty
        S_next, X_next, allowed_a_next, R, Done = env.OneStep(
            action)  # Perform one step in the environment given the action
        next_state = torch.from_numpy(np.array(X_next)).float()  # Convert the next state into a PyTorch tensor

        # Convert allowed_a_next to numpy array
        allowed_a_next = np.array(allowed_a_next)

        # Typically, in a Sarsa algorithm, the next action is chosen before knowing the outcome of the current action.
        # Here, the next action is chosen after observing the outcome of the current action.
        if Done == 0:  # If the game is not done
            next_action = agent.select_action(next_state, allowed_a_next, testing=False)  # Choose the next action
        elif Done == 1:  # If the game is done
            next_action = None  # There is no next action

        # Step 3: Update the Q value using the Q value of the next state-action pair
        agent.update(state, action, R, next_state, Done, next_action, allowed_a_next)

        total_reward += R  # Keep track of the total reward
        total_steps += 1  # Keep track of the total steps
        '''
        # [Novelty] Check the continuous steps and rewards
        # It's a novel idea to check if the reward hasn't significantly increased after a certain number of steps.
        # If there's no significant reward increase, it ends the current game.
        '''
        if steps_without_increase >= 20:  # If there are 20 steps without a significant reward increase
            if reward_sum <= 1:  # If the sum of rewards is less than or equal to 1
                Done = 1  # End the current game
            else:  # If there's a reward increase, reset the counter
                steps_without_increase = 0
                reward_sum = 0

        state = next_state  # Update the state with the next state
        allowed_a = allowed_a_next  # Update the allowed actions with the next allowed actions
        action = next_action  # Update the action with the next action

    agent.rewards.append(total_reward)  # Save the total reward for this episode
    agent.steps.append(total_steps)  # Save the total steps for this episode

    # Check and save the best model only after half the episodes
    # This is to ensure the model has had some time to learn and improve.
    if n >= N_episodes // 2:
        if total_reward > best_reward or (total_reward == best_reward and total_steps < best_steps):
            best_reward = total_reward
            best_steps = total_steps
            best_episode = n  # Update the episode of the best model
            torch.save(agent.network.state_dict(), best_model_path)  # Save the best model

    if n % 100 == 0:  # For every hundredth episode,
        print(f"Episode: {n}, Reward: {total_reward}, Steps: {total_steps}")  # Print the episode number, total reward, and total steps.

# Save the best model's performance
with open("result_Sarsa/best_performance.txt", "w") as f:  # Open a file to save the best performance.
    f.write(f"Best Episode: {best_episode}\n")  # Save the best model's episode.
    f.write(f"Best Reward: {best_reward}\n")  # Save the best model's reward.
    f.write(f"Best Steps: {best_steps}\n")  # Save the best model's number of steps.

# In order to smooth data, we use an exponential moving average
def exponential_moving_average(data, alpha=0.1):  # Function to compute the exponential moving average.
    ret = []
    for i in range(len(data)):
        if i == 0:
            ret.append(data[i])  # For the first data point, it is the average.
        else:
            ret.append(alpha * data[i] + (1-alpha) * ret[-1])  # Compute the exponential moving average.
    return ret

# Plot the reward per game versus training time
plt.figure()
plt.plot(exponential_moving_average(agent.rewards))  # Plot the rewards.
plt.title("Reward per game vs training time")  # Set the title of the plot.
plt.xlabel("Game")  # Set the x-label.
plt.ylabel("Reward")  # Set the y-label.
plt.savefig('result_Sarsa/reward_vs_training_time.png')  # Save the plot to a file.

# Plot the number of moves per game versus training time
plt.figure()
plt.plot(exponential_moving_average(agent.steps))  # Plot the steps.
plt.title("Number of moves per game vs training time")  # Set the title of the plot.
plt.xlabel("Game")  # Set the x-label.
plt.ylabel("Number of moves")  # Set the y-label.
plt.savefig('result_Sarsa/number_of_moves_vs_training_time.png')  # Save the plot to a file.

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
plt.savefig('result_Sarsa/combined_graph.png')
# Display the plot
plt.show()
