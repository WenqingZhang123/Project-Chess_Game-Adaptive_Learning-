import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from degree_freedom_queen import *
from degree_freedom_king1 import *
from degree_freedom_king2 import *
from generate_game import *
from Chess_env import *


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=200):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        return self.fc2(x)


class Agent:
    def __init__(self, state_size, action_size, hidden_size=200, lr=0.0035, gamma=0.85, epsilon=0.2, beta=0.00005):
        self.state_size = state_size
        self.action_size = action_size
        self.network = QNetwork(state_size, action_size, hidden_size)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_0 = epsilon  # 设置 epsilon 的初始值
        self.beta = beta
        self.rewards = []
        self.steps = []


    def select_action(self, state, allowed_actions, testing=False):
        if not testing and np.random.rand() < self.epsilon:
            return np.random.choice(np.where(allowed_actions.flatten() == 1)[0])
        else:
            with torch.no_grad():
                q_values = self.network(state.float())
            q_values = q_values.numpy()
            q_values[allowed_actions.flatten() == 0] = -np.inf
            return np.argmax(q_values)



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

    def decay_epsilon(self, episode):
        self.epsilon = self.epsilon_0 / (1 + self.beta * episode)

def generate_fight_filename(dir_path):
    i = 0
    while os.path.exists(f"{dir_path}/fight_{i}.txt"):
        i += 1
    return f"{dir_path}/fight_{i}.txt"


size_board = 4
env=Chess_Env(size_board)
# Simulate a game
S, X, allowed_a = env.Initialise_game()

state_size = np.shape(X)[0]
# size_board ** 2 * 12
action_size = np.shape(allowed_a)[0]
# size_board ** 2 * 8

# Initialize the agent
agent = Agent(state_size, action_size)

# Load the best model
# best_model_path = "result_Q-learning/best_model.pth"
best_model_path = "result_Sarsa/best_model.pth"
agent.network.load_state_dict(torch.load(best_model_path))
agent.network.eval()  # Set the network to evaluation mode


# fight_filename = generate_fight_filename("result_Q-learning")
fight_filename = generate_fight_filename("result_Sarsa")
with open(fight_filename, "w") as fight_file:
    fight_file.write(str(S) + "\n")
    fight_file.write('check? ' + str(env.check) + "\n")
    fight_file.write('dofk2 ' + str(np.sum(env.dfk2_constrain).astype(int)) + "\n")

    for i in range(100):  # 你可以根据需要调整此数字
        state = torch.from_numpy(X).float()
        action = agent.select_action(state, allowed_a, testing=True)
        S, X, allowed_a, R, Done = env.OneStep(action)

        fight_file.write('\n' + str(S) + "\n")
        fight_file.write('Reward: '+str(R) + ' ' +'Done: ' + str(Done) + "\n")
        fight_file.write('check? ' + str(env.check) + "\n")
        fight_file.write('dofk2 ' + str(np.sum(env.dfk2_constrain).astype(int)) + "\n")

        if Done:
            break
