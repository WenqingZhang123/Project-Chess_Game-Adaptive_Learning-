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
import os

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


size_board = 4
env=Chess_Env(size_board)
S,X,allowed_a=env.Initialise_game()
N_a=np.shape(allowed_a)[0]   # TOTAL NUMBER OF POSSIBLE ACTIONS

N_in=np.shape(X)[0]    ## INPUT SIZE
N_h=200                ## NUMBER OF HIDDEN NODES
epsilon_0 = 0.2     # STARTING VALUE OF EPSILON FOR THE EPSILON-GREEDY POLICY
beta = 0.00005      # THE PARAMETER SETS HOW QUICKLY THE VALUE OF EPSILON IS DECAYING (SEE epsilon_f BELOW)
gamma = 0.85        # THE DISCOUNT FACTOR
eta = 0.0035        # THE LEARNING RATE
N_episodes = 100000 # THE NUMBER OF GAMES TO BE PLAYED

# SAVING VARIABLES
R_save = np.zeros([N_episodes, 1])
N_moves_save = np.zeros([N_episodes, 1])


N_episodes = 100000
agent = Agent(N_in, N_a, N_h, eta, gamma, epsilon_0, beta)

if not os.path.exists('result_Q-learning'):
    os.makedirs('result_Q-learning')
# 在训练循环外初始化最佳奖励和步骤
best_reward = -np.inf
best_steps = np.inf
best_episode = 0  # 初始化最佳模型的Episode为0
best_model_path = "result_Q-learning/best_model.pth"

for n in range(N_episodes):
    agent.decay_epsilon(n)
    Done = 0
    # 初始化游戏的时候，不可能出现直接将军或者和棋的情况
    S, X, allowed_a = env.Initialise_game()
    state = torch.from_numpy(X).float()

    total_reward = 0
    total_steps = 0

    while Done == 0:
        action = agent.select_action(state, allowed_a,testing=False)
        S_next, X_next, allowed_a_next, R, Done = env.OneStep(action)
        next_state = torch.from_numpy(np.array(X_next)).float()
        agent.update(state, action, R, next_state, Done, allowed_a_next)

        total_reward += R
        total_steps += 1

        state = next_state
        allowed_a = allowed_a_next

    agent.rewards.append(total_reward)
    agent.steps.append(total_steps)

    # 检查并保存最佳模型
    if total_reward > best_reward or (total_reward == best_reward and total_steps < best_steps):
        best_reward = total_reward
        best_steps = total_steps
        best_episode = n  # 更新最佳模型的Episode
        torch.save(agent.network.state_dict(), best_model_path)

    if n % 100 == 0:
        print(f"Episode: {n}, Reward: {total_reward}, Steps: {total_steps}")


# 保存最佳模型的表现
with open("result_Q-learning/best_performance.txt", "w") as f:
    f.write(f"Best Episode: {best_episode}\n")  # 保存最佳模型的Episode
    f.write(f"Best Reward: {best_reward}\n")
    f.write(f"Best Steps: {best_steps}\n")

# 为了平滑数据，我们使用指数移动平均
def exponential_moving_average(data, alpha=0.1):
    ret = []
    for i in range(len(data)):
        if i == 0:
            ret.append(data[i])
        else:
            ret.append(alpha * data[i] + (1-alpha) * ret[-1])
    return ret

plt.figure()
plt.plot(exponential_moving_average(agent.rewards))
plt.title("Reward per game vs training time")
plt.xlabel("Game")
plt.ylabel("Reward")
plt.savefig('result_Q-learning/reward_vs_training_time.png')  # 保存图像到result文件夹

plt.figure()
plt.plot(exponential_moving_average(agent.steps))
plt.title("Number of moves per game vs training time")
plt.xlabel("Game")
plt.ylabel("Number of moves")
plt.savefig('result_Q-learning/number_of_moves_vs_training_time.png')  # 保存图像到result文件夹

plt.show()
