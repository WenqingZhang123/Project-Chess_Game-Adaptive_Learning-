'''
搞懂每一步
按老师要求变换参数
'''

'''
您的代码看起来很好，它定义了一个基于深度Q学习网络的强化学习代理，并使用epsilon贪婪策略进行动作选择。
此外，代理也使用了Q-learning的更新规则，并在每个回合结束后，根据回合的编号进行epsilon的衰减。

一些需要注意的点是：

允许行动的选择：对于允许的行动，您在选择动作时使用了一个很好的策略，即将不允许的行动的Q值设置为负无穷。这确保了代理不会选择不允许的行动。

训练循环：在训练循环中，每一步您都更新了网络的权重，并记录了总的奖励和步数。这对于后期分析训练过程的效果非常有用。

模型的保存：您还实现了一个非常有用的特性，即在每个回合结束后，如果当前回合的奖励大于之前最好的奖励，或者当奖励相同时步数较少，就保存模型的权重。这意味着您总是保存了表现最好的模型。

数据的可视化：在训练结束后，您画出了奖励和步数随时间的变化图，这对于理解和分析训练过程非常有帮助。

代码整体来看是完整的，并且使用了一些非常好的强化学习和深度学习的实践。无论是从代码的组织结构，还是从实现的功能来看，这都是一份很好的代码。
'''

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from Chess_env import *
import os
import math

# 新颖
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=200):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class Agent:
    def __init__(self, state_size, action_size, hidden_size=200, lr=0.0035, gamma=0.85, epsilon=0.2, beta=0.00005,
                 lr_decay_start=0.3):
        self.state_size = state_size
        self.action_size = action_size
        self.network = QNetwork(state_size, action_size, hidden_size)
        self.initial_lr = lr  # 记录初始学习率
        self.lr = lr  # 当前学习率
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.lr)  # 使用当前学习率初始化优化器
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_0 = epsilon  # 设置 epsilon 的初始值
        self.beta = beta
        self.lr_decay_start = lr_decay_start  # 学习率衰减开始的比例（例如，0.3表示训练的30%之后开始衰减）
        self.rewards = []
        self.steps = []

    # 新颖
    def decay_lr(self, episode, total_episodes):
        if episode >= self.lr_decay_start * total_episodes:  # 当训练超过指定比例时开始衰减
            fraction = (episode - self.lr_decay_start * total_episodes) / (
                        total_episodes - self.lr_decay_start * total_episodes)
            self.lr = self.initial_lr * (1 - fraction)  # 学习率随训练的进行逐步衰减
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.lr  # 更新优化器中的学习率

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
N_episodes = 100000# THE NUMBER OF GAMES TO BE PLAYED

# SAVING VARIABLES
R_save = np.zeros([N_episodes, 1])
N_moves_save = np.zeros([N_episodes, 1])

agent = Agent(N_in, N_a, N_h, eta, gamma, epsilon_0, beta)

if not os.path.exists('result_Q-learning'):
    os.makedirs('result_Q-learning')
# 在训练循环外初始化最佳奖励和步骤
best_reward = -np.inf
best_steps = np.inf
best_episode = 0  # 初始化最佳模型的Episode为0
best_model_path = "result_Q-learning/best_model.pth"


for n in range(N_episodes):
    # 新颖
    agent.decay_lr(n, N_episodes)  # 在每个episode开始时调用衰减函数
    agent.decay_epsilon(n)
    Done = 0
    # 初始化游戏的时候，不可能出现直接将军或者和棋的情况
    S, X, allowed_a = env.Initialise_game()
    state = torch.from_numpy(X).float()

    total_reward = 0
    total_steps = 0

    reward_sum = 0  # 增加一个奖励计数器
    steps_without_increase = 0  # 增加一个步骤计数器

    while Done == 0:
        action = agent.select_action(state, allowed_a,testing=False)
        S_next, X_next, allowed_a_next, R, Done = env.OneStep(action)
        next_state = torch.from_numpy(np.array(X_next)).float()
        agent.update(state, action, R, next_state, Done, allowed_a_next)

        total_reward += R
        total_steps += 1

        # 检查连续的步骤和奖励
        if steps_without_increase >= 20:
            # 如果连续20步的奖励总和没有明显增加，结束当前游戏
            if reward_sum <= 1:
                Done = 1
            else:  # 如果奖励有增加，重置计数器
                steps_without_increase = 0
                reward_sum = 0

        state = next_state
        allowed_a = allowed_a_next

    agent.rewards.append(total_reward)
    agent.steps.append(total_steps)

    # 新颖
    # 检查并保存最佳模型，只在1/2N_episodes后开始保存
    if n >= N_episodes // 2:
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


# 可视化
ema_rewards = exponential_moving_average(agent.rewards)
ema_steps = exponential_moving_average(agent.steps)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.plot(ema_rewards)
ax1.set_title("Reward per game vs training time")
ax1.set_xlabel("Game")
ax1.set_ylabel("Reward")
ax2.plot(ema_steps)
ax2.set_title("Number of moves per game vs training time")
ax2.set_xlabel("Game")
ax2.set_ylabel("Number of moves")
plt.figtext(0.5, 0.01, f"The discount factor γ: {gamma}, the speed β of the decaying trend of ε: {beta}, epsilon_0:{epsilon_0}", ha="center", fontsize=12)
plt.tight_layout()
plt.savefig('result_Q-learning/combined_graph.png')
plt.show()