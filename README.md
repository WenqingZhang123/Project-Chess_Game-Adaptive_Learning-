# Chess_Game-Adaptive_Learning
This is built for the Chess game for COM3240 Adaptive Intelligence

# Chess_Sarsa.py
This file implements Chess based on SARSA reinforcement learning using Neural Network and saves the model with the best performance and some visual analysis.


# Chess_Q-learning.py
This file implements Chess based on Q-learning reinforcement learning using Neural Network and saves the model with the best performance and some visual analysis.


# Chess_fight.py
This code is an automated chess simulation program. The program runs chess game simulations and automatically stores the results of each game. The storage method generates a series of text files, which record the result of a battle. These files are named according to the format of "fight_x.txt," where 'x' is the corresponding number of fights. This design allows users to easily track and analyze the results of each match, which helps to understand and optimize players' strategies.

This code uses a different execution strategy than epsilon-greedy and other action selection methods that include random strategies during development. It relies on the excellent Q function model trained by Chess_Sarsa.py and Chess_Q-learning.py to make action decisions at each step. In other words, the code's action decision is more biased towards a deterministic policy, relying on a pre-trained Q-function model rather than a random policy.
