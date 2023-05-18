"""
The Environment
You can find the environment in the file Chess_env, which contains the class Chess_env. To define an object, you need to provide the board size considered as input. In our example, size_board=4. Chess_env is composed by the following methods:

Initialise_game.
The method initialises an episode by placing the three pieces considered (Agent's king and queen, enemy's king) in the chess board. The outputs of the method are described below in order.

S
A matrix representing the board locations filled with 4 numbers: 0, no piece in that position; 1, location of the agent's king; 2 location of the queen; 3 location of the enemy king.

X
The features, that is the input to the neural network. See the assignment for more information regarding the definition of the features adopted. To personalise this, go into the Features method of the class Chess_env() and change accordingly.

allowed_a
The allowed actions that the agent can make. The agent is moving a king, with a total number of 8 possible actions, and a queen, with a total number of  (𝑏𝑜𝑎𝑟𝑑𝑠𝑖𝑧𝑒−1)×8
  actions. The total number of possible actions correspond to the sum of the two, but not all actions are allowed in a given position (movements to locations outside the borders or against chess rules). Thus, the variable allowed_a is a vector that is one (zero) for an action that the agent can (can't) make. Be careful, apply the policy considered on the actions that are allowed only.

OneStep.
The method performs a one step update of the system. Given as input the action selected by the agent, it updates the chess board by performing that action and the response of the enemy king (which is a random allowed action in the settings considered). The first three outputs are the same as for the Initialise_game method, but the variables are computed for the position reached after the update of the system. The fourth and fifth outputs are:

R
The reward. To change this, look at the OneStep method of the class where the rewards are set.

Done
A variable that is 1 if the episode has ended (checkmate or draw).

Features.
Given the chessboard position, the method computes the features.

This information and a quick analysis of the class should be all you need to get going. The other functions that the class exploits are uncommented and constitute an example on how not to write a python code. You can take a look at them if you want, but it is not necessary.
"""
import numpy as np
import matplotlib.pyplot as plt
from degree_freedom_queen import *
from degree_freedom_king1 import *
from degree_freedom_king2 import *
from generate_game import *
from Chess_env import *

size_board = 4

## INITIALISE THE ENVIRONMENT

env=Chess_Env(size_board)

# ========================================================================
# INITIALISE THE PARAMETERS OF YOUR NEURAL NETWORK AND...
# PLEASE CONSIDER TO USE A MASK OF ONE FOR THE ACTION MADE AND ZERO OTHERWISE IF YOU ARE NOT USING VANILLA GRADIENT DESCENT...
# WE SUGGEST A NETWORK WITH ONE HIDDEN LAYER WITH SIZE 200.


S,X,allowed_a=env.Initialise_game()
N_a=np.shape(allowed_a)[0]   # TOTAL NUMBER OF POSSIBLE ACTIONS

N_in=np.shape(X)[0]    ## INPUT SIZE
N_h=200                ## NUMBER OF HIDDEN NODES


## INITALISE YOUR NEURAL NETWORK...


# HYPERPARAMETERS SUGGESTED (FOR A GRID SIZE OF 4)

epsilon_0 = 0.2     # STARTING VALUE OF EPSILON FOR THE EPSILON-GREEDY POLICY
beta = 0.00005      # THE PARAMETER SETS HOW QUICKLY THE VALUE OF EPSILON IS DECAYING (SEE epsilon_f BELOW)
gamma = 0.85        # THE DISCOUNT FACTOR
eta = 0.0035        # THE LEARNING RATE

N_episodes = 100000 # THE NUMBER OF GAMES TO BE PLAYED

# SAVING VARIABLES
R_save = np.zeros([N_episodes, 1])
N_moves_save = np.zeros([N_episodes, 1])

# ========================================================================
# TRAINING LOOP BONE STRUCTURE...
# I WROTE FOR YOU A RANDOM AGENT (THE RANDOM AGENT WILL BE SLOWER TO GIVE CHECKMATE THAN AN OPTIMISED ONE,
# SO DON'T GET CONCERNED BY THE TIME IT TAKES), CHANGE WITH YOURS ...

for n in range(N_episodes):

    epsilon_f = epsilon_0 / (1 + beta * n)  ## DECAYING EPSILON
    Done = 0  ## SET DONE TO ZERO (BEGINNING OF THE EPISODE)
    i = 1  ## COUNTER FOR NUMBER OF ACTIONS

    S, X, allowed_a = env.Initialise_game()  ## INITIALISE GAME
    '''
    Size of S: 16
    Shape of S: (4, 4)
    Type of S: int32
    
    Size of X: 58
    Shape of X: (58,)
    Type of X: float64
    
    Size of allowed_a: 32
    Shape of allowed_a: (32, 1)
    Type of allowed_a: int32
    '''

    print(n)  ## REMOVE THIS OF COURSE, WE USED THIS TO CHECK THAT IT WAS RUNNING

    while Done == 0:  ## START THE EPISODE

        ## THIS IS A RANDOM AGENT, CHANGE IT...

        a, _ = np.where(allowed_a == 1)
        '''
        Size of a: 8
        Shape of a: (8,)
        Type of a: <class 'numpy.ndarray'>
        '''

        a_agent = np.random.permutation(a)[0]
        # 输出a_agent的大小、形状和类型
        '''
        Size of a_agent: 1
        Shape of a_agent: ()
        Type of a_agent: <class 'numpy.int64'>
        '''

        S_next, X_next, allowed_a_next, R, Done = env.OneStep(a_agent)
        '''
        size, shape, dtype
        S_next: 16 (4, 4) int32
        X_next: 58 (58,) float64
        allowed_a_next: 32 (32, 1) int32
        '''
        '''
        R:int
        Done:int
        '''

        ## THE EPISODE HAS ENDED, UPDATE...BE CAREFUL, THIS IS THE LAST STEP OF THE EPISODE
        if Done == 1:

            break


        # IF THE EPISODE IS NOT OVER...
        else:

            ## ONLY TO PUT SUMETHING
            PIPPO = 1

        # NEXT STATE AND CO. BECOME ACTUAL STATE...
        S = np.copy(S_next)
        X = np.copy(X_next)
        allowed_a = np.copy(allowed_a_next)

        i += 1  # UPDATE COUNTER FOR NUMBER OF ACTIONS




