import numpy as np
from degree_freedom_queen import *
from degree_freedom_king1 import *
from degree_freedom_king2 import *
from generate_game import *

class Chess_Env:
    
    def __init__(self,N_grid):
        
        # board的大小，N_grid是一个数字
        self.N_grid=N_grid                     # SIZE OF THE BOARD

        # 创建一个N_grid的正方形棋盘，棋子是0/1/2/3
        # THE BOARD, THIS WILL BE FILLED BY 0 (NO PIECE), 1 (AGENT'S KING), 2 (AGENT'S QUEEN), 3 (OPPONENT'S KING)
        # 0（没有棋子），1（代理的国王），2（代理的皇后），3（对手的国王）
        self.Board=np.zeros([N_grid,N_grid])

        # 初始化建了一个包含两行一列的二维数组，用来存放棋子的位置
        # POSITION OF THE AGENT'S KING AS COORDINATES
        # 代理王的位置
        self.p_k1=np.zeros([2,1])
        # POSITION OF THE OPPOENT'S KING AS COORDINATES
        # 对手王的位置
        self.p_k2=np.zeros([2,1])
        # POSITION OF THE AGENT'S QUEEN AS COORDINATES
        # 代理王后的位置
        self.p_q1=np.zeros([2,1])


        # ALL POSSIBLE ACTIONS FOR THE AGENT'S KING (LOCATIONS WHERE IT CAN MOVE WITHOUT THE PRESENCE OF THE OTHER PIECES)
        # 代理王的所有可能行动（它可以在没有其他棋子存在的情况下移动的位置）
        self.dfk1=np.zeros([N_grid,N_grid])
        # ALL POSSIBLE ACTIONS FOR THE OPPONENT'S KING (LOCATIONS WHERE IT CAN MOVE WITHOUT THE PRESENCE OF THE OTHER PIECES)
        # 对手王的所有可能行动（它可以在没有其他棋子存在的情况下移动的位置）
        self.dfk2=np.zeros([N_grid,N_grid])
        # ALL POSSIBLE ACTIONS FOR THE AGENT'S QUEEN (LOCATIONS WHERE IT CAN MOVE WITHOUT THE PRESENCE OF THE OTHER PIECES)
        # 代理王后的所有可能行动（它可以在没有其他棋子存在的情况下移动的位置）
        self.dfq1=np.zeros([N_grid,N_grid])


        # ALLOWED ACTIONS FOR THE AGENT'S KING CONSIDERING ALSO THE OTHER PIECES
        # 允许的行动，代理王考虑到其他棋子
        self.dfk1_constrain=np.zeros([N_grid,N_grid])
        # ALLOWED ACTIONS FOT THE OPPONENT'S KING CONSIDERING ALSO THE OTHER PIECES
        # 允许的行动，对手王考虑到其他棋子
        self.dfk2_constrain=np.zeros([N_grid,N_grid])
        # ALLOWED ACTIONS FOT THE AGENT'S QUEEN CONSIDERING ALSO THE OTHER PIECES
        # 允许的行动，代理王后考虑到其他棋子
        self.dfq1_constrain=np.zeros([N_grid,N_grid])

        # 储存国王下一步可能落子位置
        # ALLOWED ACTIONS OF THE AGENT'S KING (CONSIDERING OTHER PIECES), ONE-HOT ENCODED
        # 代理王的允许行动（考虑其他棋子），one-hot编码
        self.ak1=np.zeros([8])
        # TOTAL NUMBER OF POSSIBLE ACTIONS FOR AGENT'S KING
        # 代理王的可能行动总数
        self.possible_king_a=np.shape(self.ak1)[0]

        # 储存王后下一步可能落子位置
        # ALLOWED ACTIONS OF THE AGENT'S QUEEN (CONSIDERING OTHER PIECES), ONE-HOT ENCODED
        # 代理王后的允许行动（考虑其他棋子），one-hot编码
        self.aq1=np.zeros([8*(self.N_grid-1)])
        # TOTAL NUMBER OF POSSIBLE ACTIONS FOR AGENT'S QUEEN
        # 代理王后的可能行动总数
        self.possible_queen_a=np.shape(self.aq1)[0]

        # 敌人国王当前是否被check
        # 1 (0) IF ENEMY KING (NOT) IN CHECK
        # 1（0）如果敌人国王（不）在检查中
        self.check=0

        # 每个一维数组表示一个在棋盘上移动的方向，
        # 例如[1,0]表示向右移动一格，[0,1]表示向上移动一格，[-1,0]表示向左移动一格，[0,-1]表示向下移动一格。
        # 同时，还包括了斜向的方向，如[1,1]表示向右上方移动一格，[1,-1]表示向右下方移动一格，[-1,1]表示向左上方移动一格，[-1,-1]表示向左下方移动一格。
        # THIS MAP IS USEFUL FOR US TO UNDERSTAND THE DIRECTION OF MOVEMENT GIVEN THE ACTION MADE (SKIP...)
        self.map=np.array([[1, 0],
                            [-1, 0],
                            [0, 1],
                            [0, -1],
                            [1, 1],
                            [1, -1],
                            [-1, 1],
                            [-1, -1]])

        
        
    def Initialise_game(self):
        # START THE GAME BY SETTING PIECIES

        # 初始化游戏
        self.Board,self.p_k2,self.p_k1,self.p_q1=generate_game(self.N_grid)

       # 跟新k1位置
        # Allowed actions for the agent's king
        self.dfk1_constrain, self.a_k1, self.dfk1 = degree_freedom_king1(self.p_k1, self.p_k2, self.p_q1, self.Board)

        # 跟新queen位置
        # Allowed actions for the agent's queen
        self.dfq1_constrain, self.a_q1, self.dfq1  = degree_freedom_queen(self.p_k1, self.p_k2, self.p_q1, self.Board)

        # 跟新k2位置
        # Allowed actions for the enemy's king
        self.dfk2_constrain, self.a_k2, self.check = degree_freedom_king2(self.dfk1, self.p_k2, self.dfq1, self.Board, self.p_k1)

        # ALLOWED ACTIONS FOR THE AGENT, ONE-HOT ENCODED
        allowed_a=np.concatenate([self.a_q1,self.a_k1],0)

        '''
        X 是用来作为输入喂给神经网络的特征向量。这个特征向量包括了当前状态下棋盘上所有棋子的位置信息以及敌方国王是否被将军等信息，
        它们被编码为一个一维向量 x。self.Features() 方法用来生成这个特征向量。
        '''
        # FEATURES (INPUT TO NN) AT THIS POSITION
        X=self.Features()

        
        
        return self.Board, X, allowed_a
        
    
    def OneStep(self,a_agent):
        
        # SET REWARD TO ZERO IF GAME IS NOT ENDED
        R=0
        # SET Done TO ZERO (GAME NOT ENDED)
        Done=0
        
        
        # PERFORM THE AGENT'S ACTION ON THE CHESS BOARD
        
        if a_agent < self.possible_queen_a:    # THE AGENT MOVED ITS QUEEN 
           
           # UPDATE QUEEN'S POSITION
           direction = int(np.ceil((a_agent + 1) / (self.N_grid - 1))) - 1
           steps = a_agent - direction * (self.N_grid - 1) + 1

           self.Board[self.p_q1[0], self.p_q1[1]] = 0
           
           mov = self.map[direction, :] * steps
           self.Board[self.p_q1[0] + mov[0], self.p_q1[1] + mov[1]] = 2
           self.p_q1[0] = self.p_q1[0] + mov[0]
           self.p_q1[1] = self.p_q1[1] + mov[1]

        else:                                 # THE AGENT MOVED ITS KING                               
           
           # UPDATE KING'S POSITION
           direction = a_agent - self.possible_queen_a
           steps = 1

           self.Board[self.p_k1[0], self.p_k1[1]] = 0
           mov = self.map[direction, :] * steps
           self.Board[self.p_k1[0] + mov[0], self.p_k1[1] + mov[1]] = 1
           self.p_k1[0] = self.p_k1[0] + mov[0]
           self.p_k1[1] = self.p_k1[1] + mov[1]

        
        # COMPUTE THE ALLOWED ACTIONS AFTER AGENT'S ACTION
        # Allowed actions for the agent's king
        self.dfk1_constrain, self.a_k1, self.dfk1 = degree_freedom_king1(self.p_k1, self.p_k2, self.p_q1, self.Board)
        
        # Allowed actions for the agent's queen
        self.dfq1_constrain, self.a_q1, self.dfq1  = degree_freedom_queen(self.p_k1, self.p_k2, self.p_q1, self.Board)
        
        # Allowed actions for the enemy's king
        self.dfk2_constrain, self.a_k2, self.check = degree_freedom_king2(self.dfk1, self.p_k2, self.dfq1, self.Board, self.p_k1)

        
        # CHECK IF POSITION IS A CHECMATE, DRAW, OR THE GAME CONTINUES
        
        # CASE OF CHECKMATE
        if np.sum(self.dfk2_constrain) == 0 and self.dfq1[self.p_k2[0], self.p_k2[1]] == 1:
           
            # King 2 has no freedom and it is checked
            # Checkmate and collect reward
            Done = 1       # The epsiode ends
            R = 1          # Reward for checkmate
            allowed_a=[]   # Allowed_a set to nothing (end of the episode)
            X=[]           # Features set to nothing (end of the episode)
        
        # CASE OF DRAW
        elif np.sum(self.dfk2_constrain) == 0 and self.dfq1[self.p_k2[0], self.p_k2[1]] == 0:
           
            # King 2 has no freedom but it is not checked
            Done = 1        # The epsiode ends
            R = 0.       # Reward for draw
            allowed_a=[]    # Allowed_a set to nothing (end of the episode)
            X=[]            # Features set to nothing (end of the episode)
        
        # THE GAME CONTINUES
        else:
            
            # THE OPPONENT MOVES THE KING IN A RANDOM SAFE LOCATION
            allowed_enemy_a = np.where(self.a_k2 > 0)[0]
            a_help = int(np.ceil(np.random.rand() * allowed_enemy_a.shape[0]) - 1)
            a_enemy = allowed_enemy_a[a_help]

            direction = a_enemy
            steps = 1

            self.Board[self.p_k2[0], self.p_k2[1]] = 0
            mov = self.map[direction, :] * steps
            self.Board[self.p_k2[0] + mov[0], self.p_k2[1] + mov[1]] = 3

            self.p_k2[0] = self.p_k2[0] + mov[0]
            self.p_k2[1] = self.p_k2[1] + mov[1]
            
            
            
            # COMPUTE THE ALLOWED ACTIONS AFTER THE OPPONENT'S ACTION
            # Possible actions of the King
            self.dfk1_constrain, self.a_k1, self.dfk1 = degree_freedom_king1(self.p_k1, self.p_k2, self.p_q1, self.Board)
            
            # Allowed actions for the agent's king
            self.dfq1_constrain, self.a_q1, self.dfq1  = degree_freedom_queen(self.p_k1, self.p_k2, self.p_q1, self.Board)
            
            # Allowed actions for the enemy's king
            self.dfk2_constrain, self.a_k2, self.check = degree_freedom_king2(self.dfk1, self.p_k2, self.dfq1, self.Board, self.p_k1)

            # ALLOWED ACTIONS FOR THE AGENT, ONE-HOT ENCODED
            allowed_a=np.concatenate([self.a_q1,self.a_k1],0)
            # FEATURES
            X=self.Features()
            
            
        
        return self.Board, X, allowed_a, R, Done
        
        
    # DEFINITION OF THE FEATURES (SEE ALSO ASSIGNMENT DESCRIPTION)
    def Features(self):
        
        # 将各自的位置信息送到神经网络中去： 对应特征-3N**2
        # 这是什么意思？

        s_k1 = np.array(self.Board == 1).astype(float).reshape(-1)   # FEATURES FOR KING POSITION
        s_q1 = np.array(self.Board == 2).astype(float).reshape(-1)   # FEATURES FOR QUEEN POSITION
        s_k2 = np.array(self.Board == 3).astype(float).reshape(-1)   # FEATURE FOR ENEMY'S KING POSITION

        # 送入神经网络的另外2个值
        check=np.zeros([2])    # CHECK? FEATURE
        # 因为self.check--0/1
        check[self.check]=1   

        # 敌人方下一个位置可以走的方位，送入神经网络的8个值
        K2dof=np.zeros([8])   # NUMBER OF ALLOWED ACTIONS FOR ENEMY'S KING, ONE-HOT ENCODED
        K2dof[np.sum(self.dfk2_constrain).astype(int)]=1
        
        # ALL FEATURES...
        # 0，表示在一维纬度上连接
        x = np.concatenate([s_k1, s_q1, s_k2, check, K2dof],0)
        
        return x
        
        


        
        
        
        
        
        
