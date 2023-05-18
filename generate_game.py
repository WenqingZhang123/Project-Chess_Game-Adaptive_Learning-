import numpy as np
from degree_freedom_king1 import *
from degree_freedom_queen import *



def generate_game(size_board):
    """
    This function creates a new chess game with three pieces at random locations. The enemy King has no interaction with
    with our King and Queen. Positions range from 0 to 4
    :param size_board:
    :return:
    """

    # 目的：创建一个初始棋盘k2受到"k1+q1"威胁
    # --让k2无法攻击k1，q1，即刚开始给k1，q1提供保护
    # --让k2抓紧逃命


    # 创建一个棋盘
    s = np.zeros([size_board, size_board], dtype=int)
    # 这个等差数列的作用可能是为了在程序中给棋盘上的每个位置分配一个唯一的标识符，以便在后续的处理中使用。
    # 由于棋盘的大小是size_board x size_board，因此这个数组的长度为size_board * size_board。
    # 1 .... size_board * size_board
    c = np.linspace(1, size_board * size_board, num=size_board * size_board)

    k1 = 1  # King
    q1 = 2  # Queen
    k2 = 3  # Enemy King

    # 先把queen位置初始化，再把king位置初始化
    while(1):
        # Spawn Queen at a random location of the board
        # 由于rand函数生成的是0到1之间的小数
        # 因此将其乘以(size_board - 1)可以使得生成的随机整数在0到size_board-1之间，而向上取整可以保证生成的随机整数是整数。
        p_q1 = [int(np.ceil(np.random.rand() * (size_board - 1))), int(np.ceil(np.random.rand() * (size_board - 1)))]
        # 将queen的随机位置赋值到棋盘
        s[p_q1[0], p_q1[1]] = q1

        # Spawn King at a random location which is occupied
        while(1):
            p_k1 = [int(np.ceil(np.random.rand() * (size_board - 1))), int(np.ceil(np.random.rand() * (size_board - 1)))]
            if p_k1 != p_q1:
                break
        # 将king的随机位置赋值到棋盘
        s[p_k1[0], p_k1[1]] = k1

        # k1当前的位置+k1下一步可以放置的位置
        # King's location
        dfK1, _, _ = degree_freedom_king1(p_k1, [np.inf, np.inf], p_q1, s)
        dfK1[p_k1[0], p_k1[1]] = 1
        # queen1当前的位置+queen1下一步可以放置的位置
        # Queen's location
        # [np.inf, np.inf] 表示一个包含两个正无穷大值的列表。
        dfQ1, _, _ = degree_freedom_queen(p_k1, [np.inf, np.inf], p_q1, s)
        dfQ1[p_q1[0], p_q1[1]] = 1


        # Empty locations
        # 如果k1和q1有共有位置，则跳出循环；
        # ps:这是初始化游戏
        c1 = np.where(dfK1.reshape([-1]) == 0)[0]
        c2 = np.where(dfQ1.reshape([-1]) == 0)[0]
        c = np.intersect1d(c1, c2)

        if c.shape[0] != 0:
            break

    # 将k2随机初始化在"k1和q1有共有位置之一"
    '''
    生成游戏的代码会确保敌方国王的初始位置被King 1和Queen 1的控制区域所包围，
    即敌方国王无法直接攻击King 1和Queen 1。
    这样可以确保游戏的难度，因为这样King 1和Queen 1需要在控制敌方国王的同时保护自己。
    '''
    i = int(np.ceil(np.random.rand() * len(c)) - 1)
    s.reshape([-1])[c[i]] = k2
    s = s.reshape([size_board, size_board])
    p_k2 = np.concatenate(np.where(s == k2))

    return np.array(s), np.array(p_k2), np.array(p_k1), np.array(p_q1)