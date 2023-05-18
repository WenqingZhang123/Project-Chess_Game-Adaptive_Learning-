import numpy as np


def degree_freedom_king1(p_k1, p_k2, p_q1, s):
    """
    This function returns a matrix of ones and zeros where 1 specify a location in which the King can move to. The king
    will never choose an unsafe location.
    :param p_k1: position of King 1
    :param p_k2: position of King 2
    :param p_q1: position of Queen
    :param s: board
    :return: dfK1: Degrees of Freedom of King 1, a_k1: Allowed actions for King 1, dfK1_: Squares the King1 is threatening
    """
    # 首先，下一步得没有被q占领
    # 返回的三个参数要牢记：下面这3个是协同的，表达的含义一样，但是数据类型不一样
    # dfK1是个棋盘: 其中中心的位置为0(但是表示K1当前位置)；K1下一步的所有可能位置是否可以放的情况，k1下一步的位置可以放为1，不可以放为0
    # a_k1：表示K1当前位置的8方向的允许情况---每个位置为1则表示该位置可以被操作；为0则表示该位置不可以被操作
    # dfK1_：受到k1威胁的位置(受到k1威胁，不代表k1可以下一步走，因为不确定是否受到k2下一步的威胁)


    size_board = s.shape[0]

    # dfK1是个棋盘
    dfK1 = np.zeros([size_board, size_board], dtype=int)

    # 把k1的位置填到棋盘dfK1上
    dfK1[p_k1[0], p_k1[1]] = 1
    # King without King 2 reach
    dfK1_= np.zeros([size_board, size_board], dtype=int)

    # 把q1的位置填到棋盘dfK1_上
    dfK1_[p_q1[0], p_q1[1]] = 1

    # King 2 reach -- k2可以到达的总自由度
    # 说明第一个元素是行--表示纵坐标，第二个元素是列--表示横坐标
    k2r = [[p_k2[0] - 1, p_k2[1]],  # up
           [p_k2[0] + 1, p_k2[1]],  # down
           [p_k2[0], p_k2[1] - 1],  # left
           [p_k2[0], p_k2[1] + 1],  # right
           [p_k2[0] - 1, p_k2[1] - 1],  # up-left
           [p_k2[0] - 1, p_k2[1] + 1],  # up-right
           [p_k2[0] + 1, p_k2[1] - 1],  # down-left
           [p_k2[0] + 1, p_k2[1] + 1]]  # down-right
    # K2下一步可以到达的位置
    k2r = np.array(k2r)

    # King 1的8个方向，起始为0就是都不可以。
    # ps:变成1才可以放
    a_k1 = np.zeros([8, 1], dtype=int)

    '''
    逻辑解释：
    1.判断k1的下一步是否有q
    2.没有q，则判断这一步是否被k2威胁
    3.若没有被k2威胁，则k1下一步可以占领该地盘
    '''
    # allow_down = 0
    if p_k1[0] < (size_board - 1):
        # k1的位置不和q1的位置不重合
        if p_k1[0] + 1 != p_q1[0] or p_k1[1] != p_q1[1]:
            # k1当前的位置，威胁下面的位置
            dfK1_[p_k1[0] + 1, p_k1[1]] = 1
            # It is not the Queen's position
            tmp = np.zeros([k2r.shape[0]], dtype=int)
            for i in range(k2r.shape[0]):
                if p_k1[0] + 1 != k2r[i, 0] or p_k1[1] != k2r[i, 1]:
                    tmp[i] = 1
            # check if it will be within the reach of King 2
            if np.all(tmp):
                dfK1[p_k1[0] + 1, p_k1[1]] = 1
                a_k1[0] = 1

    # allow_up = 0
    if p_k1[0] > 0:
        if p_k1[0] - 1 != p_q1[0] or p_k1[1] != p_q1[1]:
            dfK1_[p_k1[0] - 1, p_k1[1]] = 1
            # It is not the Queen's position
            tmp = np.zeros([k2r.shape[0]], dtype=int)
            for i in range(k2r.shape[0]):
                if p_k1[0] - 1 != k2r[i, 0] or p_k1[1] != k2r[i, 1]:
                    tmp[i] = 1

            # check if it will be within the reach of King 2
            if np.all(tmp):
                dfK1[p_k1[0] - 1, p_k1[1]] = 1
                a_k1[1] = 1

    # allow_right = 0
    if p_k1[1] < (size_board - 1):
        if p_k1[0] != p_q1[0] or p_k1[1] + 1 != p_q1[1]:
            dfK1_[p_k1[0], p_k1[1] + 1] = 1
            # It is not the Queen's position
            tmp = np.zeros([k2r.shape[0]], dtype=int)
            for i in range(k2r.shape[0]):
                if p_k1[0] != k2r[i, 0] or p_k1[1] + 1 != k2r[i, 1]:
                    tmp[i] = 1

            # check if it will be within the reach of King 2
            if np.all(tmp):
                dfK1[p_k1[0], p_k1[1] + 1] = 1
                a_k1[2] = 1

    # allow_left = 0
    if p_k1[1] > 0:
        if p_k1[0] != p_q1[0] or p_k1[1] - 1 != p_q1[1]:
            dfK1_[p_k1[0], p_k1[1] - 1] = 1
            # It is not the Queen's position
            tmp = np.zeros([k2r.shape[0]], dtype=int)
            for i in range(k2r.shape[0]):
                if p_k1[0] != k2r[i, 0] or p_k1[1] - 1 != k2r[i, 1]:
                    tmp[i] = 1

            # check if it will be within the reach of King 2
            if np.all(tmp):
                dfK1[p_k1[0], p_k1[1] - 1] = 1
                a_k1[3] = 1

    # allow_down_right = 0
    if p_k1[0] < (size_board - 1) and p_k1[1] < (size_board - 1):
        if p_k1[0] + 1 != p_q1[0] or p_k1[1] + 1 != p_q1[1]:
            dfK1_[p_k1[0] + 1, p_k1[1] + 1] = 1
            # It is not the Queen's position
            tmp = np.zeros([k2r.shape[0]], dtype=int)
            for i in range(k2r.shape[0]):
                if p_k1[0] + 1 != k2r[i, 0] or p_k1[1] + 1 != k2r[i, 1]:
                    tmp[i] = 1

            # check if it will be within the reach of King 2
            if np.all(tmp):
                dfK1[p_k1[0] + 1, p_k1[1] + 1] = 1
                a_k1[4] = 1

    # allow_down_left = 0
    if p_k1[0] < (size_board - 1) and p_k1[1] > 0:
        if p_k1[0] + 1 != p_q1[0] or p_k1[1] - 1 != p_q1[1]:
            dfK1_[p_k1[0] + 1, p_k1[1] - 1] = 1
            # It is not the Queen's position
            tmp = np.zeros([k2r.shape[0]], dtype=int)
            for i in range(k2r.shape[0]):
                if p_k1[0] + 1 != k2r[i, 0] or p_k1[1] - 1 != k2r[i, 1]:
                    tmp[i] = 1

            # check if it will be within the reach of King 2
            if np.all(tmp):
                dfK1[p_k1[0] + 1, p_k1[1] - 1] = 1
                a_k1[5] = 1

    # allow_up_right = 0
    if p_k1[0] > 0 and p_k1[1] < size_board - 1:
        if p_k1[0] - 1 != p_q1[0] or p_k1[1] + 1 != p_q1[1]:
            dfK1_[p_k1[0] - 1, p_k1[1] + 1] = 1
            # It is not the Queen's position
            tmp = np.zeros([k2r.shape[0]], dtype=int)
            for i in range(k2r.shape[0]):
                if p_k1[0] - 1 != k2r[i, 0] or p_k1[1] + 1 != k2r[i, 1]:
                    tmp[i] = 1

            # check if it will be within the reach of King 2
            if np.all(tmp):
                dfK1[p_k1[0] - 1, p_k1[1] + 1] = 1
                a_k1[6] = 1

    # allow_up_left = 0
    if p_k1[0] > 0 and p_k1[1] > 0:
        if p_k1[0] - 1 != p_q1[0] or p_k1[1] - 1 != p_q1[1]:
            dfK1_[p_k1[0] - 1, p_k1[1] - 1] = 1
            # It is not the Queen's position
            tmp = np.zeros([k2r.shape[0]], dtype=int)
            for i in range(k2r.shape[0]):
                if p_k1[0] - 1 != k2r[i, 0] or p_k1[1] - 1 != k2r[i, 1]:
                    tmp[i] = 1

            # check if it will be within the reach of King 2
            if np.all(tmp):
                dfK1[p_k1[0] - 1, p_k1[1] - 1] = 1
                a_k1[7] = 1

    # previous location
    # 把K1的位置恢复成0
    dfK1[p_k1[0], p_k1[1]] = 0

    return dfK1, a_k1, dfK1_
