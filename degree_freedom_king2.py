import numpy as np


def degree_freedom_king2(dfK1, p_k2, dfQ1_, s, p_k1):
    """
    Returns a matrix of ones and zeros where one specify a location in which the oppement king can move to
    :param dfK1: Degrees of freedom for King 1 ---- k2下一步不能走的位置
    :param p_k2: Position of King 2
    :param dfQ1: Degrees of freedom for the Queen --- k2下一步不能走的位置
    :param s: board
    :param p_k1: Position of King1
    :return: dfK2: Degrees of freedom for King 2, a_k2: allowed actions for King 2, check: 1 if it is checked, -1 if not
    """
    # 牢记这几个参数：
    # dfK2：k2下一步能走的位置：能走则为1，不能走则为0
    # a_k2: k2的自由度
    # check：


    size_board = s.shape[0]

    dfK2 = np.zeros([size_board, size_board], dtype=int)
    dfK2[p_k2[0], p_k2[1]] = 1

    # King 2 reach
    k2r = [[p_k2[0] + 1, p_k2[1]],  # down
           [p_k2[0] - 1, p_k2[1]],  # up
           [p_k2[0], p_k2[1] + 1],  # right
           [p_k2[0], p_k2[1] - 1],  # left
           [p_k2[0] + 1, p_k2[1] + 1],  # down-right
           [p_k2[0] + 1, p_k2[1] - 1],  # down-left
           [p_k2[0] - 1, p_k2[1] + 1],  # up-right
           [p_k2[0] - 1, p_k2[1] - 1]]  # up-left
    k2r = np.array(k2r)

    a_k2 = np.zeros([8, 1], dtype=int)

    for i in range(k2r.shape[0]):
        if k2r[i, 0] <= -1 or k2r[i, 0] > size_board - 1 or k2r[i, 1] <= -1 or k2r[i, 1] > size_board - 1:
            continue
        else:
            # 考虑到该位置与国王1的距离超过1个单位（也就是国王1无法直接攻击到该位置）且国后1无法控制该位置。
            # 该位置与国王1的距离为1个单位，但国王1无法攻击到该位置。
            if (np.abs(k2r[i, 0] - p_k1[0]) > 1 or np.abs(k2r[i, 1] - p_k1[1]) > 1) and dfQ1_[k2r[i, 0], k2r[i, 1]] == 0:
                dfK2[k2r[i, 0], k2r[i, 1]] = 1
                a_k2[i] = 1

    dfK2[p_k2[0], p_k2[1]] = 0

    check = 0
    # 这一步和"该位置与国王1的距离为1个单位，但国王1无法攻击到该位置"，做配合
    # 重新考虑了，是否被queen所控制
    if dfQ1_[p_k2[0], p_k2[1]] == 1:
        check = 1

    return dfK2, a_k2, check