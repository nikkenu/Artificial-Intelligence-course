"""
Tic Tac Toe Player
"""

import copy

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    x_amount, o_amount = 0,0
    for row in board:
        for square in row:
            if square is X:
                x_amount += 1
            elif square is O:
                o_amount += 1
            else:
                # Square is empty so continue loop.
                pass
    return X if x_amount <= o_amount else O


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    possible_actions = set()
    for row_index in range(len(board)):
        for column_index in range(len(board[row_index])):
            if board[row_index][column_index] is EMPTY:
                possible_actions.add((row_index,column_index))

    return possible_actions


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    new_board = copy.deepcopy(board)

    if action not in actions(board):
        raise Exception("Current action is not allowed.")
    else:
        new_board[action[0]][action[1]] = player(board)
        return new_board

def diagonal_winner_check(board, player):
    """
    Check only dialogal lines
    """
    top_left_as_list = [board[0][0], board[1][1], board[2][2]]
    top_right_as_list = [board[0][2], board[1][1], board[2][0]]
    if all(x == player for x in top_left_as_list) or all(x == player for x in top_right_as_list):
        return True
    return False

def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    for player in X,O:
        for row in board:
            if len(set(row)) == player:
                return player
        if diagonal_winner_check(board, player):
            return player
    return None

def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    if all(all(square is not EMPTY for square in row) for row in board):
        return True
    
    if winner(board):
        return True
    
    return False

def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    if winner(board) is X:
        return 1
    elif winner(board) is O:
        return -1
    else:
        return 0

def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    return alphabeta(board, float('-inf'), float('inf'))[0]

def alphabeta(board, alpha, beta):
    """
    Returns a list, which contains "best" move and utility score.
    """
    if terminal(board):
        return [None, utility(board)]
    
    if player(board) is X:
        return maximizing_player(board, alpha, beta)
    else:
        return minimizing_player(board, alpha, beta)

def maximizing_player(board, alpha, beta):
    """
    For maximizing player. Returns best value and action.
    """
    best_value = [None, -2]
    for action in actions(board):
        current_value = alphabeta(result(board, action), alpha, beta)
        current_value[0] = action

        if current_value[1] > best_value[1]:
            best_value = current_value
        if best_value[1] >= beta:
            return best_value
        if best_value[1] > alpha:
            alpha = best_value[1]

    return best_value

def minimizing_player(board, alpha, beta):
    """
    For minimizing player. Returns best value and action.
    """
    best_value = [None, 2]
    for action in actions(board):
        current_value = alphabeta(result(board, action), alpha, beta)
        current_value[0] = action

        if current_value[1] < best_value[1]:
            best_value = current_value
        if best_value[1] <= alpha:
            return best_value
        if best_value[1] < beta:
            beta = best_value[1]

    return best_value