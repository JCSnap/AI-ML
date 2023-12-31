import utils

Score = int | float
Move = tuple[tuple[int, int], tuple[int, int]]

def evaluate(board):
    bcount = 0
    wcount = 0
    for r, row in enumerate(board):
        for tile in row:
            if tile == 'B':
                if r == 5:
                    return utils.WIN
                bcount += 1
            elif tile == 'W':
                if r == 0:
                    return -utils.WIN
                wcount += 1
    if wcount == 0:
        return utils.WIN
    if bcount == 0:
        return -utils.WIN
    return bcount - wcount

def generate_valid_moves(board):
    '''
    Generates a list (or iterable, if you want to) containing
    all possible moves in a particular position for black.
    '''
    valid_moves = []
    for r in range(len(board)):
        for c in range(len(board[0])):
            piece = board[r][c]
            if piece != 'B':
                continue
            src = r, c
            for d in (-1, 0, 1):
                dst = r + 1, c + d
                if utils.is_valid_move(board, src, dst):
                    valid_moves.append((src, dst))
    return valid_moves

def test_11():
    board1 = [
        ['_', '_', 'B', '_', '_', '_'],
        ['_', '_', '_', '_', '_', '_'],
        ['_', '_', '_', '_', '_', '_'],
        ['_', '_', '_', '_', '_', '_'],
        ['_', '_', '_', '_', '_', '_'],
        ['_', '_', '_', '_', '_', '_']
    ]
    assert sorted(generate_valid_moves(board1)) == [((0, 2), (1, 1)), ((0, 2), (1, 2)), ((0, 2), (1, 3))]

    board2 = [
        ['_', '_', '_', '_', 'B', '_'],
        ['_', '_', '_', '_', '_', 'B'],
        ['_', 'W', '_', '_', '_', '_'],
        ['_', '_', 'W', '_', '_', '_'],
        ['_', '_', '_', '_', '_', '_'],
        ['_', '_', 'B', '_', '_', '_']
    ]
    assert sorted(generate_valid_moves(board2)) == [((0, 4), (1, 3)), ((0, 4), (1, 4)), ((1, 5), (2, 4)), ((1, 5), (2, 5))]

    board3 = [
        ['_', '_', '_', '_', '_', '_'],
        ['_', '_', 'B', '_', '_', '_'],
        ['_', 'W', 'W', 'W', '_', '_'],
        ['_', '_', '_', '_', '_', '_'],
        ['_', '_', '_', '_', '_', '_'],
        ['_', '_', '_', '_', '_', '_']
    ]
    assert sorted(generate_valid_moves(board3)) == [((1, 2), (2, 1)), ((1, 2), (2, 3))]

# test_11()

def minimax(board, depth, max_depth, is_black: bool) -> tuple[Score, Move]:
    '''
    Finds the best move for the input board state.
    Note that you are black.

    Parameters
    ----------
    board: 2D list of lists. Contains characters 'B', 'W' and '_'
    representing black pawn, white pawn and empty cell, respectively.

    depth: int, the depth to search for the best move. When this is equal
    to `max_depth`, you should get the evaluation of the position using
    the provided heuristic function.

    max_depth: int, the maximum depth for cutoff.

    is_black: bool. True when finding the best move for black, False
    otherwise.

    Returns
    -------
    A tuple (evalutation, ((src_row, src_col), (dst_row, dst_col))):
    evaluation: the best score that black can achieve after this move.
    src_row, src_col: position of the pawn to move.
    dst_row, dst_col: position to move the pawn to.
    '''
    if depth == max_depth:
        return evaluate(board), None
    moves = generate_valid_moves(board)
    best_move = None
    v = -utils.INF if is_black else utils.INF
    for move in moves:
        updated_board = utils.state_change(board, move[0], move[1], in_place=False)
        inverted_board = utils.invert_board(updated_board, in_place=False)
        ## We do this because the evaluate function needs the board to be at the perspective of black, even though
        ## we invert white to black everytime
        board_to_check = updated_board if is_black else inverted_board
        if utils.is_game_over(board_to_check):
            return evaluate(board_to_check), move
        next = minimax(inverted_board, depth + 1, max_depth)
        if is_black and next[0] > v or not is_black and next[0] < v:
            v = next[0]
            best_move = move
    return (v, best_move)

def test_21():
    board1 = [
        list("______"),
        list("___B__"),
        list("____BB"),
        list("___WB_"),
        list("_B__WW"),
        list("_WW___"),
    ]
    #score1, _ = minimax(board1, 0, 1, True)
    #assert score1 == utils.WIN, "black should win in 1"

    board2 = [
        list("______"),
        list("___B__"),
        list("____BB"),
        list("_BW_B_"),
        list("____WW"),
        list("_WW___"),
    ]
    #score2, _ = minimax(board2, 0, 3, True)
    #assert score2 == utils.WIN, "black should win in 3"

    board3 = [
        list("______"),
        list("__B___"),
        list("_WWW__"),
        list("______"),
        list("______"),
        list("______"),
    ]
    score3, _ = minimax(board3, 0, 4, True)
    assert score3 == -utils.WIN, "white should win in 4"

# test_21()

def negamax(board, depth, max_depth) -> tuple[Score, Move]:
    '''
    Finds the best move for the input board state.
    Note that you are black.

    Parameters
    ----------
    board: 2D list of lists. Contains characters 'B', 'W' and '_'
    representing black pawn, white pawn and empty cell, respectively.

    depth: int, the depth to search for the best move. When this is equal
    to `max_depth`, you should get the evaluation of the position using
    the provided heuristic function.

    max_depth: int, the maximum depth for cutoff.

    Notice that you no longer need the parameter `is_black`.

    Returns
    -------
    A tuple (evalutation, ((src_row, src_col), (dst_row, dst_col))):
    evaluation: the best score that black can achieve after this move.
    src_row, src_col: position of the pawn to move.
    dst_row, dst_col: position to move the pawn to.
    '''
    if depth == max_depth:
        return evaluate(board), None
    moves = generate_valid_moves(board)
    best_move = None
    is_black = depth % 2 == 0
    v = -utils.INF
    for move in moves:
        updated_board = utils.state_change(board, move[0], move[1], in_place=False)
        inverted_board = utils.invert_board(updated_board, in_place=False)
        board_to_check = updated_board if is_black else inverted_board
        if utils.is_game_over(board_to_check):
            ## If current is a terminal white move, we want to invert the score
            return (evaluate(board_to_check), move) if is_black else (-evaluate(board_to_check), move)
        next_val = -negamax(inverted_board, depth + 1, max_depth)[0]
        if next_val > v:
            v = next_val
            best_move = move

    return (v, best_move)

def test_22():
    board1 = [
        list("______"),
        list("___B__"),
        list("____BB"),
        list("___WB_"),
        list("_B__WW"),
        list("_WW___"),
    ]
    score1, _ = negamax(board1, 0, 1)
    assert score1 == utils.WIN, "black should win in 1"

    board2 = [
        list("______"),
        list("___B__"),
        list("____BB"),
        list("_BW_B_"),
        list("____WW"),
        list("_WW___"),
    ]
    score2, _ = negamax(board2, 0, 3)
    assert score2 == utils.WIN, "black should win in 3"

    board3 = [
        list("______"),
        list("__B___"),
        list("_WWW__"),
        list("______"),
        list("______"),
        list("______"),
    ]
    score3, _ = negamax(board3, 0, 4)
    print(score3)
    assert score3 == -utils.WIN, "white should win in 4"

# test_22()

def minimax_alpha_beta(board, depth, max_depth, alpha, beta, is_black: bool) -> tuple[Score, Move]:
    if depth == max_depth:
        return evaluate(board), None
    moves = generate_valid_moves(board)
    best_move = None
    v = -utils.INF if is_black else utils.INF
    for move in moves:
        updated_board = utils.state_change(board, move[0], move[1], in_place=False)
        inverted_board = utils.invert_board(updated_board, in_place=False)
        ## We do this because the evaluate function needs the board to be at the perspective of black, even though
        ## we invert white to black everytime
        board_to_check = updated_board if is_black else inverted_board
        if utils.is_game_over(board_to_check):
            return evaluate(board_to_check), move
        next = minimax_alpha_beta(inverted_board, depth + 1, max_depth, alpha, beta, not is_black)
        if is_black:
            if next[0] > v:
                v = next[0]
                best_move = move
                alpha = max(alpha, v)
            if v >= beta:
                return (v, best_move)
        else:
            if next[0] < v:
                v = next[0]
                best_move = move
                beta = min(beta, v)
            if v <= alpha:
                return (v, best_move)
    return (v, best_move)

def test_31():
    board1 = [
        list("______"),
        list("__BB__"),
        list("____BB"),
        list("WBW_B_"),
        list("____WW"),
        list("_WW___"),
    ]
    score1, _ = minimax_alpha_beta(board1, 0, 3, -utils.INF, utils.INF, True)
    assert score1 == utils.WIN, "black should win in 3"

    board2 = [
        list("____B_"),
        list("___B__"),
        list("__B___"),
        list("_WWW__"),
        list("____W_"),
        list("______"),
    ]
    score2, _ = minimax_alpha_beta(board2, 0, 5, -utils.INF, utils.INF, True)
    assert score2 == utils.WIN, "black should win in 5"

    board3 = [
        list("____B_"),
        list("__BB__"),
        list("______"),
        list("_WWW__"),
        list("____W_"),
        list("______"),
    ]
    score3, _ = minimax_alpha_beta(board3, 0, 6, -utils.INF, utils.INF, True)
    assert score3 == -utils.WIN, "white should win in 6"

# test_31()

def negamax_alpha_beta(board, depth, max_depth, alpha, beta) -> tuple[Score, Move]:
    if depth == max_depth:
        return evaluate(board), None
    moves = generate_valid_moves(board)
    best_move = None
    is_black = depth % 2 == 0
    v = -utils.INF
    for move in moves:
        updated_board = utils.state_change(board, move[0], move[1], in_place=False)
        inverted_board = utils.invert_board(updated_board, in_place=False)
        board_to_check = updated_board if is_black else inverted_board
        if utils.is_game_over(board_to_check):
            ## If current is a terminal white move, we want to invert the score
            return (evaluate(board_to_check), move) if is_black else (-evaluate(board_to_check), move)
        next_val = -negamax_alpha_beta(inverted_board, depth + 1, max_depth, -beta, -alpha)[0]
        if next_val > v:
            v = next_val
            best_move = move
            alpha = max(alpha, v)
        if v >= beta:
            return (v, best_move)

    return (v, best_move)

def test_32():
    board1 = [
        list("______"),
        list("__BB__"),
        list("____BB"),
        list("WBW_B_"),
        list("____WW"),
        list("_WW___"),
    ]
    score1, _ = negamax_alpha_beta(board1, 0, 3, -utils.INF, utils.INF)
    assert score1 == utils.WIN, "black should win in 3"

    board2 = [
        list("____B_"),
        list("___B__"),
        list("__B___"),
        list("_WWW__"),
        list("____W_"),
        list("______"),
    ]
    score2, _ = negamax_alpha_beta(board2, 0, 5, -utils.INF, utils.INF)
    assert score2 == utils.WIN, "black should win in 5"

    board3 = [
        list("____B_"),
        list("__BB__"),
        list("______"),
        list("_WWW__"),
        list("____W_"),
        list("______"),
    ]
    score3, _ = negamax_alpha_beta(board3, 0, 6, -utils.INF, utils.INF)
    assert score3 == -utils.WIN, "white should win in 6"

# test_32()

# Uncomment and implement the function.
# Note: this will override the provided `evaluate` function.
# 0: 10     
# 1: 10     -50
# 2: 20     -40
# 3: 40     -20
# 4: 50     -10
# 5: 50     -10
def evaluate(board):
    bcount = 0
    bpoints = 0
    wcount = 0
    wpoints = 0
    bpoints_constant = [10, 10, 20, 40, 50]
    wpoints_constant = [0, -50, -40, -20, -10, -10]
    for r, row in enumerate(board):
        for tile in row:
            if tile == 'B':
                if r == 5:
                    return utils.WIN
                else:
                    bpoints += bpoints_constant[r]
                bcount += 1
            elif tile == 'W':
                if r == 0:
                    return -utils.WIN
                else:
                    wpoints += wpoints_constant[r]
                wcount += 1
    if wcount == 0:
        return utils.WIN
    if bcount == 0:
        return -utils.WIN
    return bpoints + wpoints

def test_41():
    board1 = [
        ['_', '_', '_', 'B', '_', '_'],
        ['_', '_', '_', 'W', '_', '_'],
        ['_', '_', '_', '_', '_', '_'],
        ['_', '_', 'B', '_', '_', '_'],
        ['_', '_', '_', '_', '_', '_'],
        ['_', '_', '_', '_', '_', '_']
    ]
    assert evaluate(board1) == 0

    board2 = [
        ['_', '_', '_', 'B', 'W', '_'],
        ['_', '_', '_', 'W', '_', '_'],
        ['_', '_', '_', '_', '_', '_'],
        ['_', '_', '_', '_', '_', '_'],
        ['_', '_', '_', '_', '_', '_'],
        ['_', '_', '_', '_', '_', '_']
    ]
    assert evaluate(board2) == -utils.WIN

    board3 = [
        ['_', '_', '_', '_', '_', '_'],
        ['_', '_', '_', '_', '_', '_'],
        ['_', '_', '_', '_', '_', '_'],
        ['_', '_', 'B', '_', '_', '_'],
        ['_', '_', '_', '_', '_', '_'],
        ['_', '_', '_', '_', '_', '_']
    ]
    assert evaluate(board3) == utils.WIN

test_41()

class PlayerAI:

    def make_move(self, board) -> Move:
        '''
        This is the function that will be called from main.py
        You should combine the functions in the previous tasks
        to implement this function.

        Parameters
        ----------
        self: object instance itself, passed in automatically by Python.
        
        board: 2D list-of-lists. Contains characters 'B', 'W', and '_' 
        representing black pawn, white pawn and empty cell respectively.
        
        Returns
        -------
        Two tuples of coordinates [row_index, col_index].
        The first tuple contains the source position of the black pawn
        to be moved, the second list contains the destination position.
        '''
        # TODO: Replace starter code with your AI
        ################
        # Starter code #
        ################
        for r in range(len(board)):
            for c in range(len(board[r])):
                # check if B can move forward directly
                if board[r][c] == 'B' and board[r + 1][c] == '_':
                    src = r, c
                    dst = r + 1, c
                    return src, dst # valid move
        return (0, 0), (0, 0) # invalid move

class PlayerNaive:
    '''
    A naive agent that will always return the first available valid move.
    '''
    def make_move(self, board):
        return utils.generate_rand_move(board)

##########################
# Game playing framework #
##########################
if __name__ == "__main__":
    assert utils.test_move([
        ['B', 'B', 'B', 'B', 'B', 'B'], 
        ['_', 'B', 'B', 'B', 'B', 'B'], 
        ['_', '_', '_', '_', '_', '_'], 
        ['_', 'B', '_', '_', '_', '_'], 
        ['_', 'W', 'W', 'W', 'W', 'W'], 
        ['W', 'W', 'W', 'W', 'W', 'W']
    ], PlayerAI())

    assert utils.test_move([
        ['_', 'B', 'B', 'B', 'B', 'B'], 
        ['_', 'B', 'B', 'B', 'B', 'B'], 
        ['_', '_', '_', '_', '_', '_'], 
        ['_', 'B', '_', '_', '_', '_'], 
        ['W', 'W', 'W', 'W', 'W', 'W'], 
        ['_', '_', 'W', 'W', 'W', 'W']
    ], PlayerAI())

    assert utils.test_move([
        ['_', '_', 'B', 'B', 'B', 'B'], 
        ['_', 'B', 'B', 'B', 'B', 'B'], 
        ['_', '_', '_', '_', '_', '_'], 
        ['_', 'B', 'W', '_', '_', '_'], 
        ['_', 'W', 'W', 'W', 'W', 'W'], 
        ['_', '_', '_', 'W', 'W', 'W']
    ], PlayerAI())

    # generates initial board
    board = utils.generate_init_state()
    res = utils.play(PlayerAI(), PlayerNaive(), board)
    # Black wins means your agent wins.
    print(res)
