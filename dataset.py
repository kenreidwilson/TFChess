import os
import chess
import chess.pgn
import numpy as np

def get_dataset(n: int = None):
    board_states, evaluations = [], []
    game_number = 0

    for fn in os.listdir("data"):
        pgn = open(os.path.join("data", fn))

        while True:
            game = chess.pgn.read_game(pgn)
            if game is None:
                break

            board = game.board()
            elo = int(game.headers['WhiteElo'])
            for _, move in enumerate(game.mainline_moves()):
                board.push(move)
                board_states.append(generate_bitboard(board))
                evaluations.append(evaluate(board, elo))

            print("parsing game %d, got %d total moves" %
                  (game_number, len(board_states)))
            if n is not None and len(board_states) > n:
                return np.array(board_states), np.array(evaluations)
            game_number += 1

    return np.array(board_states), np.array(evaluations)

def evaluate(board: chess.Board, elo: int = 0) -> float:
    values = {chess.PAWN: 1,
              chess.KNIGHT: 3,
              chess.BISHOP: 3,
              chess.ROOK: 5,
              chess.QUEEN: 9,
              chess.KING: 0}

    val = 0.0

    pm = board.piece_map()
    for x in pm:
        tval = values[pm[x].piece_type]
        if pm[x].color == chess.WHITE:
            val += tval
        else:
            val -= tval

    return val

def generate_bitboard(board: chess.Board):
    state = np.zeros((6, 8, 8), np.int8)
    piece_dict = {"P": 0, "N": 1, "B": 2, "R": 3, "Q": 4, "K": 5, \
                     "p": 0, "n":1, "b":2, "r":3, "q":4, "k": 5}

    for x in range(8):
        for y in range(8):
            piece = board.piece_at(y * 8 + x)
            if piece is None:
                continue
            
            if piece.symbol().isupper():
                state[piece_dict[piece.symbol()]][y][x] = 1
            else:
                state[piece_dict[piece.symbol()]][y][x] = -1

    return state


if __name__ == "__main__":
    X, Y = get_dataset(100000)
    np.savez("dataset.npz", X, Y)
