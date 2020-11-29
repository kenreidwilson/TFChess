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

            print("parsing game %d, got %d total moves" % (game_number, len(board_states)))
            if n is not None and len(board_states) > n:
                return np.array(board_states), np.array(evaluations)
            game_number += 1

    return np.array(board_states), np.array(evaluations)


def evaluate(board: chess.Board, elo: int = 0) -> float:
    # Stolen Evaluation Function, only used for training.
    MAXVAL = 100
    values = {chess.PAWN: 1,
              chess.KNIGHT: 3,
              chess.BISHOP: 3,
              chess.ROOK: 5,
              chess.QUEEN: 9,
              chess.KING: 0}

    val = 0.0

    if board.is_game_over():
        if board.result() == "1-0":
            return MAXVAL
        elif board.result() == "0-1":
            return -MAXVAL
        else:
            return 0

    pm = board.piece_map()
    for x in pm:
        tval = values[pm[x].piece_type]
        if pm[x].color == chess.WHITE:
            val += tval
        else:
            val -= tval

    return val


def generate_bitboard(board: chess.Board):
    bstate = np.zeros(64, np.float)
    piece_dict = {"P": 1, "N": 2, "B": 3, "R": 4, "Q": 5, "K": 6,
                  "p": -1, "n": -2, "b": -3, "r": -4, "q": -5, "k": -6}

    for i in range(64):
        piece = board.piece_at(i)
        if piece is None:
            continue

        bstate[i] = piece_dict[piece.symbol()]

    return bstate


if __name__ == "__main__":
    X, Y = get_dataset(100000)
    np.savez("processed/dataset.npz", X, Y)
