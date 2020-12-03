import chess
import numpy as np
from dataset import generate_bitboard
from network import NNetwork

class Engine():
    def __init__(self):
        self.network = NNetwork()
        self.network.load_model()

    def get_move(self, board: chess.Board) -> chess.Move:
        # TODO: Implement Minimax to explore more than a single move ahead.
        possible_moves = []
        board = board.copy()
        for move in board.legal_moves:
            board.push(move)
            possible_moves.append((self.evaluate(board), move))
            board.pop()
        sorted_moves = sorted(possible_moves, key=lambda x: x[0], reverse=board.turn)
        for move in sorted_moves[:3]:
            print(f'Move: {move[1]}, evaluated at: {move[0]}')
        return sorted_moves[0][1]

    def evaluate(self, board: chess.Board) -> float:
        bitboard = generate_bitboard(board)
        return self.network.model.predict(np.array([bitboard]))[0][0]
