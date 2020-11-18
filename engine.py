import chess
import random

class Engine():
    def __init__(self, board: chess.Board):
        self._board = board

    def get_move(self) -> chess.Move:
        return random.choice(
            [move for move in self._board.generate_legal_moves()])

    