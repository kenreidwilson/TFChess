import chess
import random
import numpy as np
from util import generate_bitboard

class Engine():
    def __init__(self, board: chess.Board):
        self._board = board

    def get_move(self) -> chess.Move:
        # TODO: Implemented Minimax to explore more than a single move ahead.
        possible_moves = []
        for move in self._board.legal_moves:
            self._board.push(move)
            possible_moves.append((self._evaluate(), move))
            self._board.pop()
        return sorted(possible_moves, key=lambda x: x[0], reverse=self._board.turn)[0][1]

    def _evaluate(self) -> int:
        # TODO: Load board into trained tensorflow model and get evaluation.
        return 0 # All board states are equal.
