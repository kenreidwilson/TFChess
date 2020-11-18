import chess
from engine import Engine

game = chess.Board()
chess_engine = Engine(game)

while not game.is_game_over():
    print(game)
    print("Choose Move:")
    try:
        move = game.parse_san(input())
    except ValueError:
        print("Invalid Move")
        continue
    game.push(move)
    game.push(chess_engine.get_move())