import chess
import numpy as np
import chess.pgn

import tensorflow as tf
from tensorflow.keras import layers, models

def create_tensor(chess_board):
    """
    Function to create a tensor based on the current layout of the chess pieces to feed into an ML Model:
    input: Board object
    output: numpy array of size (8,8,12)
    """
    chess_tensor = np.zeros((8, 8, 12))

    # Define (layer index) -> (piece_type, color) mapping
    # chess.PAWN=1, chess.KNIGHT=2, ..., chess.KING=6; chess.WHITE=True, chess.BLACK=False
    layer_to_piece_type = {}
    for color in [chess.WHITE, chess.BLACK]:
        for piece_type in range(1, 7):  # chess.PAWN==1, chess.KING==6
            if color == chess.WHITE:
                layer = piece_type - 1  # 0-indexed layers for white
            else:
                layer = piece_type - 1 + 6  # 6-11 for black
            layer_to_piece_type[layer] = (piece_type, color)

    # Fill in the tensors with the state of each tile in the chess board (what type of piece is on that
    # tile or if there is a tile on that piece at all)
    for i in range(12):  # piece layers
        piece_type, color = layer_to_piece_type[i]
        piece_pos = set(chess_board.pieces(piece_type, color))

        tracker = 0
        # Loop through squares and assign to the 3rd dimension [row][col][layer]
        for j in range(7, -1, -1):  # Ranks (dim2)
            for k in range(8):      # Files (dim1)
                if tracker in piece_pos:
                    # Direct assignment to the last dimension
                    chess_tensor[j][k][i] = 1 
                tracker += 1

    return chess_tensor

def create_training_data(filename='lichess_db_standard_rated_2013-07.pgn', num_games=None):
    """
    Loads chess games from a PGN file, creates training data arrays X (features) and y (labels),
    and saves them to 'X.npy' and 'y.npy'.
    
    :param filename: PGN file name to load games from. If None, defaults to 'lichess_db_standard_rated_2013-07.pgn'.
    :param num_games: Number of games to load. If None, loads all games in the file.
    """
    with open(filename) as chess_data:
        game_count = 0
        X_list = []
        y_list = []

        while True:
            if num_games is not None and game_count >= num_games:
                break
            game = chess.pgn.read_game(chess_data)
            if game is None:
                break  # End of file
            result = game.headers['Result']
            if result == '1-0':
                y = 1
            elif result == '0-1':
                y = -1
            else:
                y = 0

            board = game.board()
            for move in game.mainline_moves():
                board.push(move)
                X_list.append(create_tensor(board))
                y_list.append(y)

            game_count += 1

        X = np.array(X_list)
        y = np.array(y_list)
        np.save('X.npy', X)
        np.save('y.npy', y)

def create_training_data_v2(filename='lichess_db_standard_rated_2013-07.pgn', num_games=None):
    """
    Loads chess games from a PGN file, creates training data arrays X (features) and y (labels),
    and saves them to 'X_v2.npy' and 'y_v2.npy'.

    :param filename: PGN file name to load games from. If None, defaults to 'lichess_db_standard_rated_2013-07.pgn'.
    :param num_games: Number of games to load. If None, loads all games in the file.
    :result: Vector of 1 or -1 depending on the moves that led to a check/checkmate.
             1 = positions leading up to White check/checkmate
             -1 = positions leading up to Black check/checkmate
             0 = positions that don't lead to any check/checkmate
    """
    with open(filename) as chess_data:
        game_count = 0
        X_list = []
        y_list = []

        while True:
            if num_games is not None and game_count >= num_games:
                break
            game = chess.pgn.read_game(chess_data)
            if game is None:
                break  # End of file

            board = game.board()
            game_boards = []  # Store board states for this game
            game_labels = []  # Store labels for each position (initially None)
            
            # First pass: collect all board states and detect checks/checkmates
            for move in game.mainline_moves():
                board.push(move)
                game_boards.append(create_tensor(board))
                
                # Check if this move resulted in a check or checkmate
                if board.is_check() or board.is_checkmate():
                    # Determine who delivered the check/checkmate (the player who just moved)
                    # After board.push(move), board.turn has switched to the opponent
                    # So if board.turn is BLACK, WHITE just moved and delivered check
                    if board.turn == chess.BLACK:
                        # White delivered check/checkmate
                        game_labels.append(1)
                    else:
                        # Black delivered check/checkmate
                        game_labels.append(-1)
                else:
                    game_labels.append(None)  # Placeholder for non-check positions
            
            # Second pass: backfill labels - assign check labels to all positions leading up to each check
            last_check_index = -1
            
            for i in range(len(game_labels)):
                if game_labels[i] is not None:
                    # This is a check/checkmate position
                    check_label = game_labels[i]
                    
                    # Backfill all positions from last_check_index+1 to i (inclusive) with this label
                    for j in range(last_check_index + 1, i + 1):
                        game_labels[j] = check_label
                    
                    # Update tracking
                    last_check_index = i
            
            # Third pass: any remaining positions without labels get 0
            for i in range(len(game_labels)):
                if game_labels[i] is None:
                    game_labels[i] = 0
            
            # Add all positions from this game to the training data
            for i in range(len(game_boards)):
                X_list.append(game_boards[i])
                y_list.append(game_labels[i])

            print(game_labels)

            game_count += 1
            
            

        X = np.array(X_list)
        y = np.array(y_list)
        np.save('X_v2.npy', X)
        np.save('y_v2.npy', y)

board = chess.Board()

print(board.turn)

create_training_data_v2(num_games=10)


