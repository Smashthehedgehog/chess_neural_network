import chess
import numpy as np

def create_tensor(chess_board):
    """
    Function to create a tensor based on the current layout of the chess pieces to feed into an ML Model:
    input: Board object
    output: numpy array of size (12,8,8)
    """
    chess_tensor = np.zeros((12,8,8))
    dim3 = 12
    dim2 = 8
    dim1 = 8

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
    for i in range(dim3):

        # Get the set of pieces that exist on the board for that specific piece type and color
        piece_type, color = layer_to_piece_type[i]
        piece_pos = set(new_game.pieces(piece_type, color))


        tracker = 0
        for j in range(dim2-1,-1,-1):
            for k in range(dim1):
                chess_tensor[i][j][k] = tracker in piece_pos
                tracker += 1

    return chess_tensor

new_game = chess.Board()

print(create_tensor(new_game))



