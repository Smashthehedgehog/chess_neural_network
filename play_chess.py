import chess
import numpy as np

import tensorflow as tf

import pygame

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

def get_ai_move(board, model):
    """
    Function for the AI to decide on what move to make based on the current board state and the model.
    """
    best_move = None

    # If White to move, we start with a very low score; if Black, a very high one.
    best_value = -999 if board.turn == chess.WHITE else 999

    for move in board.legal_moves:
        # 1. Pretend to make the move
        board.push(move)
        
        # 2. Convert the new board state into your (8, 8, 12) tensor
        # (Using the function you wrote earlier)
        tensor = create_tensor(board) 
        
        # 3. Add a 'batch' dimension (1, 8, 8, 12) so Keras can read it
        input_tensor = np.expand_dims(tensor, axis=0)
        
        # 4. Ask the model "Who is winning now?"
        prediction = model.predict(input_tensor, verbose=0)[0][0]

        print(f"Prediction for {move}: {prediction}")

        # 5. Check if this is the best move found so far
        if board.turn == chess.BLACK: # We just made a White move, so it's now Black's turn to evaluate
            # If we are White, we want the HIGHEST prediction
            if prediction > best_value:
                best_value = prediction
                best_move = move
        else:
            # If we are Black, we want the LOWEST prediction
            if prediction < best_value:
                best_value = prediction
                best_move = move
        
        # 6. Undo the move to reset the board for the next check
        board.pop()
        
    return best_move

def get_ai_move_fast(board, model):
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None

    # 1. Build a list of ALL possible next board tensors
    tensors = []
    for move in legal_moves:
        board.push(move)
        tensors.append(create_tensor(board)) # Your 8x8x12 function
        board.pop()

    # 2. Convert list to a single 4D NumPy array (Batch, 8, 8, 12)
    batch_input = np.array(tensors)

    # 3. Predict ALL scores at once (This is significantly faster)
    predictions = model.predict(batch_input, verbose=0)

    # 4. Find the best index based on whose turn it is
    if board.turn == chess.WHITE:
        best_move_idx = np.argmax(predictions)
    else:
        best_move_idx = np.argmin(predictions)

    return legal_moves[best_move_idx]


SQUARE_SIZE = 100  # Based on an 800x800 window
BOARD_SIZE = 8 * SQUARE_SIZE

def load_images():
    pieces = ['white_pawn', 'white_knight', 'white_bishop', 'white_rook', 'white_queen', 'white_king', 'black_pawn', 'black_knight', 'black_bishop', 'black_rook', 'black_queen', 'black_king']
    images = {}
    for piece in pieces:
        # Load the 400x400 image
        img = pygame.image.load(f"images/{piece}.png")
        # Downscale it to fit your square size
        images[piece] = pygame.transform.smoothscale(img, (SQUARE_SIZE, SQUARE_SIZE))
    return images

def draw_board(screen):
    colors = [pygame.Color("white"), pygame.Color("gray")]
    for rank in range(8):
        for file in range(8):
            # If the sum of coordinates is even, it's a light square; if odd, dark.
            color = colors[((rank + file) % 2)]
            pygame.draw.rect(screen, color, pygame.Rect(file * SQUARE_SIZE, rank * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))

def get_piece_symbol(piece):
    """Convert a chess piece to its image key."""
    piece_color = "white" if piece.color == chess.WHITE else "black"
    piece_type_map = {
        chess.PAWN: "pawn",
        chess.KNIGHT: "knight",
        chess.BISHOP: "bishop",
        chess.ROOK: "rook",
        chess.QUEEN: "queen",
        chess.KING: "king"
    }
    piece_type = piece_type_map[piece.piece_type]
    return f"{piece_color}_{piece_type}"

def draw_pieces(screen, board, images, dragging_piece=None, dragging_from=None, mouse_pos=None):
    """Draw all pieces on the board, excluding the piece being dragged."""
    for square in chess.SQUARES:
        # Skip the square we're dragging from
        if dragging_from is not None and square == dragging_from:
            continue
            
        piece = board.piece_at(square)
        if piece:
            piece_symbol = get_piece_symbol(piece)
            
            # Convert chess square index to x, y coordinates
            file = chess.square_file(square)
            rank = chess.square_rank(square)
            
            # Draw the piece, flipping the rank so Rank 8 is at the top
            screen.blit(images[piece_symbol], (file * SQUARE_SIZE, (7 - rank) * SQUARE_SIZE))
    
    # Draw the dragging piece at mouse position
    if dragging_piece is not None and mouse_pos is not None:
        piece_symbol = get_piece_symbol(dragging_piece)
        # Center the piece on the mouse cursor
        x = mouse_pos[0] - SQUARE_SIZE // 2
        y = mouse_pos[1] - SQUARE_SIZE // 2
        screen.blit(images[piece_symbol], (x, y))

def highlight_square(screen, square, color):
    """Highlight a square on the board."""
    file = chess.square_file(square)
    rank = chess.square_rank(square)
    x = file * SQUARE_SIZE
    y = (7 - rank) * SQUARE_SIZE
    
    # Draw a semi-transparent overlay
    s = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE))
    s.set_alpha(128)
    s.fill(color)
    screen.blit(s, (x, y))

def get_square_from_mouse(pos):
    """Convert mouse position to chess square."""
    x, y = pos
    file = x // SQUARE_SIZE
    rank = 7 - (y // SQUARE_SIZE)
    
    # Ensure we're within bounds
    if 0 <= file < 8 and 0 <= rank < 8:
        return chess.square(file, rank)
    return None

# board = chess.Board()
# model = tf.keras.models.load_model('my_chess_model.keras')

# while not board.is_game_over():
#     if board.turn == chess.WHITE:
#         # Human Move
#         move_str = input("Enter move (e.g. e2e4): ")
#         board.push_san(move_str)
#     else:
#         # AI Move
#         print("AI is thinking...")
#         move = get_ai_move_fast(board, model)
#         board.push(move)
#         print(f"AI played: {move}")
    
#     print(board)

def main():
    pygame.init()
    screen = pygame.display.set_mode((BOARD_SIZE, BOARD_SIZE))
    pygame.display.set_caption("Chess Game - Drag and Drop")
    images = load_images()
    board = chess.Board()
    model = tf.keras.models.load_model('my_chess_model.keras')
    
    # Drag and drop state
    dragging = False
    dragging_piece = None
    dragging_from = None
    mouse_pos = None
    
    clock = pygame.time.Clock()
    
    while True:
        # A. AI TURN
        if not board.is_game_over() and board.turn == chess.BLACK and not dragging:
            move = get_ai_move_fast(board, model)
            board.push(move)
            print(f"AI played: {move}")

        # B. EVENT HANDLING (User clicks and drags)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            
            # Handle drag and drop for Human (White)
            if board.turn == chess.WHITE and not board.is_game_over():
                if event.type == pygame.MOUSEBUTTONDOWN:
                    # Start dragging
                    square = get_square_from_mouse(event.pos)
                    if square is not None:
                        piece = board.piece_at(square)
                        # Only allow dragging white pieces on white's turn
                        if piece and piece.color == chess.WHITE:
                            dragging = True
                            dragging_piece = piece
                            dragging_from = square
                            mouse_pos = event.pos
                
                elif event.type == pygame.MOUSEMOTION:
                    # Update mouse position while dragging
                    if dragging:
                        mouse_pos = event.pos
                
                elif event.type == pygame.MOUSEBUTTONUP:
                    # Drop the piece
                    if dragging:
                        drop_square = get_square_from_mouse(event.pos)
                        
                        if drop_square is not None and dragging_from is not None:
                            # Try to create a move from dragging_from to drop_square
                            move = chess.Move(dragging_from, drop_square)
                            
                            # Check for pawn promotion
                            if dragging_piece.piece_type == chess.PAWN:
                                # If pawn reaches the last rank, promote to queen
                                rank = chess.square_rank(drop_square)
                                if rank == 7:  # White pawn reaching rank 8
                                    move = chess.Move(dragging_from, drop_square, promotion=chess.QUEEN)
                            
                            # Check if this move is legal
                            if move in board.legal_moves:
                                board.push(move)
                                print(f"Player played: {move}")
                            else:
                                print(f"Illegal move attempted: {move}")
                        
                        # Reset dragging state
                        dragging = False
                        dragging_piece = None
                        dragging_from = None
                        mouse_pos = None

        # C. DRAWING
        # Draw the board once
        draw_board(screen)
        
        # Highlight the square we're dragging from
        if dragging and dragging_from is not None:
            highlight_square(screen, dragging_from, pygame.Color(255, 255, 0))  # Yellow highlight
            
            # Highlight valid move destinations
            for move in board.legal_moves:
                if move.from_square == dragging_from:
                    highlight_square(screen, move.to_square, pygame.Color(0, 255, 0))  # Green highlight
        
        # Draw pieces on top of the board
        draw_pieces(screen, board, images, dragging_piece, dragging_from, mouse_pos)
        
        # Display game over message
        if board.is_game_over():
            font = pygame.font.Font(None, 74)
            if board.is_checkmate():
                winner = "Black" if board.turn == chess.WHITE else "White"
                text = font.render(f"{winner} Wins!", True, pygame.Color(255, 0, 0))
            else:
                text = font.render("Draw!", True, pygame.Color(255, 0, 0))
            text_rect = text.get_rect(center=(BOARD_SIZE // 2, BOARD_SIZE // 2))
            screen.blit(text, text_rect)
        
        pygame.display.flip()
        clock.tick(60)  # 60 FPS

if __name__ == "__main__":
    main()
