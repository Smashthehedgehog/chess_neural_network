import chess
import numpy as np
import tensorflow as tf
import pygame
import argparse

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

def get_ai_move_fast(board, model, temperature=1.0):
    """
    Get AI move with temperature-based sampling.
    
    Args:
        board: Current chess board state
        model: AI model to use for evaluation
        temperature: Controls randomness (higher = more random, lower = more deterministic)
                     temperature=0.0 means always pick the best move (deterministic)
                     temperature=1.0 means sample proportionally to scores
                     temperature>1.0 means more exploration/randomness
    """
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None

    # 1. Build a list of ALL possible next board tensors
    tensors = []
    for move in legal_moves:
        board.push(move)
        tensors.append(create_tensor(board))
        board.pop()

    # 2. Convert list to a single 4D NumPy array (Batch, 8, 8, 12)
    batch_input = np.array(tensors)

    # 3. Predict ALL scores at once
    predictions = model.predict(batch_input, verbose=0).flatten()

    # 4. Apply temperature-based sampling
    if temperature == 0.0:
        # Deterministic: always pick the best move
        if board.turn == chess.WHITE:
            best_move_idx = np.argmax(predictions)
        else:
            best_move_idx = np.argmin(predictions)
    else:
        # Temperature-based sampling
        if board.turn == chess.WHITE:
            # White wants higher scores
            scores = predictions
        else:
            # Black wants lower scores, so invert
            scores = -predictions
        
        # Apply temperature scaling
        scaled_scores = scores / temperature
        
        # Convert to probabilities using softmax
        exp_scores = np.exp(scaled_scores - np.max(scaled_scores))  # Subtract max for numerical stability
        probabilities = exp_scores / np.sum(exp_scores)
        
        # Sample a move based on probabilities
        best_move_idx = np.random.choice(len(legal_moves), p=probabilities)

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

def play_human_vs_human():
    """Human vs Human game mode with GUI."""
    pygame.init()
    screen = pygame.display.set_mode((BOARD_SIZE, BOARD_SIZE))
    pygame.display.set_caption("Chess Game - Human vs Human")
    images = load_images()
    board = chess.Board()
    
    # Drag and drop state
    dragging = False
    dragging_piece = None
    dragging_from = None
    mouse_pos = None
    
    clock = pygame.time.Clock()
    
    while True:
        # EVENT HANDLING (User clicks and drags)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            
            # Handle drag and drop for both players
            if not board.is_game_over():
                if event.type == pygame.MOUSEBUTTONDOWN:
                    # Start dragging
                    square = get_square_from_mouse(event.pos)
                    if square is not None:
                        piece = board.piece_at(square)
                        # Only allow dragging pieces of the current player's color
                        if piece and piece.color == board.turn:
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
                                if (rank == 7 and dragging_piece.color == chess.WHITE) or \
                                   (rank == 0 and dragging_piece.color == chess.BLACK):
                                    move = chess.Move(dragging_from, drop_square, promotion=chess.QUEEN)
                            
                            # Check if this move is legal
                            if move in board.legal_moves:
                                board.push(move)
                                print(f"{'White' if board.turn == chess.BLACK else 'Black'} played: {move}")
                        
                        # Reset dragging state
                        dragging = False
                        dragging_piece = None
                        dragging_from = None
                        mouse_pos = None

        # DRAWING
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

def play_human_vs_ai(ai_model_path, temperature=0.5):
    """Human vs AI game mode with GUI."""
    pygame.init()
    screen = pygame.display.set_mode((BOARD_SIZE, BOARD_SIZE))
    pygame.display.set_caption("Chess Game - Human vs AI")
    images = load_images()
    board = chess.Board()
    
    print(f"Loading AI model from: {ai_model_path}")
    model = tf.keras.models.load_model(ai_model_path)
    
    # Drag and drop state
    dragging = False
    dragging_piece = None
    dragging_from = None
    mouse_pos = None
    
    clock = pygame.time.Clock()
    
    while True:
        # AI TURN (Black)
        if not board.is_game_over() and board.turn == chess.BLACK and not dragging:
            print("AI is thinking...")
            move = get_ai_move_fast(board, model, temperature=temperature)
            board.push(move)
            print(f"AI played: {move}")

        # EVENT HANDLING (User clicks and drags)
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
                        
                        # Reset dragging state
                        dragging = False
                        dragging_piece = None
                        dragging_from = None
                        mouse_pos = None

        # DRAWING
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

def play_ai_vs_ai(model1_path, model2_path, model1_name, model2_name, num_games=1000, temperature=1.0):
    """AI vs AI game mode - plays multiple games without GUI and reports results."""
    print(f"Loading {model1_name} from: {model1_path}")
    model1 = tf.keras.models.load_model(model1_path)
    
    print(f"Loading {model2_name} from: {model2_path}")
    model2 = tf.keras.models.load_model(model2_path)
    
    # Track results: model1 plays White, model2 plays Black
    wins_model1 = 0
    wins_model2 = 0
    draws = 0
    
    print(f"\nStarting {num_games} games: {model1_name} (White) vs {model2_name} (Black)")
    print(f"Temperature: {temperature}")
    print("=" * 60)
    
    for game_num in range(1, num_games + 1):
        board = chess.Board()
        move_count = 0
        max_moves = 200  # Prevent infinite games
        
        while not board.is_game_over() and move_count < max_moves:
            if board.turn == chess.WHITE:
                # Model 1's turn (White)
                move = get_ai_move_fast(board, model1, temperature=temperature)
            else:
                # Model 2's turn (Black)
                move = get_ai_move_fast(board, model2, temperature=temperature)
            
            if move is None:
                break
            
            board.push(move)
            move_count += 1
        
        # Determine the result
        if board.is_checkmate():
            if board.turn == chess.BLACK:
                # White won (model1)
                wins_model1 += 1
                result = f"{model1_name} wins"
            else:
                # Black won (model2)
                wins_model2 += 1
                result = f"{model2_name} wins"
        else:
            # Draw (stalemate, insufficient material, or max moves reached)
            draws += 1
            result = "Draw"
        
        # Print progress every 100 games
        if game_num % 100 == 0:
            print(f"Game {game_num}/{num_games} complete - {result}")
            print(f"  Current score: {model1_name} {wins_model1} - {draws} - {wins_model2} {model2_name}")
    
    # Print final results
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"{model1_name} (White): {wins_model1} wins ({wins_model1/num_games*100:.1f}%)")
    print(f"Draws: {draws} ({draws/num_games*100:.1f}%)")
    print(f"{model2_name} (Black): {wins_model2} wins ({wins_model2/num_games*100:.1f}%)")
    print("=" * 60)
    print(f"Win-Draw-Loss: {wins_model1}-{draws}-{wins_model2}")

def main():
    parser = argparse.ArgumentParser(description="Chess Game with AI")
    parser.add_argument(
        '--mode',
        type=str,
        choices=['human', 'ai', 'ai-vs-ai'],
        required=True,
        help='Game mode: "human" (human vs human), "ai" (human vs AI), or "ai-vs-ai" (AI vs AI)'
    )
    parser.add_argument(
        '--model',
        type=str,
        help='Path to AI model file (required for "ai" mode)'
    )
    parser.add_argument(
        '--model1',
        type=str,
        help='Path to first AI model file (required for "ai-vs-ai" mode)'
    )
    parser.add_argument(
        '--model2',
        type=str,
        help='Path to second AI model file (required for "ai-vs-ai" mode)'
    )
    parser.add_argument(
        '--name1',
        type=str,
        default='Model 1',
        help='Name for first AI model (for "ai-vs-ai" mode, default: "Model 1")'
    )
    parser.add_argument(
        '--name2',
        type=str,
        default='Model 2',
        help='Name for second AI model (for "ai-vs-ai" mode, default: "Model 2")'
    )
    parser.add_argument(
        '--games',
        type=int,
        default=1000,
        help='Number of games to play in "ai-vs-ai" mode (default: 1000)'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=1.0,
        help='Temperature for move sampling (default: 1.0). Higher = more random, lower = more deterministic. Use 0.0 for purely best moves.'
    )
    
    args = parser.parse_args()
    
    # Validate arguments based on mode
    if args.mode == 'human':
        play_human_vs_human()
    
    elif args.mode == 'ai':
        if not args.model:
            parser.error('--model is required for "ai" mode')
        play_human_vs_ai(args.model, temperature=args.temperature)
    
    elif args.mode == 'ai-vs-ai':
        if not args.model1 or not args.model2:
            parser.error('--model1 and --model2 are required for "ai-vs-ai" mode')
        play_ai_vs_ai(args.model1, args.model2, args.name1, args.name2, args.games, temperature=args.temperature)

if __name__ == "__main__":
    main()
