"""
Chess Model Learner - Streaming Data Processing for Large Lichess Databases

This script processes large Lichess chess game databases efficiently by:
1. Streaming decompression (no temporary files)
2. Incremental processing and saving
3. Minimal storage footprint

Designed for Google Colab with limited storage (~100GB).
"""

# ============================================================================
# CELL 1: Imports and Dependencies
# ============================================================================

import chess
import chess.pgn
import numpy as np
import urllib.request
import zstandard as zstd
import io
import os
import sys
from pathlib import Path

print("Dependencies loaded successfully!")
print(f"Python version: {sys.version}")
print(f"NumPy version: {np.__version__}")
print(f"python-chess version: {chess.__version__}")


# ============================================================================
# CELL 2: Helper Functions - Board Tensor Creation
# ============================================================================

def create_tensor(chess_board):
    """
    Function to create a tensor based on the current layout of the chess pieces to feed into an ML Model:
    input: Board object
    output: numpy array of size (8,8,12)
    """
    chess_tensor = np.zeros((8, 8, 12))

    # Define (layer index) -> (piece_type, color) mapping
    layer_to_piece_type = {}
    for color in [chess.WHITE, chess.BLACK]:
        for piece_type in range(1, 7):  # chess.PAWN==1, chess.KING==6
            if color == chess.WHITE:
                layer = piece_type - 1  # 0-indexed layers for white
            else:
                layer = piece_type - 1 + 6  # 6-11 for black
            layer_to_piece_type[layer] = (piece_type, color)

    # Fill in the tensors with the state of each tile in the chess board
    for i in range(12):  # piece layers
        piece_type, color = layer_to_piece_type[i]
        piece_pos = set(chess_board.pieces(piece_type, color))

        tracker = 0
        for j in range(7, -1, -1):  # Ranks (dim2)
            for k in range(8):      # Files (dim1)
                if tracker in piece_pos:
                    chess_tensor[j][k][i] = 1 
                tracker += 1

    return chess_tensor


# ============================================================================
# CELL 3: Streaming Data Processing Functions
# ============================================================================

def append_to_npy(filepath, new_data):
    """
    Append new data to an existing .npy file, or create it if it doesn't exist.
    """
    if os.path.exists(filepath):
        existing_data = np.load(filepath)
        combined_data = np.concatenate([existing_data, new_data])
        np.save(filepath, combined_data)
        print(f"  Appended {len(new_data)} samples to {filepath} (total: {len(combined_data)})")
    else:
        np.save(filepath, new_data)
        print(f"  Created {filepath} with {len(new_data)} samples")


def process_games_from_stream(file_stream, num_games=None, labeling_method='outcome'):
    """
    Process chess games from a file stream and return training data.
    
    Args:
        file_stream: Open file handle to read PGN data from
        num_games: Maximum number of games to process (None = all)
        labeling_method: 'outcome' (1/-1/0 based on game result) or 
                        'checks' (1/-1/0 based on checks/checkmates)
    
    Returns:
        X: numpy array of board tensors (N, 8, 8, 12)
        y: numpy array of labels (N,)
    """
    game_count = 0
    X_list = []
    y_list = []
    
    print(f"Processing games with labeling method: {labeling_method}")

    while True:
        if num_games is not None and game_count >= num_games:
            break
        
        game = chess.pgn.read_game(file_stream)
        if game is None:
            break  # End of stream

        if labeling_method == 'outcome':
            # Label based on game outcome
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
        
        elif labeling_method == 'checks':
            # Label based on checks/checkmates
            board = game.board()
            game_boards = []
            game_labels = []
            current_label = 0
            
            for move in game.mainline_moves():
                board.push(move)
                game_boards.append(create_tensor(board))
                
                if board.is_check() or board.is_checkmate():
                    if board.turn == chess.BLACK:
                        current_label = 1  # White delivered check
                    else:
                        current_label = -1  # Black delivered check
                
                game_labels.append(current_label)
            
            # Add all positions from this game
            X_list.extend(game_boards)
            y_list.extend(game_labels)

        game_count += 1
        
        # Progress update every 1000 games
        if game_count % 1000 == 0:
            print(f"  Processed {game_count} games, {len(X_list)} positions...")

    X = np.array(X_list)
    y = np.array(y_list)
    
    print(f"Completed: {game_count} games, {len(X)} total positions")
    return X, y


def process_lichess_file_streaming(url, month_name, output_dir='./data', 
                                   num_games=None, labeling_method='outcome'):
    """
    Download and process a Lichess database file with streaming decompression.
    
    Args:
        url: URL to the .pgn.zst file
        month_name: Identifier for this month (e.g., '2013-07')
        output_dir: Directory to save output files
        num_games: Maximum games to process (None = all)
        labeling_method: 'outcome' or 'checks'
    """
    print(f"\n{'='*60}")
    print(f"Processing: {month_name}")
    print(f"URL: {url}")
    print(f"{'='*60}")
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        # Download and stream decompress
        print("Downloading and streaming decompression...")
        with urllib.request.urlopen(url) as response:
            # Get file size for progress tracking
            file_size = response.headers.get('Content-Length')
            if file_size:
                print(f"Compressed file size: {int(file_size) / (1024**3):.2f} GB")
            
            # Stream decompress
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(response) as reader:
                text_stream = io.TextIOWrapper(reader, encoding='utf-8')
                
                # Process games from stream
                X, y = process_games_from_stream(text_stream, num_games, labeling_method)
        
        # Append to master files
        print("\nSaving data...")
        X_path = os.path.join(output_dir, f'X_{labeling_method}.npy')
        y_path = os.path.join(output_dir, f'y_{labeling_method}.npy')
        
        append_to_npy(X_path, X)
        append_to_npy(y_path, y)
        
        print(f"\n✓ Successfully processed {month_name}")
        
        # Print statistics
        if labeling_method == 'checks':
            print(f"  Label 1 (White checks): {np.sum(y == 1)}")
            print(f"  Label 0 (No checks): {np.sum(y == 0)}")
            print(f"  Label -1 (Black checks): {np.sum(y == -1)}")
        else:
            print(f"  Label 1 (White wins): {np.sum(y == 1)}")
            print(f"  Label 0 (Draws): {np.sum(y == 0)}")
            print(f"  Label -1 (Black wins): {np.sum(y == -1)}")
        
    except Exception as e:
        print(f"\n✗ Error processing {month_name}: {e}")
        raise


# ============================================================================
# CELL 4: Configuration - Files to Process
# ============================================================================

# List of Lichess database files to process
# Start with smaller/older files for testing, then scale up
LICHESS_FILES = [
    # 2013 files (smallest - good for testing)
    ('https://database.lichess.org/standard/lichess_db_standard_rated_2013-07.pgn.zst', '2013-07'),
    ('https://database.lichess.org/standard/lichess_db_standard_rated_2013-08.pgn.zst', '2013-08'),
    ('https://database.lichess.org/standard/lichess_db_standard_rated_2013-09.pgn.zst', '2013-09'),
    
    # Add more months as needed:
    # ('https://database.lichess.org/standard/lichess_db_standard_rated_2013-10.pgn.zst', '2013-10'),
    # ('https://database.lichess.org/standard/lichess_db_standard_rated_2013-11.pgn.zst', '2013-11'),
    # ('https://database.lichess.org/standard/lichess_db_standard_rated_2013-12.pgn.zst', '2013-12'),
    
    # 2014 files
    # ('https://database.lichess.org/standard/lichess_db_standard_rated_2014-01.pgn.zst', '2014-01'),
    # ... add more as needed
]


# ============================================================================
# CELL 5: Main Processing Loop
# ============================================================================

def main():
    """
    Main function to process all configured Lichess database files.
    """
    # Configuration
    OUTPUT_DIR = './chess_training_data'
    LABELING_METHOD = 'checks'  # Options: 'outcome' or 'checks'
    MAX_GAMES_PER_FILE = None  # None = process all games, or set a limit like 50000
    
    print("="*60)
    print("LICHESS DATABASE STREAMING PROCESSOR")
    print("="*60)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Labeling method: {LABELING_METHOD}")
    print(f"Max games per file: {MAX_GAMES_PER_FILE if MAX_GAMES_PER_FILE else 'All'}")
    print(f"Total files to process: {len(LICHESS_FILES)}")
    print("="*60)
    
    # Process each file
    for idx, (url, month_name) in enumerate(LICHESS_FILES, 1):
        print(f"\n[{idx}/{len(LICHESS_FILES)}] Processing {month_name}...")
        
        try:
            process_lichess_file_streaming(
                url=url,
                month_name=month_name,
                output_dir=OUTPUT_DIR,
                num_games=MAX_GAMES_PER_FILE,
                labeling_method=LABELING_METHOD
            )
        except KeyboardInterrupt:
            print("\n\nProcessing interrupted by user.")
            break
        except Exception as e:
            print(f"\nError processing {month_name}: {e}")
            print("Continuing to next file...")
            continue
    
    # Final summary
    print("\n" + "="*60)
    print("PROCESSING COMPLETE")
    print("="*60)
    
    # Load and display final statistics
    X_path = os.path.join(OUTPUT_DIR, f'X_{LABELING_METHOD}.npy')
    y_path = os.path.join(OUTPUT_DIR, f'y_{LABELING_METHOD}.npy')
    
    if os.path.exists(X_path) and os.path.exists(y_path):
        X_final = np.load(X_path)
        y_final = np.load(y_path)
        
        print(f"Total positions: {len(X_final):,}")
        print(f"Array shape: X={X_final.shape}, y={y_final.shape}")
        print(f"Storage size: X={X_final.nbytes / (1024**3):.2f} GB, y={y_final.nbytes / (1024**2):.2f} MB")
        print("\nLabel distribution:")
        print(f"  Label  1: {np.sum(y_final == 1):,} ({np.sum(y_final == 1)/len(y_final)*100:.1f}%)")
        print(f"  Label  0: {np.sum(y_final == 0):,} ({np.sum(y_final == 0)/len(y_final)*100:.1f}%)")
        print(f"  Label -1: {np.sum(y_final == -1):,} ({np.sum(y_final == -1)/len(y_final)*100:.1f}%)")
    else:
        print("No output files found.")


# ============================================================================
# CELL 6: Utility Functions - Storage Management
# ============================================================================

def check_disk_usage(path='.'):
    """Check disk usage for the given path."""
    import shutil
    total, used, free = shutil.disk_usage(path)
    
    print("\nDisk Usage:")
    print(f"  Total: {total / (1024**3):.2f} GB")
    print(f"  Used:  {used / (1024**3):.2f} GB")
    print(f"  Free:  {free / (1024**3):.2f} GB")
    print(f"  Usage: {used/total*100:.1f}%")
    
    return free


def check_output_files(output_dir='./chess_training_data', labeling_method='checks'):
    """Check the current state of output files."""
    X_path = os.path.join(output_dir, f'X_{labeling_method}.npy')
    y_path = os.path.join(output_dir, f'y_{labeling_method}.npy')
    
    print("\nCurrent Output Files:")
    
    if os.path.exists(X_path):
        X = np.load(X_path)
        size_gb = X.nbytes / (1024**3)
        print(f"  X_{labeling_method}.npy: {len(X):,} positions, {size_gb:.2f} GB")
    else:
        print(f"  X_{labeling_method}.npy: Not found")
    
    if os.path.exists(y_path):
        y = np.load(y_path)
        size_mb = y.nbytes / (1024**2)
        print(f"  y_{labeling_method}.npy: {len(y):,} labels, {size_mb:.2f} MB")
    else:
        print(f"  y_{labeling_method}.npy: Not found")


# ============================================================================
# CELL 7: Alternative Processing - Local File Processing
# ============================================================================

def process_local_zst_file(filepath, month_name, output_dir='./data', 
                           num_games=None, labeling_method='outcome'):
    """
    Process a local .pgn.zst file (if you've already downloaded it).
    
    Args:
        filepath: Path to local .pgn.zst file
        month_name: Identifier for this month
        output_dir: Directory to save output files
        num_games: Maximum games to process
        labeling_method: 'outcome' or 'checks'
    """
    print(f"\n{'='*60}")
    print(f"Processing local file: {month_name}")
    print(f"Path: {filepath}")
    print(f"{'='*60}")
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        with open(filepath, 'rb') as compressed_file:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(compressed_file) as reader:
                text_stream = io.TextIOWrapper(reader, encoding='utf-8')
                
                # Process games
                X, y = process_games_from_stream(text_stream, num_games, labeling_method)
        
        # Save results
        X_path = os.path.join(output_dir, f'X_{labeling_method}.npy')
        y_path = os.path.join(output_dir, f'y_{labeling_method}.npy')
        
        append_to_npy(X_path, X)
        append_to_npy(y_path, y)
        
        print(f"\n✓ Successfully processed {month_name}")
        
    except Exception as e:
        print(f"\n✗ Error processing {month_name}: {e}")
        raise


# ============================================================================
# CELL 8: Google Colab Specific Setup
# ============================================================================

def setup_google_colab():
    """
    Setup function for Google Colab environment.
    Mounts Google Drive and sets output directory to Drive.
    """
    try:
        from google.colab import drive
        
        print("Mounting Google Drive...")
        drive.mount('/content/drive')
        
        output_dir = '/content/drive/MyDrive/chess_training_data'
        print(f"Output directory set to: {output_dir}")
        
        return output_dir
    except ImportError:
        print("Not running in Google Colab. Using local directory.")
        return './chess_training_data'


# ============================================================================
# CELL 9: Batch Processing with Resume Capability
# ============================================================================

def process_with_resume(files_list, output_dir='./data', labeling_method='checks',
                       max_games_per_file=None, start_from_index=0):
    """
    Process multiple files with ability to resume from a specific index.
    
    Args:
        files_list: List of (url, month_name) tuples
        output_dir: Where to save output
        labeling_method: 'outcome' or 'checks'
        max_games_per_file: Limit per file
        start_from_index: Resume from this index in files_list
    """
    print(f"Starting from file index {start_from_index}")
    
    for idx in range(start_from_index, len(files_list)):
        url, month_name = files_list[idx]
        
        print(f"\n[{idx+1}/{len(files_list)}] Processing {month_name}...")
        
        try:
            process_lichess_file_streaming(
                url=url,
                month_name=month_name,
                output_dir=output_dir,
                num_games=max_games_per_file,
                labeling_method=labeling_method
            )
            
            # Check disk space after each file
            free_space = check_disk_usage()
            if free_space < 10 * (1024**3):  # Less than 10GB free
                print("\n⚠ WARNING: Less than 10GB free space remaining!")
                print("Consider stopping or clearing space.")
        
        except KeyboardInterrupt:
            print(f"\n\nInterrupted at index {idx}.")
            print(f"To resume, call: process_with_resume(LICHESS_FILES, start_from_index={idx})")
            break
        except Exception as e:
            print(f"\nError: {e}")
            print("Continuing to next file...")


# ============================================================================
# CELL 10: Example Usage and Testing
# ============================================================================

def test_small_sample():
    """
    Test the pipeline with a small sample from one file.
    """
    print("Running test with small sample...")
    
    test_url = 'https://database.lichess.org/standard/lichess_db_standard_rated_2013-07.pgn.zst'
    
    process_lichess_file_streaming(
        url=test_url,
        month_name='2013-07-test',
        output_dir='./test_data',
        num_games=1000,  # Only 1000 games for testing
        labeling_method='checks'
    )
    
    print("\nTest complete! Check ./test_data/ for output files.")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process Lichess chess databases")
    parser.add_argument(
        '--mode',
        type=str,
        choices=['test', 'full', 'resume'],
        default='test',
        help='Processing mode: test (1000 games), full (all configured files), resume (continue from index)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./chess_training_data',
        help='Output directory for training data'
    )
    parser.add_argument(
        '--labeling',
        type=str,
        choices=['outcome', 'checks'],
        default='checks',
        help='Labeling method: outcome (game result) or checks (check-based)'
    )
    parser.add_argument(
        '--max-games',
        type=int,
        default=None,
        help='Maximum games to process per file (None = all)'
    )
    parser.add_argument(
        '--resume-from',
        type=int,
        default=0,
        help='Resume from this file index (for resume mode)'
    )
    parser.add_argument(
        '--colab',
        action='store_true',
        help='Setup for Google Colab (mount Drive)'
    )
    
    args = parser.parse_args()
    
    # Setup output directory
    if args.colab:
        output_dir = setup_google_colab()
    else:
        output_dir = args.output_dir
    
    # Execute based on mode
    if args.mode == 'test':
        print("Running in TEST mode (1000 games from first file)")
        test_url, test_month = LICHESS_FILES[0]
        process_lichess_file_streaming(
            url=test_url,
            month_name=test_month,
            output_dir=output_dir,
            num_games=1000,
            labeling_method=args.labeling
        )
    
    elif args.mode == 'full':
        print("Running in FULL mode (processing all configured files)")
        process_with_resume(
            files_list=LICHESS_FILES,
            output_dir=output_dir,
            labeling_method=args.labeling,
            max_games_per_file=args.max_games,
            start_from_index=0
        )
    
    elif args.mode == 'resume':
        print(f"Running in RESUME mode (starting from index {args.resume_from})")
        process_with_resume(
            files_list=LICHESS_FILES,
            output_dir=output_dir,
            labeling_method=args.labeling,
            max_games_per_file=args.max_games,
            start_from_index=args.resume_from
        )
    
    # Final status
    print("\n" + "="*60)
    check_output_files(output_dir, args.labeling)
    check_disk_usage()
    print("="*60)
