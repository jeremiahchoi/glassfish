import torch
import chess
import numpy as np

def get_piece_planes(board):
    planes = np.zeros((12, 8, 8), dtype=np.float32)
    piece_map = {chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2, 
                 chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5}
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            color_offset = 0 if piece.color == chess.WHITE else 6
            # Standard row/col mapping for an 8x8 grid
            row, col = divmod(square, 8)
            planes[piece_map[piece.piece_type] + color_offset][row][col] = 1.0
    return planes

def board_to_19_tensor(board):
    # 1. Piece Planes (12 channels)
    piece_stack = get_piece_planes(board)

    # 2. Meta Planes (7 channels)
    # Side to move
    turn = np.full((1, 8, 8), 1.0 if board.turn == chess.WHITE else 0.0, dtype=np.float32)
    
    # Castling Rights
    castle = np.zeros((4, 8, 8), dtype=np.float32)
    castle[0] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0 
    castle[1] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
    castle[2] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
    castle[3] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0
    
    # En Passant
    ep = np.zeros((1, 8, 8), dtype=np.float32)
    if board.ep_square:
        row, col = divmod(board.ep_square, 8)
        ep[0][row][col] = 1.0
        
    # 50-move rule progress
    progress = np.full((1, 8, 8), board.halfmove_clock / 100.0, dtype=np.float32)
    
    # Final Concatenation: 12 + 1 + 4 + 1 + 1 = 19 planes
    full_stack = np.concatenate([piece_stack, turn, castle, ep, progress], axis=0)
    
    # Return as [Batch, Channels, H, W]
    return torch.from_numpy(full_stack)