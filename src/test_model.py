import torch
import chess
from model import NN
from utils import board_to_19_tensor
import numpy as np

# Configuration
MODEL_PATH = "models/glassfish_v1_2.pth"
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def get_nn_evaluation(fen, model):
    board = chess.Board(fen)
    tensor = board_to_19_tensor(board).unsqueeze(0).to(DEVICE) # Add batch dimension
    
    model.eval()
    with torch.no_grad():
        output = model(tensor)
        # The output is tanh-normalized [-1, 1]
        val = output.item()
    return val

def run_check():
    # 1. Load Model
    model = NN().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    print(f"✅ Loaded weights from {MODEL_PATH}")
    print("-" * 50)

    # 2. Test FENs
    test_cases = [
        ("Starting Position", chess.STARTING_FEN),
        ("Complex Middlegame (Equal)", "r1bq1rk1/pp2bppp/2nppn2/8/2PNP3/2N5/PP2BPPP/R1BQ1RK1 w - - 0 1"),

        # 1. White up a Knight (Black is missing the c6 Knight from the middlegame)
        ("White up Knight (Middlegame)", "r1bq1rk1/pp2bppp/2pp1n2/8/2PNP3/2N5/PP2BPPP/R1BQ1RK1 w - - 0 1"),
        
        # 2. Black up a Knight (White is missing the d4 Knight from the middlegame)
        ("Black up Knight (Middlegame)", "r1bq1rk1/pp2bppp/2nppn2/8/2P1P3/2N5/PP2BPPP/R1BQ1RK1 b - - 0 1"),

        # 3. White up a Queen (Black is missing the d8 Queen from the middlegame)
        ("White up Queen (Middlegame)", "r1b2rk1/pp2bppp/2nppn2/8/2PNP3/2N5/PP2BPPP/R1BQ1RK1 w - - 0 1"),
        
        # 4. Black up a Queen (White is missing the d1 Queen from the middlegame)
        ("Black up Queen (Middlegame)", "r1bq1rk1/pp2bppp/2nppn2/8/2PNP3/2N5/PP2BPPP/R1B2RK1 b - - 0 1"),

        # Endgames
        ("K+R vs K (White Winning)", "4k3/8/8/8/8/2K5/2R5/8 w - - 0 1"),
        ("K+R vs K (Black Winning)", "8/2r5/2k5/8/8/8/8/4K3 b - - 0 1")
    ]

    for label, fen in test_cases:
        prediction = get_nn_evaluation(fen, model)
        
        # Convert prediction back to rough "Centipawns" for Stockfish comparison
        # Since we used tanh(cp/400), we inverse it: cp = 400 * arctanh(val)
        # Note: arctanh is only stable if val is not exactly 1 or -1
        try:
            est_cp = 400 * np.arctanh(np.clip(prediction, -0.99, 0.99))
        except:
            est_cp = 1000 if prediction > 0 else -1000

        print(f"Position: {label}")
        print(f"FEN: {fen}")
        print(f"NN Output ([-1, 1]): {prediction:.4f}")
        print(f"Estimated CP: {est_cp:+.0f}")
        print("-" * 50)

if __name__ == "__main__":
    run_check()